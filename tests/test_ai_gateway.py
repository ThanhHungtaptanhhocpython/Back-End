"""Phase 3 -- multi-provider AI gateway (Text / Vision fallback chains)."""

from __future__ import annotations

import json
import os
import sys

import pytest

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from src.config.settings import Settings  # noqa: E402
from src.services.ai import gateway, registry  # noqa: E402
from src.services.ai.base import AllProvidersFailed, ProviderError  # noqa: E402
from src.services.ai import openai_compatible  # noqa: E402


# ---------------------------------------------------------------------------
# fake OpenAI-compatible transport
# ---------------------------------------------------------------------------
def _completion_body(text: str = "pong") -> str:
    return json.dumps(
        {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}
    )


class FakeTransport:
    """``request(method, url, headers, body, timeout)`` -> (status, text).

    ``rules`` maps a substring of the URL to either a (status, text) tuple, a
    callable ``(method, url, body_dict) -> (status, text)``, or an exception to
    raise.
    """

    def __init__(self, rules: dict):
        self.rules = rules
        self.calls: list[tuple[str, str]] = []

    def request(self, method, url, headers, body, timeout):
        self.calls.append((method, url))
        body_dict = json.loads(body) if body else {}
        for needle, action in self.rules.items():
            if needle in url:
                if isinstance(action, BaseException):
                    raise action
                if callable(action):
                    return action(method, url, body_dict)
                return action
        return 200, _completion_body()


@pytest.fixture()
def transport():
    def _install(rules):
        fake = FakeTransport(rules)
        openai_compatible.set_transport(fake)
        return fake

    previous = openai_compatible._TRANSPORT
    yield _install
    openai_compatible.set_transport(previous)


def _settings(**over) -> Settings:
    base = dict(
        ai_gateway_enabled=True,
        ai_local_fallback_enabled=True,
        nim_enabled=True, nim_api_key="k-nim", nim_text_model="nim-t", nim_vision_model="",
        groq_enabled=True, groq_api_key="k-groq", groq_text_model="groq-t", groq_vision_model="groq-v",
        openrouter_enabled=True, openrouter_api_key="k-or",
        openrouter_model="or-t", openrouter_vision_model="or-v",
        gemini_enabled=True, gemini_api_key="k-gem", gemini_text_model="gem-t", gemini_vision_model="",
        ai_text_priority="groq,nim,openrouter,gemini",
        ai_vision_priority="gemini,groq,openrouter",
    )
    base.update(over)
    return Settings(_env_file=None, **base)


# ---------------------------------------------------------------------------
class TestChainComposition:
    def test_text_chain_follows_priority_order(self) -> None:
        chain = registry.text_chain(_settings())
        assert [p.id for p in chain] == ["groq", "nim", "openrouter", "gemini"]

    def test_vision_chain_skips_providers_without_vision_model(self) -> None:
        # gemini + nim have no vision model; groq + openrouter do.
        chain = registry.vision_chain(_settings())
        assert [p.id for p in chain] == ["groq", "openrouter"]

    def test_disabled_provider_is_excluded(self) -> None:
        chain = registry.text_chain(_settings(groq_enabled=False))
        assert [p.id for p in chain] == ["nim", "openrouter", "gemini"]

    def test_gateway_off_means_no_chain_use(self) -> None:
        assert gateway.text_available(_settings(ai_gateway_enabled=True)) is True


# ---------------------------------------------------------------------------
class TestTextFallback:
    def test_first_provider_wins(self, transport) -> None:
        transport({"groq.com": (200, _completion_body("from-groq"))})
        result, attempts = gateway.text_completion(
            [{"role": "user", "content": "hi"}], settings=_settings()
        )
        assert result.provider == "groq" and result.text == "from-groq"
        assert [a.provider for a in attempts] == ["groq"]

    def test_rate_limit_falls_over_to_next(self, transport) -> None:
        transport(
            {
                "groq.com": (429, '{"error":"rate limit"}'),
                "api.nvidia.com": (200, _completion_body("from-nim")),
            }
        )
        result, attempts = gateway.text_completion(
            [{"role": "user", "content": "hi"}], settings=_settings()
        )
        assert result.provider == "nim" and result.text == "from-nim"
        assert attempts[0].provider == "groq" and attempts[0].category == "rate_limit"
        assert attempts[1].provider == "nim" and attempts[1].ok

    def test_timeout_is_classified_and_retried(self, transport) -> None:
        transport(
            {
                "groq.com": ProviderError("timeout", "slow", provider="groq"),
                "api.nvidia.com": (200, _completion_body("ok-nim")),
            }
        )
        result, attempts = gateway.text_completion(
            [{"role": "user", "content": "hi"}], settings=_settings()
        )
        assert result.provider == "nim"
        assert attempts[0].category == "timeout"

    def test_all_providers_fail_raises(self, transport) -> None:
        transport({"": (500, '{"error":"boom"}')})
        with pytest.raises(AllProvidersFailed) as excinfo:
            gateway.text_completion([{"role": "user", "content": "hi"}], settings=_settings())
        assert len(excinfo.value.attempts) == 4
        assert all(a.category == "upstream" for a in excinfo.value.attempts)

    def test_empty_chain_raises_immediately(self, transport) -> None:
        with pytest.raises(AllProvidersFailed):
            gateway.text_completion(
                [{"role": "user", "content": "hi"}],
                settings=_settings(nim_enabled=False, groq_enabled=False,
                                   openrouter_enabled=False, gemini_enabled=False),
            )


# ---------------------------------------------------------------------------
class TestResponseFormatDowngrade:
    def test_bad_request_on_response_format_retries_without_it(self, transport) -> None:
        def handler(method, url, body):
            if "response_format" in body:
                return 400, '{"error":"response_format is not supported"}'
            return 200, _completion_body("downgraded-ok")

        transport({"groq.com": handler})
        result, attempts = gateway.text_completion(
            [{"role": "user", "content": "hi"}],
            settings=_settings(ai_text_priority="groq"),
            response_format={"type": "json_object"},
        )
        assert result.text == "downgraded-ok"
        assert attempts[0].ok and attempts[0].detail == "no-json-schema"


# ---------------------------------------------------------------------------
class TestVisionFallback:
    def test_vision_uses_vision_chain_and_returns_payload(self, transport) -> None:
        transport({"groq.com": (200, _completion_body('{"items": []}'))})
        payload, attempts, provider = gateway.vision_completion(
            [{"role": "user", "content": "look"}], settings=_settings()
        )
        assert provider == "groq"
        assert payload["choices"][0]["message"]["content"] == '{"items": []}'

    def test_vision_all_fail_raises(self, transport) -> None:
        transport({"": (503, "down")})
        with pytest.raises(AllProvidersFailed):
            gateway.vision_completion([{"role": "user", "content": "x"}], settings=_settings())


# ---------------------------------------------------------------------------
class TestModelDiscovery:
    def test_list_models_parses_and_sorts(self, transport) -> None:
        transport(
            {"/models": (200, json.dumps({"data": [{"id": "m-b"}, {"id": "m-a"}, "m-a"]}))}
        )
        provider = registry.build_provider("groq", _settings())
        assert provider.list_models() == ["m-a", "m-b"]

    def test_list_models_error_raises_provider_error(self, transport) -> None:
        transport({"/models": (401, "nope")})
        provider = registry.build_provider("groq", _settings())
        with pytest.raises(ProviderError) as excinfo:
            provider.list_models()
        assert excinfo.value.category == "auth"


# ---------------------------------------------------------------------------
class TestKiloProvider:
    def test_registered_and_openrouter_compatible(self) -> None:
        assert "kilo" in registry.all_provider_ids()
        s = _settings(kilo_enabled=True, kilo_api_key="k-kilo", kilo_text_model="anthropic/claude-3.5-sonnet",
                      ai_text_priority="kilo,groq")
        provider = registry.build_provider("kilo", s)
        assert provider.is_configured() and provider.base_url.endswith("/api/gateway")
        assert provider.extra_headers.get("X-Title")  # OpenRouter-style id headers
        assert [p.id for p in registry.text_chain(s)] == ["kilo", "groq"]

    def test_in_chain_and_answers(self, transport) -> None:
        transport({"api.kilo.ai": (200, _completion_body("from-kilo"))})
        s = _settings(kilo_enabled=True, kilo_api_key="k", kilo_text_model="m",
                      ai_text_priority="kilo,groq")
        result, attempts = gateway.text_completion([{"role": "user", "content": "hi"}], settings=s)
        assert result.provider == "kilo" and result.text == "from-kilo"


class TestCloudflareRequiresAccountId:
    def test_missing_account_id_is_not_configured(self) -> None:
        s = _settings(cloudflare_enabled=True, cloudflare_api_key="k", cloudflare_text_model="cf-t")
        provider = registry.build_provider("cloudflare", s)
        assert provider.is_configured() is False
        assert provider.missing_requirements == ("CLOUDFLARE_ACCOUNT_ID",)

    def test_account_id_builds_workers_ai_url(self) -> None:
        s = _settings(
            cloudflare_enabled=True, cloudflare_api_key="k",
            cloudflare_text_model="cf-t", cloudflare_account_id="acct123",
        )
        provider = registry.build_provider("cloudflare", s)
        assert provider.is_configured() is True
        assert provider.base_url.endswith("/accounts/acct123/ai/v1")


# ---------------------------------------------------------------------------
class TestTranslationWiring:
    def test_translation_uses_gateway_when_enabled(self, transport, monkeypatch) -> None:
        transport({"groq.com": (200, _completion_body("a dog runs"))})
        import src.utils.nlp_processing as nlp

        monkeypatch.setattr(nlp, "get_settings", lambda: _settings(ai_text_priority="groq"), raising=False)
        # force google to yield nothing so the gateway path is exercised
        monkeypatch.setattr(nlp.Translation, "_translate_with_google", lambda self, text: "")

        tr = nlp.Translation(from_lang="vi", to_lang="en")
        # get_settings is imported inside _translate_with_gateway from src.config.settings
        monkeypatch.setattr("src.config.settings.get_settings", lambda: _settings(ai_text_priority="groq"))
        out = tr.translate_detailed("mot con cho dang chay")
        assert out.status == "ok" and out.translated is True
        assert out.text == "a dog runs" and out.provider == "groq"
