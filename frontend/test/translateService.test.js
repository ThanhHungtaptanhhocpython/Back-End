import assert from "node:assert/strict";
import test from "node:test";

import { translateText, translateTextDetailed } from "../src/services/translateService.js";

function jsonResponse(body, { ok = true, status = 200 } = {}) {
  return { ok, status, json: async () => body };
}

test("returns a live translation when the backend succeeds", async () => {
  const result = await translateTextDetailed("người phụ nữ mặc áo hồng", "vi-en", {
    fetchImpl: async (url, init) => {
      assert.match(url, /\/users\/translate$/);
      assert.deepEqual(JSON.parse(init.body), {
        text: "người phụ nữ mặc áo hồng",
        from_lang: "vi",
        to_lang: "en",
      });
      return jsonResponse({
        success: true,
        translated_text: "the woman wearing a pink shirt",
        translated: true,
        provider: "google",
        status: "ok",
      });
    },
  });

  assert.deepEqual(result, {
    text: "the woman wearing a pink shirt",
    live: true,
    provider: "google",
    status: "ok",
  });
});

test("distinguishes an unreachable backend from an unavailable provider", async () => {
  const offline = await translateTextDetailed("xin chào", "vi-en", {
    fetchImpl: async () => {
      throw new TypeError("Failed to fetch");
    },
  });
  assert.equal(offline.live, false);
  assert.equal(offline.status, "backend_unreachable");
  assert.equal(offline.text, "xin chào"); // original query kept

  const providerDown = await translateTextDetailed("xin chào", "vi-en", {
    fetchImpl: async () =>
      jsonResponse(
        {
          success: false,
          translated_text: "xin chào",
          translated: false,
          provider: "none",
          status: "provider_unavailable",
          error_code: "provider_unavailable",
        },
        { ok: false, status: 503 },
      ),
  });
  assert.equal(providerDown.live, false);
  assert.equal(providerDown.status, "provider_unavailable");
  assert.equal(providerDown.text, "xin chào");
});

test("a 200 identity response for a cross-language request is not a success", async () => {
  // Legacy backend: 200 + translated:false + provider:identity, no status field.
  const result = await translateTextDetailed("nguoi phu nu mac ao hong", "vi-en", {
    fetchImpl: async () =>
      jsonResponse({
        success: true,
        translated_text: "nguoi phu nu mac ao hong",
        translated: false,
        provider: "identity",
      }),
  });

  assert.equal(result.live, false);
  assert.equal(result.status, "provider_unavailable");
  assert.equal(result.text, "nguoi phu nu mac ao hong");
});

test("maps a 400 to invalid_input and keeps the trimmed original text", async () => {
  const result = await translateTextDetailed("   x   ", "vi-en", {
    fetchImpl: async () =>
      jsonResponse(
        {
          success: false,
          translated_text: "",
          translated: false,
          provider: "none",
          status: "invalid_input",
          error_code: "invalid_input",
        },
        { ok: false, status: 400 },
      ),
  });

  assert.equal(result.status, "invalid_input");
  assert.equal(result.live, false);
  assert.equal(result.text, "x");
});

test("empty input short-circuits without a request", async () => {
  let called = false;
  const result = await translateTextDetailed("   ", "vi-en", {
    fetchImpl: async () => {
      called = true;
      return jsonResponse({});
    },
  });

  assert.equal(called, false);
  assert.deepEqual(result, { text: "", live: false, provider: "none", status: "empty" });
});

test("translateText stays a string-returning API", async () => {
  const good = await translateText("người phụ nữ mặc áo hồng", "vi-en", {
    fetchImpl: async () =>
      jsonResponse({
        success: true,
        translated_text: "the woman wearing a pink shirt",
        translated: true,
        provider: "google",
        status: "ok",
      }),
  });
  assert.equal(good, "the woman wearing a pink shirt");

  const bad = await translateText("xin chào", "vi-en", {
    fetchImpl: async () => {
      throw new TypeError("Failed to fetch");
    },
  });
  assert.equal(bad, "xin chào"); // falls back to the original, still a string
});
