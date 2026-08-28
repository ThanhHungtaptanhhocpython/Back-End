import os
import sys

import pytest

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)
from src.config.settings import get_settings
from src.services.openrouter_agent_planner import _normalise_llm_plan
from src.services.agent_query_coordinator import build_agent_plan, _rerank_with_light_verifier


@pytest.fixture(autouse=True)
def disable_real_agent_llm(monkeypatch):
    monkeypatch.setenv("AGENT_LLM_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_cycling_finish_agent_plan_builds_structured_checks():
    query = (
        "C\u1ea3nh quay ch\u1eadm t\u1ea1i v\u1ecb tr\u00ed v\u1ea1ch \u0111\u00edch c\u1ee7a cu\u1ed9c \u0111ua xe \u0111\u1ea1p. "
        "G\u00f3c m\u00e1y s\u00e1t m\u1eb7t \u0111\u01b0\u1eddng, 1 tay \u0111ua \u00e1o v\u00e0ng qu\u1ea7n \u0111en, "
        "1 tay \u0111ua \u00e1o xanh d\u01b0\u01a1ng qu\u1ea7n \u0111en v\u00e0 1 tay \u0111ua \u00e1o xanh d\u01b0\u01a1ng qu\u1ea7n \u0111\u1ecf"
    )

    plan = build_agent_plan(query, topk=20)

    assert plan["precision_profile"] == "cycling_finish_low_angle"
    assert plan["visual_query"] == "road level low angle slow motion shot of three cyclists crossing the finish line with yellow and blue jerseys"
    assert any(check["id"] == "low_road_angle" for check in plan["must_have_checks"])
    assert any(check["id"] == "blue_red" for check in plan["must_have_checks"])
    assert any("close up bicycle wheels" in query for query in plan["visual_queries"])


def test_agent_plan_executes_primary_visual_query_by_default():
    query = (
        "Canh o tram xang, co 4 tai xe xe om cong nghe. "
        "3 nguoi dung cho, 1 nguoi chay xe tu trai sang phai. "
        "Co bang gia xang dau trong khung hinh."
    )

    plan = build_agent_plan(query, topk=20)

    assert len(plan["visual_queries"]) > 1
    assert plan["executed_visual_queries"] == [plan["visual_query"]]
    assert plan["support_visual_queries"] == plan["visual_queries"][1:]
    assert plan["execution_strategy"]["mode"] == "primary_holistic_first"


def test_agent_visual_query_limit_can_be_raised(monkeypatch):
    monkeypatch.setenv("AGENT_VISUAL_QUERY_LIMIT", "2")
    get_settings.cache_clear()

    plan = build_agent_plan(
        "Canh o tram xang co 4 tai xe xe om cong nghe va bang gia xang dau",
        topk=20,
    )

    assert plan["executed_visual_queries"] == plan["visual_queries"][:2]
    assert plan["support_visual_queries"] == plan["visual_queries"][2:]



def test_generic_agent_plan_extracts_subject_action_object_and_appearance():
    plan = build_agent_plan("Mot nguoi phu nu ao do cam o di qua duong", topk=20)

    assert plan["precision_profile"] == "generic_action_appearance_count_object_subject"
    assert plan["visual_queries"] == ["woman person wearing a red shirt crossing the street holding an umbrella"]
    assert any(check["query_en"] == "person wearing a red shirt" for check in plan["must_have_checks"])
    assert "nguoi" not in plan["visual_query"]


def test_generic_agent_plan_routes_visible_text_to_ocr():
    plan = build_agent_plan("Khung hinh co chu TON DONG A tren bang hieu", topk=20)

    assert plan["ocr_query"] == "TON DONG A"
    assert plan["routing"]["ocr"] > 0
    assert any(query["kind"] == "ocr" and query["query_en"] == "TON DONG A" for query in plan["expanded_queries"])
    assert any(check["id"] == "ocr_ton_dong_a" for check in plan["must_have_checks"])


def test_generic_agent_plan_routes_speech_to_asr():
    plan = build_agent_plan("Nghe tieng phat bieu trong mot buoi phong van", topk=20)

    assert plan["asr_query"]
    assert plan["routing"]["asr"] > 0
    assert any(query["kind"] == "asr" for query in plan["expanded_queries"])
    assert any(check["id"] == "speech_audio" for check in plan["must_have_checks"])


def test_light_verifier_adds_temporal_neighbors_without_vlm():
    plan = build_agent_plan("Mot nguoi phu nu ao do cam o di qua duong", topk=5)
    frames = [
        {
            "global_frame_id": "seed",
            "video_id": "V1",
            "frame_id": "0010",
            "frame_name": "V1_0010",
            "timestamp": 10.0,
            "agent_score": 1.0,
            "agent_queries": ["woman crossing the street", "person wearing a red shirt"],
            "agent_checklist_coverage": 0.8,
            "agent_matched_checks": ["woman", "person wearing a red shirt"],
            "agent_missing_checks": ["holding an umbrella"],
            "score": 0.9,
        }
    ]

    def neighbor_provider(video_id, around_frame_id, limit):
        assert video_id == "V1"
        assert around_frame_id == "0010"
        return [
            {"global_frame_id": "prev", "video_id": "V1", "frame_id": "0009", "frame_name": "V1_0009", "timestamp": 9.0},
            {"global_frame_id": "seed", "video_id": "V1", "frame_id": "0010", "frame_name": "V1_0010", "timestamp": 10.0},
            {"global_frame_id": "next", "video_id": "V1", "frame_id": "0011", "frame_name": "V1_0011", "timestamp": 11.0},
        ]

    verified, summary = _rerank_with_light_verifier(frames, plan, topk=3, neighbor_provider=neighbor_provider)

    assert summary["method"] == "light_no_vlm"
    assert summary["temporal_neighbors"] == 3
    assert verified[0]["global_frame_id"] == "seed"
    assert any(item["global_frame_id"] == "next" for item in verified)
    neighbor = next(item for item in verified if item["global_frame_id"] == "next")
    assert "temporal_neighbor" in neighbor["agent_verification"]["sources"]
    assert neighbor["agent_verification"]["note"].startswith("No VLM verification")


def test_light_verifier_boosts_candidates_with_ocr_evidence():
    plan = {
        "original_query": "Khung hinh co chu TON DONG A tren bang hieu",
        "visual_queries": ["signboard with text TON DONG A"],
        "executed_visual_queries": ["signboard with text TON DONG A"],
        "ocr_query": "TON DONG A",
        "asr_query": "",
        "routing": {"visual": 0.55, "ocr": 0.45, "asr": 0.0},
        "must_have_checks": [{"id": "text", "label": "visible text TON DONG A", "query_en": "TON DONG A", "weight": 1.0}],
    }
    frames = [
        {
            "global_frame_id": "visual_only",
            "video_id": "V1",
            "frame_id": "0010",
            "timestamp": 10.0,
            "agent_score": 0.7,
            "agent_queries": ["signboard with text TON DONG A"],
            "score_breakdown": {"visual": 0.7, "ocr": 0.0, "asr": 0.0},
        },
        {
            "global_frame_id": "with_ocr",
            "video_id": "V2",
            "frame_id": "0020",
            "timestamp": 20.0,
            "agent_score": 0.65,
            "agent_queries": ["signboard with text TON DONG A"],
            "score_breakdown": {"visual": 0.65, "ocr": 0.95, "asr": 0.0},
            "ocr_text": "TON DONG A",
        },
    ]

    verified, _summary = _rerank_with_light_verifier(frames, plan, topk=2, neighbor_provider=lambda *_args: [])

    assert verified[0]["global_frame_id"] == "with_ocr"
    assert verified[0]["agent_verification"]["modality_evidence_score"] > 0.0
    assert "OCR: TON DONG A" in verified[0]["agent_matched_checks"]


def test_gas_station_motorbike_taxi_plan_uses_domain_queries_not_asr():
    query = (
        "Canh o tram xang, co 4 tai xe xe om cong nghe. "
        "3 nguoi dung cho, 1 nguoi chay xe tu trai sang phai. "
        "Co bang gia xang dau trong khung hinh."
    )

    plan = build_agent_plan(query, topk=20)

    assert plan["precision_profile"] == "gas_station_motorbike_taxi"
    assert plan["routing"] == {"visual": 1.0, "ocr": 0.0, "asr": 0.0}
    assert plan["visual_query"] == "gas station with four app-based motorbike taxi drivers, three waiting and one riding left to right"
    assert "fuel price board at a gas station with motorbike taxi drivers and motorbikes" in plan["visual_queries"]
    assert any(check["id"] == "fuel_price_board" for check in plan["must_have_checks"])
    assert any(check["id"] == "left_to_right" for check in plan["must_have_checks"])


def test_fishing_net_temporal_plan_prioritizes_specific_sequence_terms():
    query = (
        "Mot nguoi dung duoi nuoc va roi den. "
        "Tiep theo la canh nguoi nay keo luoi ca luc binh minh, "
        "sau do duoc mot nhom nguoi khac tien den dung may quay ghi hinh."
    )

    plan = build_agent_plan(query, topk=20)

    assert plan["precision_profile"] == "fishing_net_temporal"
    assert plan["routing"] == {"visual": 1.0, "ocr": 0.0, "asr": 0.0}
    assert plan["visual_query"] == "person standing in shallow water shining a flashlight then pulling a fishing net at dawn"
    assert "dawn fishing scene with a person pulling a net in shallow water while people approach with a video camera" in plan["visual_queries"]
    assert "camera crew filming a fisherman pulling a fishing net at sunrise" in plan["visual_queries"]
    assert any(check["id"] == "pulling_fishing_net" for check in plan["must_have_checks"])
    assert any(check["id"] == "camera_crew" for check in plan["must_have_checks"])


def test_agent_plan_can_use_openrouter_enriched_plan_once(monkeypatch):
    from src.services import openrouter_agent_planner

    calls = []

    def fake_openrouter_plan(prompt, local_plan):
        calls.append((prompt, local_plan.get("profile")))
        return {
            "profile": "llm_enriched",
            "planner_source": "openrouter",
            "intent": "Find the exact gas station scene.",
            "visual_queries": [
                "gas station scene with four motorbike taxi drivers, three waiting, one rider moving left to right, and a fuel price board"
            ],
            "ocr_queries": [],
            "asr_queries": [],
            "must_have_checks": [
                {"id": "gas_station", "label": "gas station", "query_en": "gas station", "weight": 1.2},
                {"id": "fuel_price_board", "label": "fuel price board", "query_en": "fuel price board", "weight": 1.0},
            ],
            "negative_checks": [],
            "rerank_focus": ["prefer all requested details in one frame"],
            "local_fallback_profile": local_plan.get("profile"),
        }

    monkeypatch.setattr(openrouter_agent_planner, "plan_agent_query_with_openrouter", fake_openrouter_plan)

    plan = build_agent_plan("Canh o tram xang co 4 tai xe xe om cong nghe va bang gia xang dau", topk=20)

    assert len(calls) == 1
    assert calls[0][1] == "gas_station_motorbike_taxi"
    assert plan["planner_source"] == "openrouter"
    assert plan["visual_query"].startswith("gas station scene with four motorbike taxi drivers")
    assert plan["search_plan"]["local_fallback_profile"] == "gas_station_motorbike_taxi"

def test_openrouter_planner_does_not_turn_pumpkin_into_red_lion():
    payload = {
        "profile": "llm_enriched",
        "visual_queries": [
            "a red lion dance performed by two people standing upright and spinning on top of a pole, then grabbing a pumpkin with a yellow flower"
        ],
        "must_have_checks": [
            {"id": "lion", "label": "red lion dance", "query_en": "red lion dance on top of poles", "weight": 1.0},
            {"id": "pumpkin", "label": "pumpkin with yellow flower", "query_en": "pumpkin with a yellow flower", "weight": 1.0},
        ],
    }
    prompt = (
        "Con lân do hai người điều khiển đang đứng thẳng và xoay vòng trên đỉnh cột. "
        "Sau vài giây nghỉ, con lân nhảy qua hai chiếc cột kế bên, "
        "chúi đầu xuống ngoạm lấy quả bí đỏ kèm bông hoa màu vàng."
    )

    plan = _normalise_llm_plan(payload, prompt, {})

    assert "red lion" not in plan["visual_queries"][0].lower()
    assert "lion dance" in plan["visual_queries"][0].lower()
    assert any(check["query_en"] == "lion dance on top of poles" for check in plan["must_have_checks"])


def test_openrouter_planner_keeps_explicit_red_lion():
    payload = {
        "profile": "llm_enriched",
        "visual_queries": ["a red lion dance costume jumping between poles during a performance"],
    }
    prompt = "Một con lân màu đỏ đang nhảy trên các cột."

    plan = _normalise_llm_plan(payload, prompt, {})

    assert "red lion" in plan["visual_queries"][0].lower()
