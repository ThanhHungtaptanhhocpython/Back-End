import base64
import csv
import math
import os
import re
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from src.config.settings import get_settings
from src.services.reranker_service import reranker_service
from src.utils.nlp_processing import QueryPlanner, Translation

'''
"global_frame_id": int17
"video_id": string"V001"
"frame_name": string"keyframe_L21_V001_0001.webp"
"frame_index": int0
"split": string"videos-l21-a"
'''


logger = logging.getLogger(__name__)


class TRAKE:
    def __init__(self, faiss_searcher: Any = None):
        """Initialize ordered-event retrieval on the active BEiT3 corpus.

        ``faiss_searcher`` remains optional only for backwards compatibility;
        TRAKE no longer loads or queries the legacy CLIP index.
        """
        settings = get_settings()
        self.faiss_searcher = faiss_searcher
        self.settings = settings
        self.keyframes_base_path = str(settings.get_keyframes_root())
        self.map_keyframes_dir = os.path.join(str(settings.src_dir), "dict", "map-keyframes")
        self._keyframe_map_cache: Dict[str, Dict[int, int]] = {}
        self.enable_vqa = bool(settings.trake_enable_vqa)
        self.vqa_max_sequences = int(settings.trake_vqa_max_sequences)
        self.min_event_gap = max(0.0, float(settings.trake_min_event_gap_seconds))
        self.max_event_gap = max(self.min_event_gap, float(settings.trake_max_event_gap_seconds))
        self.max_sequence_span = max(self.max_event_gap, float(settings.trake_max_sequence_span_seconds))
        self.temporal_decay = max(0.0, float(settings.trake_temporal_decay))
        self.evidence_window = max(0.1, float(settings.trake_evidence_window_seconds))

    @staticmethod
    def _float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
            return parsed if math.isfinite(parsed) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalise_retrieval_scores(candidates: List[Dict]) -> None:
        if not candidates:
            return
        values = [TRAKE._float(candidate.get("score")) for candidate in candidates]
        low, high = min(values), max(values)
        span = high - low
        for candidate, value in zip(candidates, values):
            candidate["raw_visual_score"] = value
            candidate["visual_score"] = (value - low) / span if span > 1e-9 else 1.0
            candidate["score"] = candidate["visual_score"]

    def _plan_event(self, event_query: str) -> Dict[str, Any]:
        modality_plan = QueryPlanner.parse_query(event_query)
        translated = Translation()(modality_plan.get("visual_query") or event_query) or event_query
        local_plan = {
            "profile": "trake_ordered_event",
            "visual_queries": [translated],
            "ocr_queries": [modality_plan["ocr_query"]] if modality_plan.get("ocr_query") else [],
            "asr_queries": [modality_plan["asr_query"]] if modality_plan.get("asr_query") else [],
            "must_have_checks": [],
        }
        llm_plan: Dict[str, Any] = {}
        try:
            from src.services.openrouter_agent_planner import plan_agent_query_with_openrouter

            llm_plan = plan_agent_query_with_openrouter(event_query, local_plan)
        except Exception as exc:
            logger.warning("TRAKE event enrichment failed; using translated query: %s", exc)
        return {
            "original_query": event_query,
            "visual_query": (llm_plan.get("visual_queries") or [translated])[0],
            "ocr_query": (llm_plan.get("ocr_queries") or local_plan["ocr_queries"] or [""])[0],
            "asr_query": (llm_plan.get("asr_queries") or local_plan["asr_queries"] or [""])[0],
            "planner_source": llm_plan.get("planner_source") or "translation",
        }

    def _search_event_evidence(self, event_plan: Dict[str, Any]) -> Dict[str, List[Dict]]:
        evidence: Dict[str, List[Dict]] = {"ocr": [], "asr": []}
        if not event_plan.get("ocr_query") and not event_plan.get("asr_query"):
            return evidence
        try:
            from src.services.user_service import get_elastic_processor

            processor = get_elastic_processor()
            if self.settings.trake_ocr_enabled and event_plan.get("ocr_query"):
                evidence["ocr"] = processor.search_ocr(event_plan["ocr_query"], topk=80)
            if self.settings.trake_asr_enabled and event_plan.get("asr_query"):
                evidence["asr"] = processor.search_asr(event_plan["asr_query"], topk=80)
        except Exception as exc:
            logger.warning("TRAKE text evidence retrieval failed; continuing visual-only: %s", exc)
        return evidence

    def _apply_event_evidence(self, candidates: List[Dict], evidence: Dict[str, List[Dict]]) -> None:
        for modality in ("ocr", "asr"):
            rows = evidence.get(modality) or []
            if not rows:
                continue
            raw_scores = [self._float(row.get("_score")) for row in rows]
            low, high = min(raw_scores), max(raw_scores)
            span = high - low
            for row, raw_score in zip(rows, raw_scores):
                row["_trake_score"] = (raw_score - low) / span if span > 1e-9 else 1.0

        for candidate in candidates:
            video_id = str(candidate.get("video_id") or "")
            timestamp = self._float(candidate.get("timestamp"))
            evidence_scores = {"ocr": 0.0, "asr": 0.0}
            evidence_text: Dict[str, str] = {}
            for modality in ("ocr", "asr"):
                for row in evidence.get(modality) or []:
                    if str(row.get("video_id") or "") != video_id:
                        continue
                    row_timestamp = self._float(
                        row.get("nearest_timestamp")
                        if modality == "asr"
                        else row.get("timestamp")
                    )
                    gap = abs(timestamp - row_timestamp)
                    if gap > self.evidence_window:
                        continue
                    proximity = math.exp(-gap / self.evidence_window)
                    score = self._float(row.get("_trake_score")) * proximity
                    if score > evidence_scores[modality]:
                        evidence_scores[modality] = score
                        evidence_text[modality] = str(row.get("text") or row.get("ocr_text") or "")[:240]

            visual = self._float(candidate.get("visual_score"))
            active = [score for score in evidence_scores.values() if score > 0]
            evidence_score = max(active) if active else 0.0
            candidate["evidence_scores"] = evidence_scores
            candidate["evidence_text"] = evidence_text
            candidate["score"] = min(1.0, (0.82 * visual) + (0.18 * evidence_score)) if active else visual

    def retrieve_top_k(self, query: str, k: int = 200) -> List[Dict]:
        """Retrieve candidates from the active retrieval backend's corpus.

        The active backend is RETRIEVAL_BACKEND (BEiT3 or Jina CLIP v2, see
        src/services/retrieval_backend.py). Both expose the same result shape
        for the fields this method reads -- a frame_path from the same corpus
        as the served images, and a per-video frame_idx that is never the
        FAISS/vector id (that stays in vector_id and must not reach the
        submission CSV).
        """
        from src.services.retrieval_backend import get_active_retriever

        event_plan = self._plan_event(query)
        translated_query = event_plan["visual_query"]
        retriever = get_active_retriever()
        results = retriever.search_visual(translated_query, top_k=k)

        candidates = []
        for result in results:
            frame_path = str(result.get("frame_path") or "").replace("\\", "/")
            asset_key = str(result.get("asset_key") or "").replace("\\", "/") or frame_path
            path_parts = frame_path.split("/")
            namespace = str(result.get("namespace") or result.get("split") or (path_parts[0] if path_parts else ""))
            split = namespace
            frame_idx_value = result.get("frame_idx", result.get("frame_id"))
            try:
                global_frame_id = int(frame_idx_value)
            except (TypeError, ValueError):
                continue

            candidates.append({
                "faiss_idx": int(result.get("vector_id", -1)),
                "vector_id": int(result.get("vector_id", -1)),
                "global_frame_id": global_frame_id,
                "submission_frame_id": global_frame_id,
                "frame_id": result.get("frame_id"),
                "timestamp": result.get("timestamp", 0.0) or 0.0,
                "frame_name": Path(frame_path).name or str(result.get("frame_name") or ""),
                "video_id": str(result.get("video_id") or ""),
                "split": split,
                "score": float(result.get("score", 0.0)),
                "image_path": frame_path,
                "asset_key": asset_key,
                "query": query,
                "query_en": translated_query or query,
                "planner_source": event_plan["planner_source"],
                "retrieval_backend": getattr(retriever, "backend_id", None),
            })

        self._normalise_retrieval_scores(candidates)
        self._apply_event_evidence(candidates, self._search_event_evidence(event_plan))
        return candidates
    def group_by_video(self, candidates_list: List[List[Dict]]) -> Dict:
        """
        Group candidates by their fully-qualified video ID.

        The short legacy ID, for example V006, is not unique across
        splits, so grouping by it can mix frames from different videos.
        """
        video_groups = defaultdict(lambda: [[] for _ in range(len(candidates_list))])

        for event_idx, candidates in enumerate(candidates_list):
            for candidate in candidates:
                video_id = self._normalize_video_key(
                    candidate.get("split", ""), candidate.get("video_id", "")
                )
                if not video_id:
                    continue
                video_groups[video_id][event_idx].append(candidate)

        limit = max(1, int(self.settings.trake_candidates_per_event_video))
        for event_lists in video_groups.values():
            for event_index, candidates in enumerate(event_lists):
                event_lists[event_index] = sorted(
                    candidates,
                    key=lambda item: self._float(item.get("score")),
                    reverse=True,
                )[:limit]

        return video_groups

    def _candidate_timestamp(self, candidate: Dict) -> float:
        if candidate.get("timestamp") is not None:
            return self._float(candidate.get("timestamp"))
        return self._float(candidate.get("global_frame_id")) / 25.0

    def beam_search_sequences(self, video_id: str, event_candidates: List[List[Dict]], beam_width: int = 50) -> List[Dict]:
        """
        Find top temporal sequences using beam search.
        """
        beam = []
        for candidate in event_candidates[0]:
            timestamp = self._candidate_timestamp(candidate)
            seq_info = {
                "video_id": video_id,
                "frames": [candidate["frame_name"]],
                "global_frame_ids": [candidate["global_frame_id"]],
                "timestamps": [timestamp],
                "splits": [candidate["split"]],
                "base_score": candidate["score"],
                "total_score": candidate["score"],
                "frame_details": [candidate],
                "temporal_gaps": [],
            }
            beam.append(seq_info)

        beam.sort(key=lambda x: x["total_score"], reverse=True)
        beam = beam[:beam_width]

        for event_idx in range(1, len(event_candidates)):
            new_beam = []
            next_candidates = event_candidates[event_idx]
            next_candidates.sort(key=self._candidate_timestamp)

            for seq in beam:
                last_timestamp = seq["timestamps"][-1]
                for candidate in next_candidates:
                    candidate_timestamp = self._candidate_timestamp(candidate)
                    adjacent_gap = candidate_timestamp - last_timestamp
                    total_span = candidate_timestamp - seq["timestamps"][0]
                    if adjacent_gap > self.min_event_gap and adjacent_gap <= self.max_event_gap and total_span <= self.max_sequence_span:
                        new_base_score = seq["base_score"] + self._float(candidate.get("score"))
                        penalty = math.exp(-self.temporal_decay * total_span) if total_span > 0 else 1.0
                        new_total_score = new_base_score * penalty

                        new_seq = {
                            "video_id": video_id,
                            "frames": seq["frames"] + [candidate["frame_name"]],
                            "global_frame_ids": seq["global_frame_ids"] + [candidate["global_frame_id"]],
                            "timestamps": seq["timestamps"] + [candidate_timestamp],
                            "splits": list(set(seq["splits"] + [candidate["split"]])),
                            "base_score": new_base_score,
                            "total_score": new_total_score,
                            "frame_details": seq["frame_details"] + [candidate],
                            "temporal_gaps": seq.get("temporal_gaps", []) + [adjacent_gap],
                        }
                        new_beam.append(new_seq)

            new_beam.sort(key=lambda x: x["total_score"], reverse=True)
            beam = new_beam[:beam_width]

            if not beam:
                break

        return beam

    def find_valid_sequences(self, video_groups: Dict, n_events: int) -> List[Dict]:
        """
        Find valid sequences where events occur in temporal order.
        """
        valid_sequences = []

        for video_id, event_candidates in video_groups.items():
            if any(len(candidates) == 0 for candidates in event_candidates):
                continue

            sequences = self.beam_search_sequences(
                video_id,
                event_candidates,
                beam_width=max(1, int(self.settings.trake_beam_width)),
            )
            valid_sequences.extend(sequences)

        return valid_sequences

    def rank_sequences(self, sequences: List[Dict], top_n: int = 20) -> List[Dict]:
        """
        Rank sequences by relevance score.
        """
        sequences.sort(key=lambda x: x["total_score"], reverse=True)
        return sequences[:top_n]

    def _split_to_folder(self, split: str) -> str:
        clean = str(split or "").replace("videos-", "").replace("videos_", "")
        if re.fullmatch(r"L\d+_[A-Za-z0-9]+", clean, flags=re.IGNORECASE):
            prefix, suffix = clean.split("_", 1)
            return f"{prefix.upper()}_{suffix.lower()}"
        parts = clean.split("-")
        if len(parts) >= 2:
            return f"{parts[0].upper()}_{'_'.join(parts[1:])}"
        return clean.upper()

    def _normalize_video_key(self, split: str, video_id: str) -> str:
        folder = self._split_to_folder(split)
        prefix = folder.split("_")[0] if folder else ""
        video_key = str(video_id or "")
        if prefix and video_key and not video_key.startswith(prefix):
            return f"{prefix}_{video_key}"
        return video_key

    def _load_keyframe_map(self, video_key: str) -> Dict[int, int]:
        if video_key in self._keyframe_map_cache:
            return self._keyframe_map_cache[video_key]

        mapping: Dict[int, int] = {}
        map_path = os.path.join(self.map_keyframes_dir, f"{video_key}.csv")
        try:
            with open(map_path, newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    mapping[int(row["n"])] = int(row["frame_idx"])
        except Exception as exc:
            logger.warning("Error loading keyframe map %s: %s", map_path, exc)

        self._keyframe_map_cache[video_key] = mapping
        return mapping

    def _resolve_image_path(self, frame_detail: Dict) -> str:
        raw_path = frame_detail.get("image_path", "")
        if raw_path and os.path.isabs(raw_path) and os.path.exists(raw_path):
            return raw_path

        split = str(frame_detail.get("split", ""))
        video_id = str(frame_detail.get("video_id", ""))
        frame_name = str(frame_detail.get("frame_name", ""))
        folder_key = self._split_to_folder(split)
        video_key = self._normalize_video_key(split, video_id)

        candidates = []
        if raw_path:
            candidates.append(os.path.join(self.keyframes_base_path, raw_path))
        if folder_key and video_key and frame_name:
            candidates.append(os.path.join(self.keyframes_base_path, folder_key, video_key, frame_name))

        match = re.search(r"_(\d+)\.[^.]+$", frame_name)
        if folder_key and video_key and match:
            legacy_n = int(match.group(1))
            frame_idx = self._load_keyframe_map(video_key).get(legacy_n)
            if frame_idx is not None:
                candidates.append(os.path.join(self.keyframes_base_path, folder_key, video_key, f"{frame_idx:06d}.webp"))

        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate
        return candidates[-1] if candidates else raw_path

    @staticmethod
    def _digits_to_int(value: Any) -> Any:
        text = re.sub(r"[^\d]", "", str(value if value is not None else ""))
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    def _resolve_submission_frame_id(self, frame_detail: Dict) -> Any:
        """Original per-video frame index for the submission CSV.

        Resolution order, none of which is the FAISS/vector id:
          1. the explicit ``submission_frame_id`` carried since retrieval,
          2. digits of the resolved keyframe filename (``000123.webp``),
          3. ``global_frame_id`` (retriever's normalized ``frame_idx``),
          4. the map-keyframes ``frame_idx`` for the legacy ``n`` in the name.
        Returns ``None`` when nothing trustworthy resolves.
        """
        explicit = frame_detail.get("submission_frame_id")
        if isinstance(explicit, int):
            return explicit
        parsed = self._digits_to_int(explicit)
        if parsed is not None:
            return parsed

        resolved_path = self._resolve_image_path(frame_detail)
        if resolved_path:
            stem_digits = self._digits_to_int(os.path.splitext(os.path.basename(resolved_path))[0])
            if stem_digits is not None:
                return stem_digits

        gid = frame_detail.get("global_frame_id")
        if isinstance(gid, int):
            return gid
        parsed_gid = self._digits_to_int(gid)
        if parsed_gid is not None:
            return parsed_gid

        match = re.search(r"_(\d+)\.[^.]+$", str(frame_detail.get("frame_name", "")))
        if match:
            video_key = self._normalize_video_key(
                frame_detail.get("split", ""), frame_detail.get("video_id", "")
            )
            mapped = self._load_keyframe_map(video_key).get(int(match.group(1)))
            if mapped is not None:
                return int(mapped)
        return None

    def _sequence_is_valid(self, sequence: Dict, n_events: int) -> bool:
        frame_details = sequence.get("frame_details") or []
        if len(frame_details) != n_events or n_events < 2:
            return False

        video_key = self._normalize_video_key(sequence.get("video_id", ""), "") or str(
            sequence.get("video_id") or ""
        )
        last_ts = None
        for frame in frame_details:
            frame_video = self._normalize_video_key(
                frame.get("split", ""), frame.get("video_id", "")
            )
            if video_key and frame_video and frame_video != sequence.get("video_id") and frame_video != video_key:
                return False
            if self._resolve_submission_frame_id(frame) is None:
                return False
            ts = self._float(frame.get("timestamp"))
            if last_ts is not None and ts <= last_ts:
                return False
            last_ts = ts
        return True

    def _get_image_base64(self, frame_detail: Dict) -> str:
        """
        Get base64 encoded image.
        """
        try:
            full_image_path = self._resolve_image_path(frame_detail)
            with open(full_image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logger.warning("Error loading image %s: %s", frame_detail.get("image_path", ""), e)
            return ""

    def format_response(self, sequences: List[Dict]) -> List[Dict]:
        """
        Format sequences into the required API response format.
        """
        response = []

        for seq_id, sequence in enumerate(sequences):
            frames = []
            submission_ids = []

            for frame_id, frame_detail in enumerate(sequence["frame_details"]):
                folder_key = self._split_to_folder(frame_detail.get("split", ""))
                video_key = self._normalize_video_key(frame_detail.get("split", ""), frame_detail.get("video_id", ""))
                resolved_path = self._resolve_image_path(frame_detail)
                submission_frame_id = self._resolve_submission_frame_id(frame_detail)
                submission_ids.append(submission_frame_id)
                frame_key = os.path.splitext(os.path.basename(resolved_path))[0] if resolved_path else frame_detail.get("global_frame_id")
                image_b64 = self._get_image_base64(frame_detail)

                frames.append({
                    "id": frame_id,
                    "event_index": frame_id + 1,
                    "event_query": frame_detail.get("query"),
                    "event_query_en": frame_detail.get("query_en"),
                    "folder_key": folder_key,
                    "video_key": video_key,
                    "frame_key": frame_key,
                    "submission_frame_id": submission_frame_id,
                    "timestamp": frame_detail.get("timestamp", 0.0),
                    "image": image_b64,
                    "evidence": {
                        "scores": frame_detail.get("evidence_scores", {}),
                        "text": frame_detail.get("evidence_text", {}),
                    },
                })

            sequence_video = sequence.get("video_id")
            sequence_key = "{}#{}".format(
                sequence_video,
                "-".join("?" if value is None else str(value) for value in submission_ids),
            )

            response.append({
                "id": seq_id,
                "sequence_id": sequence.get("sequence_id") or sequence_key,
                "video_id": sequence_video,
                "frame_names": [frame.get("frame_name") for frame in sequence["frame_details"]],
                "timestamps": [frame.get("timestamp", 0.0) for frame in sequence["frame_details"]],
                "score": sequence.get("total_score", 0.0),
                "base_score": sequence.get("base_score", 0.0),
                "temporal_gaps": sequence.get("temporal_gaps", []),
                "event_queries": [frame.get("query") for frame in sequence["frame_details"]],
                "event_queries_en": [frame.get("query_en") for frame in sequence["frame_details"]],
                "evidence": [
                    {
                        "scores": frame.get("evidence_scores", {}),
                        "text": frame.get("evidence_text", {}),
                    }
                    for frame in sequence["frame_details"]
                ],
                "verification_score": sequence.get("verification_score"),
                "vlm_score": sequence.get("vlm_score"),
                "vlm_decision": sequence.get("vlm_decision"),
                "vlm_reason": sequence.get("vlm_reason"),
                "vlm_matched_events": sequence.get("vlm_matched_events", []),
                "vlm_missing_events": sequence.get("vlm_missing_events", []),
                "verification": sequence.get("verification", {}),
                "frames": frames,
            })

        return response

    def process_temporal_search(self, queries: List[Dict], top_k: int = 100, top_results: int = 20) -> List[Dict]:
        """
        Main function to process temporal search queries.
        """
        events = [q["query"] for q in queries]

        # NOTE: use ``logger`` here, never ``print``. Event text is Vietnamese and
        # a bare ``print`` to a cp1252 Windows stdout raises UnicodeEncodeError,
        # which would surface as an unhandled 500 (and, lacking CORS headers, as a
        # "service unavailable" transport error in the browser).
        logger.info("TRAKE processing %d events", len(events))

        candidates_list = []
        for i, event in enumerate(events):
            logger.info("Retrieving candidates for event %d: %.60s", i + 1, event)
            retrieval_top_k = max(top_k, int(self.settings.trake_retrieval_top_k))
            candidates = self.retrieve_top_k(event, retrieval_top_k)
            candidates_list.append(candidates)
            logger.info("Found %d candidates for event %d", len(candidates), i + 1)

        video_groups = self.group_by_video(candidates_list)
        logger.info("Grouped candidates into %d videos", len(video_groups))

        valid_sequences = self.find_valid_sequences(video_groups, len(events))
        logger.info("Beam search produced %d candidate sequences", len(valid_sequences))

        n_events = len(events)
        valid_sequences = [
            sequence for sequence in valid_sequences if self._sequence_is_valid(sequence, n_events)
        ]
        logger.info(
            "%d sequences pass the contract (all events, one video, ordered, resolvable frame ids)",
            len(valid_sequences),
        )

        if not valid_sequences:
            return []

        ranked_sequences = self.rank_sequences(valid_sequences, top_results)

        if self.enable_vqa:
            logger.info("Validating top sequences with BLIP-VQA")
            for seq in ranked_sequences[: self.vqa_max_sequences]:
                vqa_scores = []
                for i, frame_detail in enumerate(seq["frame_details"]):
                    event_query = events[i]
                    vqa_question = QueryPlanner.generate_vqa_question(event_query)
                    img_path = self._resolve_image_path(frame_detail)

                    score = 0.0
                    if os.path.exists(img_path):
                        score = reranker_service.score_image(img_path, vqa_question)
                    vqa_scores.append(score)

                avg_vqa = sum(vqa_scores) / len(vqa_scores) if vqa_scores else 0.0
                vqa_scaled = avg_vqa * len(events)
                old_score = seq["total_score"]
                seq["total_score"] = (old_score * 0.7) + (vqa_scaled * 0.3)
                seq["vqa_confidence"] = avg_vqa

            ranked_sequences.sort(key=lambda x: x["total_score"], reverse=True)
        else:
            logger.debug("Skipping BLIP-VQA validation (set TRAKE_ENABLE_VQA=true to enable).")

        try:
            from src.services.openrouter_trake_verifier import verify_trake_sequences

            ranked_sequences, vlm_summary = verify_trake_sequences(
                ranked_sequences,
                events,
                self._resolve_image_path,
            )
        except Exception as exc:
            logger.warning("TRAKE sequence VLM verification failed; keeping temporal ranking: %s", exc)
            vlm_summary = {"enabled": True, "status": "fallback", "evaluated": 0, "error": str(exc)[:180]}
        for sequence in ranked_sequences:
            sequence["verification"] = {
                "method": "openrouter_sequence_vlm" if sequence.get("vlm_score") is not None else "temporal_evidence",
                "summary": vlm_summary,
            }

        logger.info("Formatting %d ranked TRAKE sequences", len(ranked_sequences))
        return self.format_response(ranked_sequences)
