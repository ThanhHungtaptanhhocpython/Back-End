import base64
import csv
import math
import os
import re
import unicodedata
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
        self._event_plan_cache: Dict[str, Dict[str, Any]] = {}
        self.enable_vqa = bool(settings.trake_enable_vqa)
        self.vqa_max_sequences = int(settings.trake_vqa_max_sequences)
        self.min_event_gap = max(0.0, float(settings.trake_min_event_gap_seconds))
        self.max_event_gap = max(self.min_event_gap, float(settings.trake_max_event_gap_seconds))
        self.max_sequence_span = max(self.max_event_gap, float(settings.trake_max_sequence_span_seconds))
        self.temporal_decay = max(0.0, float(settings.trake_temporal_decay))
        self.consecutive_compact_span = max(0.1, float(settings.trake_consecutive_compact_span_seconds))
        self.consecutive_span_decay = max(0.0, float(settings.trake_consecutive_span_decay))
        self.vlm_max_total_sequences = max(1, int(settings.trake_vlm_max_total_sequences))
        self.evidence_window = max(0.1, float(settings.trake_evidence_window_seconds))
        logger.info(
            "TRAKE config: min_gap=%.2fs decay=%.4f compact_span=%.2fs compact_decay=%.4f vlm_total=%d anchor=%s videos_per_event=%d timeline_top_k=%d jina_m0=%s",
            self.min_event_gap,
            self.temporal_decay,
            self.consecutive_compact_span,
            self.consecutive_span_decay,
            self.vlm_max_total_sequences,
            bool(settings.trake_anchor_expansion_enabled),
            int(settings.trake_anchor_video_limit),
            int(settings.trake_anchor_timeline_top_k),
            bool(settings.jina_reranker_enabled),
        )

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
        cached = self._event_plan_cache.get(event_query)
        if cached is not None:
            return dict(cached)
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
        plan = {
            "original_query": event_query,
            "visual_query": (llm_plan.get("visual_queries") or [translated])[0],
            "ocr_query": (llm_plan.get("ocr_queries") or local_plan["ocr_queries"] or [""])[0],
            "asr_query": (llm_plan.get("asr_queries") or local_plan["asr_queries"] or [""])[0],
            "planner_source": llm_plan.get("planner_source") or "translation",
        }
        self._event_plan_cache[event_query] = dict(plan)
        return plan

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

    def _candidates_from_results(
        self,
        query: str,
        event_plan: Dict[str, Any],
        results: List[Dict],
        *,
        apply_evidence: bool = True,
    ) -> List[Dict]:
        translated_query = event_plan["visual_query"]
        candidates = []
        for result in results:
            frame_path = str(result.get("frame_path") or "").replace("\\", "/")
            path_parts = frame_path.split("/")
            namespace = str(result.get("namespace") or (path_parts[0] if path_parts else ""))
            split = namespace
            # ``frame_idx`` from either retriever is the *original per-video
            # frame index* (map-keyframes ``frame_idx`` column), already stripped
            # of zero padding. It is never the FAISS/vector id -- that lives in
            # ``vector_id`` and must not reach the submission.
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
                "query": query,
                "query_en": translated_query or query,
                "planner_source": event_plan["planner_source"],
            })

        self._normalise_retrieval_scores(candidates)
        if apply_evidence:
            self._apply_event_evidence(candidates, self._search_event_evidence(event_plan))
        return candidates

    def retrieve_top_k(self, query: str, k: int = 200) -> List[Dict]:
        """Retrieve one event from the active, internally consistent corpus."""
        from src.services.visual_retriever import get_visual_retriever

        event_plan = self._plan_event(query)
        results = get_visual_retriever().search_visual(event_plan["visual_query"], top_k=k)
        return self._candidates_from_results(query, event_plan, results)

    def retrieve_events_batch(self, queries: List[str], k: int = 200) -> List[List[Dict]]:
        """Plan all events, then encode/search them in one Jina model call."""
        from src.services.visual_retriever import get_visual_retriever

        event_plans = [self._plan_event(query) for query in queries]
        visual_queries = [plan["visual_query"] for plan in event_plans]
        retriever = get_visual_retriever()
        if hasattr(retriever, "search_visual_batch"):
            result_batches = retriever.search_visual_batch(visual_queries, top_k=k)
        else:
            result_batches = [retriever.search_visual(query, top_k=k) for query in visual_queries]
        return [
            self._candidates_from_results(query, plan, results)
            for query, plan, results in zip(queries, event_plans, result_batches)
        ]

    @staticmethod
    def _candidate_identity(candidate: Dict) -> tuple[str, str]:
        """Stable identity used when global and anchor passes overlap."""
        return (
            str(candidate.get("video_id") or ""),
            str(candidate.get("vector_id") or candidate.get("image_path") or candidate.get("frame_name") or ""),
        )

    def _rescore_event_candidates(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Normalise one merged event pool and restore optional OCR/ASR evidence."""
        if not candidates:
            return candidates
        for candidate in candidates:
            candidate["score"] = self._float(
                candidate.get("raw_visual_score", candidate.get("score"))
            )
        self._normalise_retrieval_scores(candidates)
        self._apply_event_evidence(candidates, self._search_event_evidence(self._plan_event(query)))
        return candidates

    def _expand_anchor_videos(self, events: List[str], candidates_list: List[List[Dict]]) -> List[List[Dict]]:
        """Recover events by scoring timelines anchored by every event.

        Global FAISS retrieval can omit the correct frame even when one event
        identifies the correct video. Candidate sources are selected in an
        event-round-robin order so E1 cannot monopolise the anchor budget.
        Each selected video's local timeline is scored for every event and
        merged before beam search.
        """
        if (
            not bool(getattr(self.settings, "trake_anchor_expansion_enabled", True))
            or not events
            or not candidates_list
            or not any(candidates_list)
        ):
            return candidates_list

        try:
            from src.services.visual_retriever import get_visual_retriever

            retriever = get_visual_retriever()
        except Exception as exc:  # noqa: BLE001 - global retrieval remains useful
            logger.warning("TRAKE anchor expansion unavailable: %s", exc)
            return candidates_list
        if not hasattr(retriever, "search_video_timelines") and not hasattr(retriever, "search_video_timeline"):
            logger.debug("Active visual retriever does not support TRAKE anchor expansion.")
            return candidates_list

        anchors_per_event = max(1, int(getattr(self.settings, "trake_anchor_video_limit", 12)))
        timeline_top_k = max(1, int(getattr(self.settings, "trake_anchor_timeline_top_k", 24)))
        anchors: List[str] = []
        anchor_sources: Dict[str, List[tuple[int, int]]] = {}
        ranked_events = [
            sorted(candidates, key=lambda item: self._float(item.get("score")), reverse=True)
            for candidates in candidates_list
        ]
        for event_index, candidates in enumerate(ranked_events, start=1):
            contributed = 0
            event_videos: set[str] = set()
            for rank, candidate in enumerate(candidates, start=1):
                video_id = str(candidate.get("video_id") or "").strip()
                if not video_id or video_id in event_videos:
                    continue
                event_videos.add(video_id)
                if video_id not in anchor_sources:
                    anchors.append(video_id)
                    anchor_sources[video_id] = []
                anchor_sources[video_id].append((event_index, rank))
                contributed += 1
                if contributed >= anchors_per_event:
                    break
        if not anchors:
            return candidates_list

        logger.info(
            "TRAKE anchor expansion using %d videos from up to %d/event: %s",
            len(anchors),
            anchors_per_event,
            ", ".join(
                "%s(%s)" % (
                    video_id,
                    ",".join("E%d#%d" % (event_index, rank) for event_index, rank in sources),
                )
                for video_id, sources in anchor_sources.items()
            ),
        )
        merged = [list(candidates) for candidates in candidates_list]
        known = [{self._candidate_identity(candidate) for candidate in candidates} for candidates in merged]
        additions = 0
        event_plans = [self._plan_event(event) for event in events]
        timeline_batches = None
        if hasattr(retriever, "search_video_timelines"):
            try:
                timeline_batches = retriever.search_video_timelines(
                    [plan["visual_query"] for plan in event_plans], anchors, top_k=timeline_top_k,
                )
            except Exception as exc:  # noqa: BLE001 - retain the single-query compatibility path
                logger.warning("TRAKE batched anchor scoring failed; using single-query fallback: %s", exc)
        for event_index, (event, event_plan) in enumerate(zip(events, event_plans)):
            for video_id in anchors:
                try:
                    local_results = (
                        (timeline_batches.get(video_id, [[] for _ in events])[event_index])
                        if timeline_batches is not None
                        else retriever.search_video_timeline(
                            event_plan["visual_query"], video_id, top_k=timeline_top_k,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - skip one video, retain global pass
                    logger.warning("TRAKE anchor scoring failed for %s event %d: %s", video_id, event_index + 1, exc)
                    continue
                # Evidence is retrieved once per merged event below. Calling
                # OCR/ASR here for every anchor video makes a 12-video pass
                # needlessly slow without changing its score.
                for candidate in self._candidates_from_results(
                    event, event_plan, local_results, apply_evidence=False,
                ):
                    identity = self._candidate_identity(candidate)
                    if identity in known[event_index]:
                        continue
                    known[event_index].add(identity)
                    merged[event_index].append(candidate)
                    additions += 1

        if additions:
            for event, candidates in zip(events, merged):
                self._rescore_event_candidates(event, candidates)
            logger.info("TRAKE anchor expansion added %d local timeline candidates", additions)
        return merged

    def _rerank_event_candidates(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Optionally apply Jina M0 after recall, before same-video grouping."""
        if not candidates or not bool(getattr(self.settings, "jina_reranker_enabled", False)):
            return candidates
        try:
            from src.services.jina_reranker import rerank_kis_results

            # M0 receives a bounded pool. Reserve a slot for the strongest
            # candidate in each video first; otherwise many near-duplicate
            # frames from one wrong video can consume the entire request and
            # a useful anchor video never reaches multimodal reranking.
            ordered = sorted(candidates, key=lambda item: self._float(item.get("score")), reverse=True)
            pool_size = max(1, min(int(self.settings.jina_reranker_candidate_pool), len(ordered), 100))
            diverse: List[Dict] = []
            seen_videos: set[str] = set()
            for candidate in ordered:
                video_id = self._normalize_video_key(candidate.get("split", ""), candidate.get("video_id", ""))
                if video_id and video_id not in seen_videos:
                    seen_videos.add(video_id)
                    diverse.append(candidate)
                if len(diverse) >= pool_size:
                    break
            selected_ids = {id(candidate) for candidate in diverse}
            staged = diverse + [candidate for candidate in ordered if id(candidate) not in selected_ids]
            logger.info(
                "TRAKE Jina M0 pool for event %.60s: %d candidates from %d videos",
                query,
                pool_size,
                len(diverse),
            )
            reranked = rerank_kis_results(query, staged, settings=self.settings)
            reranked_count = sum(1 for candidate in reranked if candidate.get("reranker_score") is not None)
            if reranked_count:
                logger.info("TRAKE Jina M0 reranked %d candidates for event: %.60s", len(reranked), query)
            else:
                logger.warning(
                    "TRAKE Jina M0 did not return reranker scores for event %.60s; retaining retrieval order.",
                    query,
                )
            return reranked
        except Exception as exc:  # noqa: BLE001 - reranking must never remove recall
            logger.warning("TRAKE Jina M0 reranking failed; retaining CLIP order: %s", exc)
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

    def _log_candidate_trace(self, candidates_list: List[List[Dict]]) -> None:
        """Emit a bounded, query-agnostic recall trace when explicitly enabled."""
        if not bool(getattr(self.settings, "trake_trace_candidates", False)):
            return

        traces: Dict[str, Dict[int, tuple[int, float, float]]] = {}
        for event_index, candidates in enumerate(candidates_list, start=1):
            ranked = sorted(candidates, key=lambda item: self._float(item.get("score")), reverse=True)
            for rank, candidate in enumerate(ranked, start=1):
                video_id = self._normalize_video_key(
                    candidate.get("split", ""), candidate.get("video_id", "")
                )
                if not video_id:
                    continue
                event_trace = traces.setdefault(video_id, {})
                if event_index not in event_trace:
                    event_trace[event_index] = (
                        rank,
                        self._float(candidate.get("score")),
                        self._candidate_timestamp(candidate),
                    )

        limit = max(1, int(getattr(self.settings, "trake_trace_video_limit", 20)))
        ordered = sorted(
            traces.items(),
            key=lambda item: (-len(item[1]), sum(entry[0] for entry in item[1].values())),
        )[:limit]
        for video_id, events in ordered:
            details = " ".join(
                "E%d=#%d score=%.3f t=%.2f" % (event, rank, score, timestamp)
                for event, (rank, score, timestamp) in sorted(events.items())
            )
            logger.info(
                "TRAKE candidate trace: video=%s coverage=%d/%d %s",
                video_id,
                len(events),
                len(candidates_list),
                details,
            )

    def _candidate_trace_ranked_videos(self, candidates_list: List[List[Dict]]) -> List[str]:
        """Rank videos that have at least one candidate for every temporal event."""
        if not candidates_list:
            return []

        traces: Dict[str, Dict[int, tuple[int, float, float]]] = {}
        for event_index, candidates in enumerate(candidates_list, start=1):
            ranked = sorted(candidates, key=lambda item: self._float(item.get("score")), reverse=True)
            for rank, candidate in enumerate(ranked, start=1):
                video_id = self._normalize_video_key(
                    candidate.get("split", ""), candidate.get("video_id", "")
                )
                if not video_id:
                    continue
                event_trace = traces.setdefault(video_id, {})
                if event_index not in event_trace:
                    event_trace[event_index] = (
                        rank,
                        self._float(candidate.get("score")),
                        self._candidate_timestamp(candidate),
                    )

        required_events = len(candidates_list)
        complete = []
        for video_id, events in traces.items():
            if len(events) != required_events:
                continue
            ranks = [entry[0] for entry in events.values()]
            scores = [entry[1] for entry in events.values()]
            timestamps = [entry[2] for entry in events.values()]
            span = max(timestamps) - min(timestamps) if timestamps else 0.0
            complete.append((video_id, sum(ranks), span, -sum(scores)))

        limit = max(1, int(getattr(self.settings, "trake_trace_video_limit", 20)))
        complete.sort(key=lambda item: (item[1], item[2], item[3], item[0]))
        return [video_id for video_id, *_ in complete[:limit]]

    def _prioritise_trace_video_sequences_for_verification(
        self,
        ranked_sequences: List[Dict],
        valid_sequences: List[Dict],
        trace_video_order: List[str],
        max_promoted: int,
        events: List[str] | None = None,
        shared_context: str = "",
    ) -> List[Dict]:
        """Put coverage-complete trace videos where VLM can actually inspect them."""
        if not trace_video_order or max_promoted <= 0:
            return ranked_sequences

        prefer_compact = self._prefers_compact_sequence(events or [], shared_context)
        preferred_span = max(0.0, float(self.consecutive_compact_span))

        def sequence_key(sequence: Dict) -> tuple[float, float, float]:
            score = self._float(sequence.get("total_score"))
            span = self._sequence_span_seconds(sequence)
            if prefer_compact:
                outside_compact = 0.0 if span <= preferred_span else 1.0
                return (outside_compact, span, -score)
            return (0.0, -score, span)

        candidates_by_video: Dict[str, List[Dict]] = {}
        for sequence in valid_sequences:
            video_id = str(sequence.get("video_id") or "")
            if not video_id:
                continue
            if str(sequence.get("vlm_decision") or "") == "wrong":
                continue
            candidates_by_video.setdefault(video_id, []).append(sequence)

        best_by_video: Dict[str, Dict] = {}
        for video_id, sequences in candidates_by_video.items():
            best_by_video[video_id] = min(sequences, key=sequence_key)

        promoted: List[Dict] = []
        for video_id in trace_video_order:
            sequence = best_by_video.get(video_id)
            if sequence is None:
                continue
            sequence["trace_verification_candidate"] = True
            if prefer_compact:
                sequence["trace_compact_selected"] = True
            promoted.append(sequence)
            if len(promoted) >= max_promoted:
                break

        if not promoted:
            return ranked_sequences

        promoted_ids = {id(sequence) for sequence in promoted}
        merged = promoted + [sequence for sequence in ranked_sequences if id(sequence) not in promoted_ids]
        logger.info(
            "TRAKE prioritized %d coverage-complete trace videos for VLM verification: %s",
            len(promoted),
            ", ".join(
                "%s@%.2fs" % (sequence.get("video_id"), self._sequence_span_seconds(sequence))
                for sequence in promoted[:10]
            ),
        )
        return merged
    def _promote_trace_video_sequences(
        self,
        ranked_sequences: List[Dict],
        valid_sequences: List[Dict],
        trace_video_order: List[str],
        max_promoted: int,
        rejected_videos: set[str] | None = None,
    ) -> List[Dict]:
        """Move coverage-complete trace videos near the front of fallback output."""
        if not ranked_sequences or not trace_video_order or max_promoted <= 0:
            return ranked_sequences

        rejected_videos = rejected_videos or set()
        best_by_video: Dict[str, Dict] = {}
        for sequence in sorted(valid_sequences, key=lambda item: self._float(item.get("total_score")), reverse=True):
            video_id = str(sequence.get("video_id") or "")
            if not video_id or video_id in best_by_video or video_id in rejected_videos:
                continue
            if str(sequence.get("vlm_decision") or "") == "wrong":
                continue
            best_by_video[video_id] = sequence

        promoted = []
        for video_id in trace_video_order:
            if video_id in rejected_videos:
                continue
            sequence = best_by_video.get(video_id)
            if sequence is None:
                continue
            sequence["forced_trace_candidate"] = True
            promoted.append(sequence)
            if len(promoted) >= max_promoted:
                break

        if not promoted:
            return ranked_sequences

        promoted_ids = {id(sequence) for sequence in promoted}
        merged = promoted + [sequence for sequence in ranked_sequences if id(sequence) not in promoted_ids]
        logger.info(
            "TRAKE promoted %d coverage-complete trace videos into fallback output: %s",
            len(promoted),
            ", ".join(str(sequence.get("video_id")) for sequence in promoted[:10]),
        )
        return merged

    def _diversify_fallback_sequences_by_video(self, sequences: List[Dict]) -> List[Dict]:
        """Prefer one sequence per video before returning duplicate fallback variants."""
        if len(sequences) <= 1:
            return sequences

        first_by_video = []
        variants = []
        seen_videos: set[str] = set()
        for sequence in sequences:
            video_id = str(sequence.get("video_id") or "")
            if video_id and video_id not in seen_videos:
                seen_videos.add(video_id)
                sequence["fallback_video_diversified"] = True
                first_by_video.append(sequence)
            else:
                variants.append(sequence)

        if not variants:
            return sequences

        diversified = first_by_video + variants
        logger.info(
            "TRAKE diversified fallback output by video: unique_videos=%d variants_deferred=%d",
            len(first_by_video),
            len(variants),
        )
        return diversified

    def _sequence_meets_compact_requirement(
        self,
        sequence: Dict,
        events: List[str],
        shared_context: str = "",
    ) -> bool:
        if not self._prefers_compact_sequence(events, shared_context):
            return True
        return self._sequence_span_seconds(sequence) <= max(0.0, float(self.consecutive_compact_span))

    def _make_sequence_from_candidates(self, video_id: str, chosen: List[Dict]) -> Dict:
        timestamps = [self._candidate_timestamp(candidate) for candidate in chosen]
        base_score = sum(self._float(candidate.get("score")) for candidate in chosen)
        event_count = max(1, len(chosen))
        semantic_score = base_score / event_count
        span = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else 0.0
        penalty = math.exp(-self.temporal_decay * span) if span > 0 else 1.0
        return {
            "video_id": video_id,
            "frames": [candidate["frame_name"] for candidate in chosen],
            "global_frame_ids": [candidate["global_frame_id"] for candidate in chosen],
            "timestamps": timestamps,
            "splits": list({candidate["split"] for candidate in chosen}),
            "base_score": base_score,
            "semantic_score": semantic_score,
            "temporal_penalty": penalty,
            "total_score": semantic_score * penalty,
            "frame_details": list(chosen),
            "temporal_gaps": [timestamps[index] - timestamps[index - 1] for index in range(1, len(timestamps))],
            "sequence_span_seconds": span,
            "trace_compact_candidate": True,
        }

    def _best_compact_sequence_for_video(self, video_id: str, event_candidates: List[List[Dict]]) -> Dict | None:
        preferred_span = max(0.0, float(self.consecutive_compact_span))
        if not event_candidates or any(not candidates for candidates in event_candidates):
            return None

        sorted_events = [sorted(candidates, key=self._candidate_timestamp) for candidates in event_candidates]
        best: tuple[tuple[float, float], List[Dict]] | None = None

        def walk(event_index: int, chosen: List[Dict], start_time: float, last_time: float) -> None:
            nonlocal best
            if event_index >= len(sorted_events):
                sequence = self._make_sequence_from_candidates(video_id, chosen)
                span = self._sequence_span_seconds(sequence)
                score = self._float(sequence.get("total_score"))
                key = (span, -score)
                if best is None or key < best[0]:
                    best = (key, chosen.copy())
                return

            for candidate in sorted_events[event_index]:
                timestamp = self._candidate_timestamp(candidate)
                if chosen:
                    adjacent_gap = timestamp - last_time
                    if adjacent_gap <= self.min_event_gap or adjacent_gap > self.max_event_gap:
                        continue
                    if timestamp - start_time > preferred_span:
                        continue
                walk(event_index + 1, chosen + [candidate], start_time if chosen else timestamp, timestamp)

        walk(0, [], 0.0, 0.0)
        if best is None:
            return None
        return self._make_sequence_from_candidates(video_id, best[1])

    def _build_compact_trace_sequences(
        self,
        candidates_list: List[List[Dict]],
        trace_video_order: List[str],
        events: List[str],
        shared_context: str,
        max_sequences: int,
    ) -> List[Dict]:
        if max_sequences <= 0 or not trace_video_order or not self._prefers_compact_sequence(events, shared_context):
            return []

        sequences: List[Dict] = []
        for video_id in trace_video_order:
            event_candidates: List[List[Dict]] = []
            for candidates in candidates_list:
                filtered = [
                    candidate for candidate in candidates
                    if self._normalize_video_key(candidate.get("split", ""), candidate.get("video_id", "")) == video_id
                ]
                event_candidates.append(filtered)
            sequence = self._best_compact_sequence_for_video(video_id, event_candidates)
            if sequence is None:
                continue
            sequence["trace_verification_candidate"] = True
            sequence["trace_compact_selected"] = True
            sequences.append(sequence)
            if len(sequences) >= max_sequences:
                break

        if sequences:
            logger.info(
                "TRAKE built %d compact trace sequences for VLM: %s",
                len(sequences),
                ", ".join(
                    "%s@%.2fs" % (sequence.get("video_id"), self._sequence_span_seconds(sequence))
                    for sequence in sequences[:10]
                ),
            )
        return sequences
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
                "semantic_score": candidate["score"],
                "temporal_penalty": 1.0,
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
                    if (
                        adjacent_gap > self.min_event_gap
                        and adjacent_gap <= self.max_event_gap
                        and total_span <= self.max_sequence_span
                    ):
                        new_base_score = seq["base_score"] + self._float(candidate.get("score"))
                        penalty = math.exp(-self.temporal_decay * total_span) if total_span > 0 else 1.0
                        event_count = len(seq["frame_details"]) + 1
                        semantic_score = new_base_score / event_count
                        new_total_score = semantic_score * penalty

                        new_seq = {
                            "video_id": video_id,
                            "frames": seq["frames"] + [candidate["frame_name"]],
                            "global_frame_ids": seq["global_frame_ids"] + [candidate["global_frame_id"]],
                            "timestamps": seq["timestamps"] + [candidate_timestamp],
                            "splits": list(set(seq["splits"] + [candidate["split"]])),
                            "base_score": new_base_score,
                            "semantic_score": semantic_score,
                            "temporal_penalty": penalty,
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

    @staticmethod
    def _strip_accents(text: str) -> str:
        return "".join(
            char
            for char in unicodedata.normalize("NFKD", str(text or ""))
            if not unicodedata.combining(char)
        )

    def _prefers_compact_sequence(self, events: List[str], shared_context: str = "") -> bool:
        text = " ".join([shared_context or "", *events]).lower()
        plain = self._strip_accents(text)
        phrases = (
            "lien tiep",
            "lien tuc",
            "ke tiep",
            "ngay sau",
            "consecutive",
            "consecutively",
            "successive",
            "in sequence",
            "one after another",
        )
        return any(phrase in plain for phrase in phrases)

    def _sequence_span_seconds(self, sequence: Dict) -> float:
        timestamps = [self._float(value) for value in sequence.get("timestamps", [])]
        if len(timestamps) < 2:
            return 0.0
        return max(timestamps) - min(timestamps)

    def _apply_consecutive_sequence_prior(
        self,
        sequences: List[Dict],
        events: List[str],
        shared_context: str = "",
    ) -> bool:
        """Prefer short-span sequences only when the query asks for consecutive scenes."""
        if not sequences or not self._prefers_compact_sequence(events, shared_context):
            return False

        preferred_span = self.consecutive_compact_span
        spans = [self._sequence_span_seconds(sequence) for sequence in sequences]
        has_compact = any(span <= preferred_span for span in spans)
        if not has_compact:
            logger.info(
                "TRAKE consecutive prior detected but no sequence fits compact span %.2fs; keeping wide temporal ranking.",
                preferred_span,
            )
            return False

        for sequence, span in zip(sequences, spans):
            excess = max(0.0, span - preferred_span)
            compact_penalty = math.exp(-self.consecutive_span_decay * excess) if excess > 0 else 1.0
            sequence["sequence_span_seconds"] = span
            sequence["compact_sequence_penalty"] = compact_penalty
            sequence["total_score"] = self._float(sequence.get("total_score")) * compact_penalty

        logger.info(
            "TRAKE consecutive prior applied: preferred_span=%.2fs decay=%.4f compact_sequences=%d/%d",
            preferred_span,
            self.consecutive_span_decay,
            sum(1 for span in spans if span <= preferred_span),
            len(sequences),
        )
        return True


    @staticmethod
    def _sequence_passes_vlm(
        sequence: Dict,
        threshold: float,
        require_match: bool,
        event_count: int | None = None,
    ) -> bool:
        score = sequence.get("vlm_score")
        if score is None:
            return False
        decision = str(sequence.get("vlm_decision") or "")
        passes = float(score) >= threshold
        if require_match:
            if not passes or decision != "match":
                return False
            if event_count is None:
                return True
            expected_events = set(range(1, event_count + 1))
            try:
                matched_events = {int(value) for value in (sequence.get("vlm_matched_events") or [])}
                missing_events = {int(value) for value in (sequence.get("vlm_missing_events") or [])}
            except (TypeError, ValueError):
                return False
            return matched_events == expected_events and not missing_events
        return passes and decision != "wrong"

    @staticmethod
    def _sequence_sort_score(sequence: Dict) -> float:
        return TRAKE._float(sequence.get("verification_score", sequence.get("total_score", 0.0)))

    def _verify_ranked_sequences_until_match(
        self,
        verify_func,
        ranked_sequences: List[Dict],
        events: List[str],
        shared_context: str,
        threshold: float,
        require_match: bool,
    ) -> tuple[List[Dict], Dict[str, Any]]:
        """Verify ranked TRAKE sequences in batches until one passes or budget is exhausted."""
        if not ranked_sequences:
            return ranked_sequences, {"enabled": False, "status": "disabled", "evaluated": 0}

        max_total = max(int(self.settings.trake_vlm_max_sequences), self.vlm_max_total_sequences)
        total_evaluated = 0
        total_requested = 0
        total_missing_images = 0
        rounds = 0
        errors: List[str] = []
        contract_errors: List[str] = []
        payload_previews: List[str] = []
        last_summary: Dict[str, Any] = {"enabled": True, "status": "fallback", "evaluated": 0}
        logged_sequences: set[int] = set()

        while total_evaluated < max_total:
            remaining = [sequence for sequence in ranked_sequences if sequence.get("vlm_score") is None]
            if not remaining:
                break
            remaining_budget = max_total - total_evaluated
            batch_input = remaining[:remaining_budget]
            rounds += 1
            batch_sequences, batch_summary = verify_func(
                batch_input,
                events,
                self._resolve_image_path,
                shared_context=shared_context,
            )
            last_summary = batch_summary
            evaluated = int(batch_summary.get("evaluated") or 0)
            requested = int(batch_summary.get("requested") or evaluated or 0)
            total_evaluated += evaluated
            total_requested += requested
            if batch_summary.get("missing_images") is not None:
                total_missing_images += int(batch_summary.get("missing_images") or 0)
            errors.extend(str(item) for item in (batch_summary.get("errors") or []))
            contract_errors.extend(str(item) for item in (batch_summary.get("contract_errors") or []))
            preview = str(batch_summary.get("payload_preview") or "")
            if preview:
                payload_previews.append(preview)

            logger.info(
                "TRAKE sequence VLM round %d summary: status=%s evaluated=%s requested=%s missing_images=%s errors=%s contract_errors=%s payload=%s",
                rounds,
                batch_summary.get("status"),
                batch_summary.get("evaluated"),
                batch_summary.get("requested"),
                batch_summary.get("missing_images"),
                batch_summary.get("errors") or [],
                batch_summary.get("contract_errors") or [],
                str(batch_summary.get("payload_preview") or "")[:240],
            )
            for sequence in batch_sequences:
                if sequence.get("vlm_score") is None or id(sequence) in logged_sequences:
                    continue
                logged_sequences.add(id(sequence))
                logger.info(
                    "TRAKE VLM verdict: video=%s timestamps=%s decision=%s score=%.3f matched=%s missing=%s reason=%s",
                    sequence.get("video_id"),
                    sequence.get("timestamps"),
                    sequence.get("vlm_decision"),
                    float(sequence.get("vlm_score") or 0.0),
                    sequence.get("vlm_matched_events"),
                    sequence.get("vlm_missing_events"),
                    str(sequence.get("vlm_reason") or "")[:180],
                )

            if not batch_summary.get("enabled"):
                break
            if evaluated <= 0:
                break
            if any(self._sequence_passes_vlm(sequence, threshold, require_match, len(events)) and self._sequence_meets_compact_requirement(sequence, events, shared_context) for sequence in ranked_sequences):
                break

        compact_required = self._prefers_compact_sequence(events, shared_context)
        ranked_sequences.sort(
            key=lambda sequence: (
                sequence.get("verification_score") is not None,
                (not compact_required) or self._sequence_meets_compact_requirement(sequence, events, shared_context),
                self._sequence_sort_score(sequence),
            ),
            reverse=True,
        )
        summary = {
            "enabled": bool(last_summary.get("enabled")),
            "status": "verified" if total_evaluated > 0 else last_summary.get("status", "fallback"),
            "evaluated": total_evaluated,
            "requested": total_requested or last_summary.get("requested"),
            "missing_images": total_missing_images if total_evaluated > 0 else last_summary.get("missing_images"),
            "rounds": rounds,
            "max_total": max_total,
            "errors": errors[:3],
            "contract_errors": contract_errors[:6],
            "payload_preview": payload_previews[0] if payload_previews else "",
        }
        if total_evaluated >= max_total and not any(
            self._sequence_passes_vlm(sequence, threshold, require_match, len(events))
            and self._sequence_meets_compact_requirement(sequence, events, shared_context)
            for sequence in ranked_sequences
        ):
            summary["exhausted"] = True
        return ranked_sequences, summary


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

        # Reuse the shared resolver so Azure/S3-backed keyframes are fetched
        # into the local cloud-assets cache before sequence VLM verification.
        try:
            from src.services.openrouter_vlm_verifier import resolve_keyframe_path

            resolved = resolve_keyframe_path(frame_detail)
            if resolved is not None:
                return str(resolved)
        except Exception as exc:
            logger.debug("TRAKE cloud keyframe resolution failed: %s", exc)

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

        raw_path = str(frame_detail.get("image_path") or "")
        if raw_path:
            stem_digits = self._digits_to_int(os.path.splitext(os.path.basename(raw_path))[0])
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

        # Last-resort legacy fallback. Normal Jina/BEiT3 responses already
        # carry a source-frame ID or frame path, so this never fetches a cloud
        # asset on the ordinary API formatting path.
        resolved_path = self._resolve_image_path(frame_detail)
        if resolved_path:
            stem_digits = self._digits_to_int(os.path.splitext(os.path.basename(resolved_path))[0])
            if stem_digits is not None:
                return stem_digits
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

    @staticmethod
    def _image_mime_type(path: str) -> str:
        """Return the browser MIME type for a locally resolved keyframe."""
        extension = os.path.splitext(str(path or ""))[1].lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(extension, "image/jpeg")

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
                submission_frame_id = self._resolve_submission_frame_id(frame_detail)
                submission_ids.append(submission_frame_id)
                raw_path = str(frame_detail.get("image_path") or "").replace("\\", "/")
                frame_key = os.path.splitext(os.path.basename(raw_path))[0] if raw_path else frame_detail.get("global_frame_id")

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
                    # The frontend serves the source through /keyframes. Azure
                    # keyframe filename extensions are not trustworthy, so a
                    # MIME-labelled base64 data URI can fail to decode.
                    "image": "",
                    "image_mime": self._image_mime_type(raw_path),
                    "frame_path": raw_path,
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
                "semantic_score": sequence.get("semantic_score", sequence.get("total_score", 0.0)),
                "temporal_penalty": sequence.get("temporal_penalty", 1.0),
                "temporal_gaps": sequence.get("temporal_gaps", []),
                "sequence_span_seconds": sequence.get("sequence_span_seconds"),
                "compact_sequence_penalty": sequence.get("compact_sequence_penalty"),
                "trace_compact_selected": sequence.get("trace_compact_selected"),
                "noncompact_vlm_match": sequence.get("noncompact_vlm_match"),
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
        shared_context = next(
            (str(q.get("context") or "").strip() for q in queries if str(q.get("context") or "").strip()),
            "",
        )

        # NOTE: use ``logger`` here, never ``print``. Event text is Vietnamese and
        # a bare ``print`` to a cp1252 Windows stdout raises UnicodeEncodeError,
        # which would surface as an unhandled 500 (and, lacking CORS headers, as a
        # "service unavailable" transport error in the browser).
        logger.info("TRAKE processing %d events", len(events))

        retrieval_top_k = max(top_k, int(self.settings.trake_retrieval_top_k))
        retrieve_override = getattr(self.retrieve_top_k, "__func__", None) is not TRAKE.retrieve_top_k

        def _retrieve_and_enhance(limit: int) -> List[List[Dict]]:
            batches = (
                [self.retrieve_top_k(event, limit) for event in events]
                if retrieve_override
                else self.retrieve_events_batch(events, limit)
            )
            # Tests and integrations that override retrieve_top_k provide the
            # complete candidate universe themselves; do not load a live
            # retriever merely to add an anchor pass in that mode.
            if not retrieve_override:
                batches = self._expand_anchor_videos(events, batches)
            return [self._rerank_event_candidates(event, candidates) for event, candidates in zip(events, batches)]

        candidates_list = (
            _retrieve_and_enhance(retrieval_top_k)
        )
        self._log_candidate_trace(candidates_list)
        for i, (event, candidates) in enumerate(zip(events, candidates_list)):
            logger.info("Retrieved event %d (%.60s): %d candidates", i + 1, event, len(candidates))

        video_groups = self.group_by_video(candidates_list)
        logger.info("Grouped candidates into %d videos", len(video_groups))

        complete_videos = sum(
            1 for event_lists in video_groups.values()
            if len(event_lists) == len(events) and all(event_lists)
        )
        adaptive_top_k = max(
            retrieval_top_k,
            int(self.settings.trake_adaptive_retrieval_top_k),
        )
        adaptive_recall_used = False
        if complete_videos == 0 and adaptive_top_k > retrieval_top_k:
            logger.info(
                "No video covers all %d events at top_k=%d; retrying with top_k=%d",
                len(events), retrieval_top_k, adaptive_top_k,
            )
            candidates_list = _retrieve_and_enhance(adaptive_top_k)
            self._log_candidate_trace(candidates_list)
            video_groups = self.group_by_video(candidates_list)
            complete_videos = sum(
                1 for event_lists in video_groups.values()
                if len(event_lists) == len(events) and all(event_lists)
            )
            logger.info(
                "Adaptive recall grouped candidates into %d videos; %d cover every event",
                len(video_groups), complete_videos,
            )
            adaptive_recall_used = True

        valid_sequences = self.find_valid_sequences(video_groups, len(events))
        logger.info("Beam search produced %d candidate sequences", len(valid_sequences))

        # A video may contain candidates for every event while none of their
        # timestamps form a valid ordered sequence. Treat that as a recall miss
        # too; complete-video coverage alone is not sufficient for TRAKE.
        if not valid_sequences and not adaptive_recall_used and adaptive_top_k > retrieval_top_k:
            logger.info(
                "No ordered sequence at top_k=%d; retrying all %d events with top_k=%d",
                retrieval_top_k, len(events), adaptive_top_k,
            )
            candidates_list = _retrieve_and_enhance(adaptive_top_k)
            self._log_candidate_trace(candidates_list)
            video_groups = self.group_by_video(candidates_list)
            complete_videos = sum(
                1 for event_lists in video_groups.values()
                if len(event_lists) == len(events) and all(event_lists)
            )
            logger.info(
                "Adaptive recall grouped candidates into %d videos; %d cover every event",
                len(video_groups), complete_videos,
            )
            valid_sequences = self.find_valid_sequences(video_groups, len(events))
            logger.info(
                "Adaptive beam search produced %d candidate sequences",
                len(valid_sequences),
            )

        trace_video_order = self._candidate_trace_ranked_videos(candidates_list)

        n_events = len(events)
        valid_sequences = [
            sequence for sequence in valid_sequences if self._sequence_is_valid(sequence, n_events)
        ]
        logger.info(
            "%d sequences pass the contract (all events, one video, ordered, resolvable frame ids)",
            len(valid_sequences),
        )

        compact_trace_sequences = self._build_compact_trace_sequences(
            candidates_list,
            trace_video_order,
            events,
            shared_context,
            max_sequences=max(1, int(getattr(self.settings, "trake_vlm_max_total_sequences", 32))),
        )
        if compact_trace_sequences:
            existing_sequence_ids = {
                (str(sequence.get("video_id") or ""), tuple(sequence.get("global_frame_ids") or []))
                for sequence in valid_sequences
            }
            additions = []
            for sequence in compact_trace_sequences:
                identity = (str(sequence.get("video_id") or ""), tuple(sequence.get("global_frame_ids") or []))
                if identity in existing_sequence_ids:
                    continue
                existing_sequence_ids.add(identity)
                additions.append(sequence)
            if additions:
                valid_sequences = additions + valid_sequences
                logger.info("TRAKE added %d compact trace sequences before VLM ranking", len(additions))

        self._apply_consecutive_sequence_prior(valid_sequences, events, shared_context)

        if not valid_sequences:
            return []

        verification_pool_size = min(
            len(valid_sequences),
            max(
                top_results,
                int(self.settings.trake_vlm_max_sequences) * 50,
                300,
            ),
        )
        ranked_sequences = self.rank_sequences(valid_sequences, verification_pool_size)
        ranked_sequences = self._prioritise_trace_video_sequences_for_verification(
            ranked_sequences,
            valid_sequences,
            trace_video_order,
            max_promoted=min(
                max(1, int(getattr(self.settings, "trake_trace_video_limit", 20))),
                max(1, int(getattr(self.settings, "trake_vlm_max_total_sequences", 32))),
            ),
            events=events,
            shared_context=shared_context,
        )
        logger.info(
            "Ranked %d TRAKE sequences for verification pool; requested output top_results=%d",
            len(ranked_sequences),
            top_results,
        )

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
                old_score = seq["total_score"]
                seq["total_score"] = (old_score * 0.7) + (avg_vqa * 0.3)
                seq["vqa_confidence"] = avg_vqa

            ranked_sequences.sort(key=lambda x: x["total_score"], reverse=True)
        else:
            logger.debug("Skipping BLIP-VQA validation (set TRAKE_ENABLE_VQA=true to enable).")

        threshold = max(0.0, min(float(self.settings.agent_min_verification_score), 1.0))
        require_match = bool(self.settings.agent_require_vlm_match)
        try:
            from src.services.openrouter_trake_verifier import verify_trake_sequences

            ranked_sequences, vlm_summary = self._verify_ranked_sequences_until_match(
                verify_trake_sequences,
                ranked_sequences,
                events,
                shared_context,
                threshold,
                require_match,
            )
            logger.info(
                "TRAKE sequence VLM summary: status=%s evaluated=%s requested=%s missing_images=%s rounds=%s max_total=%s",
                vlm_summary.get("status"),
                vlm_summary.get("evaluated"),
                vlm_summary.get("requested"),
                vlm_summary.get("missing_images"),
                vlm_summary.get("rounds"),
                vlm_summary.get("max_total"),
            )
        except Exception as exc:
            logger.warning("TRAKE sequence VLM verification failed; keeping temporal ranking: %s", exc)
            vlm_summary = {"enabled": True, "status": "fallback", "evaluated": 0, "error": str(exc)[:180]}

        promote_trace_candidates = False
        rejected_videos: set[str] = set()
        if int(vlm_summary.get("evaluated") or 0) > 0:
            accepted = []
            noncompact_vlm_matches = []
            deferred = []
            rejected = []
            for sequence in ranked_sequences:
                decision = str(sequence.get("vlm_decision") or "")
                score = sequence.get("vlm_score")
                evaluated = score is not None
                if not evaluated:
                    deferred.append(sequence)
                    if not require_match:
                        accepted.append(sequence)
                    continue

                vlm_passes = self._sequence_passes_vlm(sequence, threshold, require_match, len(events))
                compact_passes = self._sequence_meets_compact_requirement(sequence, events, shared_context)
                if vlm_passes and compact_passes:
                    accepted.append(sequence)
                elif vlm_passes:
                    sequence["noncompact_vlm_match"] = True
                    noncompact_vlm_matches.append(sequence)
                    rejected.append(sequence)
                else:
                    rejected.append(sequence)

            vlm_summary["minimum_score"] = threshold
            vlm_summary["require_match"] = require_match
            vlm_summary["rejected"] = len(rejected)
            vlm_summary["deferred_unverified"] = len(deferred)
            rejected_videos = set()
            if accepted:
                ranked_sequences = accepted
            elif noncompact_vlm_matches:
                vlm_summary["fallback"] = "noncompact_vlm_match_after_no_compact_match"
                vlm_summary["noncompact_vlm_matches"] = len(noncompact_vlm_matches)
                logger.warning(
                    "TRAKE found %d VLM matches but none fit compact span %.2fs; returning best noncompact VLM match instead of unverified fallback.",
                    len(noncompact_vlm_matches),
                    float(self.consecutive_compact_span),
                )
                ranked_sequences = noncompact_vlm_matches
            elif deferred:
                vlm_summary["fallback"] = "unverified_pool_after_all_vlm_checked_sequences_rejected"
                logger.warning(
                    "TRAKE VLM rejected all %d checked sequences; returning %d unverified deferred sequences for inspection.",
                    int(vlm_summary.get("evaluated") or 0),
                    len(deferred),
                )
                ranked_sequences = deferred
                promote_trace_candidates = True
            else:
                ranked_sequences = []
        elif vlm_summary.get("enabled"):
            vlm_summary["minimum_score"] = threshold
            vlm_summary["require_match"] = require_match
            if str(vlm_summary.get("status") or "") == "fallback":
                vlm_summary["fallback"] = "verification_unavailable_returning_temporal_pool"
                vlm_summary["rejected"] = 0
                logger.warning(
                    "TRAKE VLM verification produced no usable verdicts; returning %d temporal-ranked sequences for inspection.",
                    len(ranked_sequences),
                )
                promote_trace_candidates = True
            elif require_match:
                vlm_summary["rejected"] = len(ranked_sequences)
                ranked_sequences = []

        if promote_trace_candidates:
            ranked_sequences = self._promote_trace_video_sequences(
                ranked_sequences,
                valid_sequences,
                trace_video_order,
                max_promoted=min(
                    top_results,
                    max(1, int(getattr(self.settings, "trake_trace_video_limit", 20))),
                ),
                rejected_videos=rejected_videos,
            )

            ranked_sequences = self._diversify_fallback_sequences_by_video(ranked_sequences)
            vlm_summary["fallback_video_diversified"] = True
        ranked_sequences = ranked_sequences[:top_results]

        for sequence in ranked_sequences:
            sequence["verification"] = {
                "method": "openrouter_sequence_vlm" if sequence.get("vlm_score") is not None else "temporal_evidence",
                "summary": vlm_summary,
            }

        logger.info("Formatting %d ranked TRAKE sequences", len(ranked_sequences))
        return self.format_response(ranked_sequences)
