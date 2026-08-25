import base64
import csv
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from .faiss_processing import MyFaiss
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


class TRAKE:
    def __init__(self, faiss_searcher: MyFaiss):
        """
        Initialize TRAKE system with MyFaiss.

        Args:
            faiss_searcher: Existing instance of MyFaiss.
        """
        settings = get_settings()
        self.faiss_searcher = faiss_searcher
        self.keyframes_base_path = str(settings.get_keyframes_root())
        self.map_keyframes_dir = os.path.join(str(settings.src_dir), "dict", "map-keyframes")
        self._keyframe_map_cache: Dict[str, Dict[int, int]] = {}
        self.enable_vqa = os.getenv("TRAKE_ENABLE_VQA", "false").strip().lower() in {"1", "true", "yes", "on"}
        self.vqa_max_sequences = int(os.getenv("TRAKE_VQA_MAX_SEQUENCES", "5"))

    def retrieve_top_k(self, query: str, k: int = 200) -> List[Dict]:
        """Retrieve candidates from the BEiT3 corpus used by the current keyframes.

        The old OpenCLIP index references legacy keyframe ordinals that no
        longer map reliably to files in KEYFRAMES_ROOT. BEiT3 records carry a
        frame_path from the same corpus as the on-disk images.
        """
        from src.services.beit3_retriever import get_beit3_retriever

        translated_query = Translation()(query)
        results = get_beit3_retriever().search_visual(translated_query or query, top_k=k)

        candidates = []
        for result in results:
            frame_path = str(result.get("frame_path") or "").replace("\\", "/")
            path_parts = frame_path.split("/")
            namespace = str(result.get("namespace") or (path_parts[0] if path_parts else ""))
            split = f"videos-{namespace.replace('_', '-')}" if namespace else ""
            frame_id = result.get("frame_idx", result.get("frame_id"))
            try:
                global_frame_id = int(frame_id)
            except (TypeError, ValueError):
                continue

            candidates.append({
                "faiss_idx": int(result.get("vector_id", -1)),
                "global_frame_id": global_frame_id,
                "timestamp": result.get("timestamp", 0.0) or 0.0,
                "frame_name": Path(frame_path).name or str(result.get("frame_name") or ""),
                "video_id": str(result.get("video_id") or ""),
                "split": split,
                "score": float(result.get("score", 0.0)),
                "image_path": frame_path,
            })

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

        return video_groups
    def beam_search_sequences(self, video_id: str, event_candidates: List[List[Dict]], beam_width: int = 50) -> List[Dict]:
        """
        Find top temporal sequences using beam search.
        """
        beam = []
        for candidate in event_candidates[0]:
            seq_info = {
                "video_id": video_id,
                "frames": [candidate["frame_name"]],
                "global_frame_ids": [candidate["global_frame_id"]],
                "timestamps": [candidate.get("timestamp", 0.0)],
                "splits": [candidate["split"]],
                "base_score": candidate["score"],
                "total_score": candidate["score"],
                "frame_details": [candidate],
            }
            beam.append(seq_info)

        beam.sort(key=lambda x: x["total_score"], reverse=True)
        beam = beam[:beam_width]

        for event_idx in range(1, len(event_candidates)):
            new_beam = []
            next_candidates = event_candidates[event_idx]
            next_candidates.sort(key=lambda x: x["global_frame_id"])

            for seq in beam:
                last_frame_id = seq["global_frame_ids"][-1]
                for candidate in next_candidates:
                    if candidate["global_frame_id"] > last_frame_id:
                        new_base_score = seq["base_score"] + candidate["score"]
                        time_gap = candidate.get("timestamp", 0.0) - seq["timestamps"][0]
                        alpha = 0.01
                        penalty = math.exp(-alpha * time_gap) if time_gap > 0 else 1.0
                        new_total_score = new_base_score * penalty

                        new_seq = {
                            "video_id": video_id,
                            "frames": seq["frames"] + [candidate["frame_name"]],
                            "global_frame_ids": seq["global_frame_ids"] + [candidate["global_frame_id"]],
                            "timestamps": seq["timestamps"] + [candidate.get("timestamp", 0.0)],
                            "splits": list(set(seq["splits"] + [candidate["split"]])),
                            "base_score": new_base_score,
                            "total_score": new_total_score,
                            "frame_details": seq["frame_details"] + [candidate],
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

            sequences = self.beam_search_sequences(video_id, event_candidates, beam_width=50)
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
            print(f"Error loading keyframe map {map_path}: {exc}")

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

    def _get_image_base64(self, frame_detail: Dict) -> str:
        """
        Get base64 encoded image.
        """
        try:
            full_image_path = self._resolve_image_path(frame_detail)
            with open(full_image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            print(f"Error loading image {frame_detail.get('image_path', '')}: {e}")
            return ""

    def format_response(self, sequences: List[Dict]) -> List[Dict]:
        """
        Format sequences into the required API response format.
        """
        response = []

        for seq_id, sequence in enumerate(sequences):
            frames = []

            for frame_id, frame_detail in enumerate(sequence["frame_details"]):
                folder_key = self._split_to_folder(frame_detail.get("split", ""))
                video_key = self._normalize_video_key(frame_detail.get("split", ""), frame_detail.get("video_id", ""))
                resolved_path = self._resolve_image_path(frame_detail)
                frame_key = os.path.splitext(os.path.basename(resolved_path))[0] if resolved_path else frame_detail["global_frame_id"]
                image_b64 = self._get_image_base64(frame_detail)

                frames.append({
                    "id": frame_id,
                    "folder_key": folder_key,
                    "video_key": video_key,
                    "frame_key": frame_key,
                    "timestamp": frame_detail.get("timestamp", 0.0),
                    "image": image_b64,
                })

            response.append({
                "id": seq_id,
                "video_id": sequence.get("video_id"),
                "frame_names": [frame.get("frame_name") for frame in sequence["frame_details"]],
                "timestamps": [frame.get("timestamp", 0.0) for frame in sequence["frame_details"]],
                "frames": frames,
            })

        return response

    def process_temporal_search(self, queries: List[Dict], top_k: int = 100, top_results: int = 20) -> List[Dict]:
        """
        Main function to process temporal search queries.
        """
        events = [q["query"] for q in queries]

        print(f"Processing {len(events)} events...")

        candidates_list = []
        for i, event in enumerate(events):
            print(f"Retrieving candidates for event {i + 1}: {event[:50]}...")
            candidates = self.retrieve_top_k(event, top_k)
            candidates_list.append(candidates)
            print(f"Found {len(candidates)} candidates for event {i + 1}")

        print("Grouping candidates by video...")
        video_groups = self.group_by_video(candidates_list)
        print(f"Found candidates in {len(video_groups)} videos")

        print("Finding valid temporal sequences...")
        valid_sequences = self.find_valid_sequences(video_groups, len(events))
        print(f"Found {len(valid_sequences)} valid sequences")

        if not valid_sequences:
            return []

        print("Ranking sequences...")
        ranked_sequences = self.rank_sequences(valid_sequences, top_results)

        if self.enable_vqa:
            print("Validating Top sequences with BLIP-VQA...")
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
            print("Skipping BLIP-VQA validation (set TRAKE_ENABLE_VQA=true to enable).")

        print("Formatting response...")
        return self.format_response(ranked_sequences)
