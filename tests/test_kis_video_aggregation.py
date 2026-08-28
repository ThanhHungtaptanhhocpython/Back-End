import os
import unittest
from unittest.mock import patch

from src.services.deep_keyframe_search import _event_video_search, _temporal_events


class KISVideoAggregationTests(unittest.TestCase):
    def test_roi_den_is_not_split_as_temporal_connector(self):
        events = _temporal_events(
            "Một người đứng dưới nước và rọi đèn. Tiếp theo là cảnh người này kéo lưới, "
            "sau đó một nhóm người tiến đến quay phim."
        )
        self.assertEqual(len(events), 3)
        self.assertIn("rọi đèn", events[0])

    @patch.dict(
        os.environ,
        {
            "KIS_EVENT_RECALL_K": "3",
            "KIS_VIDEO_RERANK_VIDEOS": "2",
            "KIS_VQA_FRAMES_PER_EVENT": "1",
        },
        clear=False,
    )
    @patch("src.services.deep_keyframe_search._resolve_keyframe_path", return_value="")
    @patch("src.services.deep_keyframe_search._image_content_key", side_effect=lambda item: item["frame_path"])
    @patch("src.services.deep_keyframe_search.getImageDataSingleTextSearch")
    def test_full_ordered_video_is_ranked_before_reversed_video(self, search, _content_key, _path):
        def frame(video_id, frame_id, timestamp, score):
            return {
                "video_id": video_id,
                "frame_id": frame_id,
                "global_frame_id": int(frame_id),
                "frame_path": f"L00_a/{video_id}/{frame_id}.webp",
                "timestamp": timestamp,
                "score": score,
            }

        search.side_effect = [
            [frame("L00_GOOD", "000010", 10, 0.8), frame("L00_BAD", "000030", 30, 0.9)],
            [frame("L00_GOOD", "000020", 20, 0.8), frame("L00_BAD", "000020", 20, 0.9)],
            [frame("L00_GOOD", "000030", 30, 0.8), frame("L00_BAD", "000010", 10, 0.9)],
        ]

        result = _event_video_search(["event one", "event two", "event three"], topk=3, per_query=3)

        self.assertEqual(result["video_results"][0]["video_id"], "L00_GOOD")
        self.assertTrue(result["video_results"][0]["ordered"])
        self.assertEqual([row["timestamp"] for row in result["video_results"][0]["evidence_frames"]], [10.0, 20.0, 30.0])


if __name__ == "__main__":
    unittest.main()