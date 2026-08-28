from src.utils.trake_processing import TRAKE


class MockFaiss:
    pass


def test_group_by_video_keeps_same_short_id_in_different_splits_separate():
    trake = TRAKE(MockFaiss())
    candidates = [[
        {"video_id": "V006", "split": "videos-l22-a"},
        {"video_id": "V006", "split": "videos-l29-a"},
    ]]

    grouped = trake.group_by_video(candidates)

    assert set(grouped) == {"L22_V006", "L29_V006"}
    assert len(grouped["L22_V006"][0]) == 1
    assert len(grouped["L29_V006"][0]) == 1