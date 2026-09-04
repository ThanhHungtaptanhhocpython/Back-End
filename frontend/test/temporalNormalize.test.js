import assert from "node:assert/strict";
import test from "node:test";

import {
  looksLikeTemporalPayload,
  normalizeTemporalResponse,
  reindexTemporalSequences,
  resolveSubmissionFrameId,
  temporalSequenceToExportRow,
} from "../src/shared/temporalNormalize.js";

const payload = {
  success: true,
  data: {
    total_items: 2,
    items: [
      {
        sequence_id: "L21_V001#100-220-540",
        video_id: "L21_V001",
        score: 0.81,
        verification_score: 0.9,
        timestamps: [4.0, 8.8, 21.6],
        temporal_gaps: [4.8, 12.8],
        vlm_decision: "match",
        vlm_matched_events: [1, 2, 3],
        verification: { method: "openrouter_sequence_vlm", summary: { status: "verified" } },
        event_queries: ["bột vào tô", "miến gặp dầu", "miến rời chảo"],
        frames: [
          { event_index: 1, video_key: "L21_V001", frame_key: "000100", submission_frame_id: 100, timestamp: 4.0, image: "aaa", image_mime: "image/jpeg", vector_id: 918273 },
          { event_index: 2, video_key: "L21_V001", frame_key: "000220", submission_frame_id: 220, timestamp: 8.8, image: "bbb", image_mime: "image/png", vector_id: 918280 },
          { event_index: 3, video_key: "L21_V001", frame_key: "000540", submission_frame_id: 540, timestamp: 21.6, image: "ccc", image_mime: "image/webp", vector_id: 918290 },
        ],
      },
      {
        video_id: "L21_V002",
        timestamps: [2, 1],
        frames: [
          { event_index: 1, video_key: "L21_V002", frame_key: "000030", submission_frame_id: 30, timestamp: 2 },
          { event_index: 2, video_key: "L21_V002", frame_key: "000010", submission_frame_id: 10, timestamp: 1 },
        ],
      },
    ],
  },
};

test("keeps sequences nested instead of flattening into frame cards", () => {
  const result = normalizeTemporalResponse(payload, { latency: 40 });
  assert.equal(result.type, "TEMPORAL");
  assert.equal(result.sequences.length, 2);
  assert.equal(result.sequences[0].frames.length, 3);
  assert.equal(result.sequences[0].videoKey, "L21_V001");
  assert.deepEqual(result.sequences[0].timestamps, [4.0, 8.8, 21.6]);
  assert.deepEqual(result.sequences[0].frames.map((frame) => frame.gapSeconds), [0, 4.8, 12.8]);
});

test("uses the MIME type supplied by the backend for temporal base64 previews", () => {
  const result = normalizeTemporalResponse(payload, { latency: 0 });
  assert.equal(result.sequences[0].frames[0].image, "data:image/jpeg;base64,aaa");
  assert.equal(result.sequences[0].frames[1].image, "data:image/png;base64,bbb");
  assert.equal(result.sequences[0].frames[2].image, "data:image/webp;base64,ccc");
});

test("prefers the backend keyframe route when the sequence includes a frame path", () => {
  const withPaths = structuredClone(payload);
  withPaths.data.items[0].frames[0].frame_path = "L21_a/L21_V001/keyframe_0001.jpg";
  const result = normalizeTemporalResponse(withPaths, {
    baseUrl: "http://127.0.0.1:8000/users",
  });
  assert.equal(
    result.sequences[0].frames[0].image,
    "http://127.0.0.1:8000/keyframes/L21_a/L21_V001/keyframe_0001.jpg?asset=v2",
  );
});

test("submission id is the per-video frame index, never a vector id", () => {
  const result = normalizeTemporalResponse(payload, { latency: 40 });
  const frames = result.sequences[0].frames;
  assert.deepEqual(frames.map((f) => f.submissionFrameId), [100, 220, 540]);
  assert.equal(frames[0].eventIndex, 1);
  assert.equal(frames[0].eventQuery, "bột vào tô");
});

test("marks a sequence invalid when timestamps are not increasing", () => {
  const result = normalizeTemporalResponse(payload, { latency: 40 });
  assert.equal(result.sequences[0].valid, true);
  assert.equal(result.sequences[1].valid, false);
  assert.equal(result.sequences[1].orderOk, false);
});

test("resolveSubmissionFrameId falls back to digits in the frame key", () => {
  assert.equal(resolveSubmissionFrameId({ frame_key: "003048" }), 3048);
  assert.equal(resolveSubmissionFrameId({ frame_name: "L21_V001_000660" }), 660);
  assert.equal(resolveSubmissionFrameId({}), null);
});

test("looksLikeTemporalPayload detects nested frames", () => {
  assert.equal(looksLikeTemporalPayload(payload), true);
  assert.equal(looksLikeTemporalPayload({ data: { items: [{ id: 1 }] } }), false);
});

test("reindexTemporalSequences puts a pinned (chosen) sequence first without an edit", () => {
  const result = normalizeTemporalResponse(payload, { latency: 0 });
  const [a, b] = result.sequences;
  const reordered = reindexTemporalSequences([a, { ...b, chosen: true }]);
  assert.equal(reordered[0].id, b.id);
  assert.equal(reordered[0].rank, 1);
  assert.equal(reordered[1].id, a.id);
});

test("reindexTemporalSequences puts edited sequences first and dedupes", () => {
  const result = normalizeTemporalResponse(payload, { latency: 0 });
  const [a, b] = result.sequences;
  const edited = { ...a, edited: true };
  const dupOfA = { ...a };
  const reordered = reindexTemporalSequences([b, dupOfA, edited]);
  assert.equal(reordered[0].id, edited.id);
  assert.equal(reordered[0].rank, 1);
  // dupOfA duplicates the edited sequence's video+frames tuple -> dropped
  assert.equal(reordered.length, 2);
  assert.equal(reordered[1].videoKey, "L21_V002");
});

test("temporalSequenceToExportRow returns ordered frame ids", () => {
  const result = normalizeTemporalResponse(payload, { latency: 0 });
  assert.deepEqual(temporalSequenceToExportRow(result.sequences[0]), {
    videoId: "L21_V001",
    frameIds: [100, 220, 540],
  });
});
