import assert from "node:assert/strict";
import test from "node:test";

import { buildSubmissionCsv, sanitizeQueryFileName } from "../src/shared/submissionExport.js";

test("KIS export prefers per-video frame id over FAISS/vector id", () => {
  const csv = buildSubmissionCsv([
    {
      videoKey: "L30_V017",
      globalFrameId: 277466,
      faissIndex: 277466,
      backend: {
        video_id: "L30_V017",
        vector_id: 277466,
        global_frame_id: 277466,
        frame_id: "003048",
      },
    },
  ], "kis");

  assert.equal(csv, "L30_V017,3048");
});

test("KIS export uses explicit submissionFrameId first", () => {
  const csv = buildSubmissionCsv([
    {
      videoKey: "L21_V024",
      submissionFrameId: 20616,
      globalFrameId: 14728,
      backend: { frame_id: "000001" },
    },
  ], "kis");

  assert.equal(csv, "L21_V024,20616");
});

test("QA export quotes answer cells", () => {
  const csv = buildSubmissionCsv([
    { videoKey: "L21_V001", submissionFrameId: "000660", answer: 'a "quoted", answer' },
  ], "qa");

  assert.equal(csv, 'L21_V001,660,"a ""quoted"", answer"');
});

test("sanitizeQueryFileName keeps csv extension and strips unsafe path characters", () => {
  assert.equal(sanitizeQueryFileName('query-p1-17-kis?.csv', 'kis'), 'query-p1-17-kis_.csv');
});