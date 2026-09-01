import assert from "node:assert/strict";
import test from "node:test";

import {
  buildSubmissionCsv,
  makeSubmissionZip,
  makeUtf8CsvBlob,
  sanitizeQueryFileName,
} from "../src/shared/submissionExport.js";

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
test("QA export uses fallback answer when result answer is empty", () => {
  const csv = buildSubmissionCsv([
    { videoKey: "L26_V006", submissionFrameId: "000599", answer: "" },
  ], "qa", "5");

  assert.equal(csv, 'L26_V006,599,"5"');
});

test("QA manually entered answer overrides model answers on every row", () => {
  const csv = buildSubmissionCsv([
    { videoKey: "L01_V001", submissionFrameId: 10, answer: "wrong model answer" },
    { videoKey: "L01_V002", submissionFrameId: 20, answer: "another model answer" },
  ], "qa", "Năm người");

  assert.equal(csv, 'L01_V001,10,"Năm người"\r\nL01_V002,20,"Năm người"');
});

test("QA export keeps spaces, escapes CSV characters, and limits answers to 100 Unicode characters", () => {
  const longAnswer = `${"😀".repeat(99)},extra`;
  const csv = buildSubmissionCsv([
    { videoKey: "L01_V003.mp4", submissionFrameId: 30 },
  ], "qa", longAnswer);
  const expectedAnswer = `${"😀".repeat(99)},`;

  assert.equal(csv, `L01_V003,30,"${expectedAnswer}"`);
  assert.equal(Array.from(expectedAnswer).length, 100);
  assert.equal(
    buildSubmissionCsv([{ videoKey: "L01_V003", submissionFrameId: 30 }], "qa", "  Năm người  "),
    'L01_V003,30,"  Năm người  "'
  );
});

test("submission CSV never exports more than 100 rows", () => {
  const items = Array.from({ length: 101 }, (_, index) => ({
    videoKey: "L01_V001",
    submissionFrameId: index + 1,
  }));

  assert.equal(buildSubmissionCsv(items, "qa", "5").split("\r\n").length, 100);
});

test("standalone CSV blob starts with UTF-8 BOM for Excel and preserves Vietnamese", async () => {
  const blob = makeUtf8CsvBlob('L02_V011,1200,"Năm người"');
  const bytes = new Uint8Array(await blob.arrayBuffer());

  assert.deepEqual(Array.from(bytes.slice(0, 3)), [0xef, 0xbb, 0xbf]);
  assert.equal(new TextDecoder().decode(bytes.slice(3)), 'L02_V011,1200,"Năm người"');
});

test("submission ZIP stores official CSV as BOM-free UTF-8 inside submission folder", async () => {
  const content = 'L03_V005,2800,"Màu đỏ, rất đẹp"';
  const zip = makeSubmissionZip([{ name: "query-p1-3-qa.csv", content, queryType: "qa" }]);
  const bytes = new Uint8Array(await zip.arrayBuffer());
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  let offset = 0;
  const entries = [];
  while (view.getUint32(offset, true) === 0x04034b50) {
    const flags = view.getUint16(offset + 6, true);
    const size = view.getUint32(offset + 18, true);
    const nameLength = view.getUint16(offset + 26, true);
    const extraLength = view.getUint16(offset + 28, true);
    const nameStart = offset + 30;
    const contentStart = nameStart + nameLength + extraLength;
    entries.push({
      name: new TextDecoder().decode(bytes.slice(nameStart, nameStart + nameLength)),
      flags,
      content: bytes.slice(contentStart, contentStart + size),
    });
    offset = contentStart + size;
  }

  assert.equal(entries[1].name, "submission/query-p1-3-qa.csv");
  assert.equal(entries[1].flags & 0x0800, 0x0800);
  assert.notDeepEqual(Array.from(entries[1].content.slice(0, 3)), [0xef, 0xbb, 0xbf]);
  assert.equal(new TextDecoder("utf-8", { fatal: true }).decode(entries[1].content), content);
});

test("sanitizeQueryFileName keeps csv extension and strips unsafe path characters", () => {
  assert.equal(sanitizeQueryFileName('query-p1-17-kis?.csv', 'kis'), 'query-p1-17-kis_.csv');
});
