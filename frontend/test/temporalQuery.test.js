import assert from "node:assert/strict";
import test from "node:test";

import {
  buildTemporalEventQueries,
  isRunnableTemporalQuery,
  parseTemporalQuery,
  serializeTemporalQuery,
} from "../src/shared/temporalQuery.js";

test("splits plain E1..E4 lines into exactly four events with no context", () => {
  const parsed = parseTemporalQuery(
    [
      "E1: Khoảnh khắc đầu tiên bột được bỏ vào tô măng tây.",
      "E2: Khoảnh khắc đầu tiên thấy miến măng tây đầu tiên tiếp xúc với dầu trong chảo.",
      "E3: Khoảnh khắc miếng măng tây đầu tiên rời khỏi chảo dầu.",
      "E4: Khoảng khắc miếng măng tây cuối cùng rời chảo dầu và nằm hoàn toàn trên dĩa.",
    ].join("\n"),
  );

  assert.equal(parsed.events.length, 4);
  assert.equal(parsed.context, "");
  assert.equal(parsed.warnings.length, 0);
  assert.match(parsed.events[0], /^Khoảnh khắc đầu tiên bột/);
});

test("keeps a leading scene description as context, not an event", () => {
  const parsed = parseTemporalQuery(
    [
      "Đoạn video múa lân một con lân màu vàng đen trắng, tìm các sự kiện sau:",
      "E1: Lân quay vòng trên cột số 4 bằng 2 chân trước rồi tiếp đất. Khoảnh khắc đầu tiên mà lân bắt đầu xoay vòng.",
      "E2: Khoảnh khắc 4 chân hoàn toàn chạm đất đầu tiên.",
      "E3: Khoảnh khắc đầu tiên 2 người biểu diễn lân cuối chào ban giám khảo.",
      "E4: Sau đó lân tiến lại chào một con rồng. Khoảnh khắc đầu tiên con rồng cử động đầu.",
    ].join("\n"),
  );

  assert.equal(parsed.events.length, 4);
  assert.ok(parsed.context.startsWith("Đoạn video múa lân"));
  assert.doesNotMatch(parsed.context, /tìm các sự kiện sau/i);
  assert.equal(parsed.warnings.length, 0);
});

test("tolerates repeated / skipped E labels (E1,E2,E2,E4) and a context lead-in", () => {
  const parsed = parseTemporalQuery(
    [
      "Trong đoạn video nấu ăn một món ăn về nấm, gồm các khoảnh khắc sơ chế:",
      "E1: Khoảnh khắc đầu tiên thấy cắt nấm.",
      "E2: Khoảnh khắc đầu tiên cắt củ năng.",
      "E2: Khoảnh khắc đầu tiên cắt đậu hủ.",
      "E4: Khoảnh khắc chảo đặt lên bếp, đầu bếp mở lửa và thấy lửa bắt đầu xuất hiện",
    ].join("\n"),
  );

  assert.equal(parsed.events.length, 4);
  assert.ok(parsed.context.startsWith("Trong đoạn video nấu ăn"));
  assert.doesNotMatch(parsed.context, /:\s*$/);
});

test("splits E-labels that have no separator (E1 text, not E1: text)", () => {
  const parsed = parseTemporalQuery(
    [
      "Đoạn video bắt đầu bằng ảnh cận đầu một con lân trắng, mũi đỏ, bên cạnh lá cờ trắng viền đỏ.",
      "E1 Khoảnh khắc đầu tiên xuất hiện đầy đủ hai con rồng vàng đang xoay vòng.",
      "E2 Khoảnh khắc đầu tiên con lân hoàn tất cú xoay người trên các thanh trụ (thời điểm đâu tiên các chân của lân đặt trên trụ sau khi xoay).",
      "E3 Khoảnh khắc đầu tiên dùi chạm vào kẻng đồng múa lân.",
    ].join("\n"),
  );

  assert.equal(parsed.events.length, 3);
  assert.ok(parsed.context.startsWith("Đoạn video bắt đầu bằng ảnh cận đầu"));
  assert.match(parsed.events[0], /^Khoảnh khắc đầu tiên xuất hiện đầy đủ hai con rồng/);
  assert.match(parsed.events[1], /thanh trụ .*sau khi xoay\)\.$/);
  assert.match(parsed.events[2], /dùi chạm vào kẻng đồng/);
  assert.equal(parsed.warnings.length, 0);
});

test("reads a query-p<phase>-<id>-trake header line", () => {
  const parsed = parseTemporalQuery(
    ["query-p2-8-trake:", "E1: người mở cửa xe", "E2: người bước ra khỏi xe"].join("\n"),
  );
  assert.deepEqual(parsed.header, { phase: 2, id: 8, type: "trake" });
  assert.equal(parsed.events.length, 2);
});

test("header text on the same line is still parsed as body", () => {
  const parsed = parseTemporalQuery("query-p1-4-trake: E1: a happens\nE2: b happens");
  assert.equal(parsed.header.id, 4);
  assert.deepEqual(parsed.events, ["a happens", "b happens"]);
});

test("splits a single connector line into ordered events", () => {
  const parsed = parseTemporalQuery("người đàn ông mở cửa -> sau đó bước vào phòng -> rồi ngồi xuống ghế");
  assert.deepEqual(parsed.events, ["người đàn ông mở cửa", "bước vào phòng", "ngồi xuống ghế"]);
});

test("numbered lists need at least two items", () => {
  const parsed = parseTemporalQuery("1. first moment\n2) second moment\n3. third moment");
  assert.deepEqual(parsed.events, ["first moment", "second moment", "third moment"]);
});

test("warns when fewer than two events are found", () => {
  const parsed = parseTemporalQuery("chỉ có một mô tả cảnh không thể tách");
  assert.ok(parsed.warnings.length >= 1);
  assert.equal(isRunnableTemporalQuery(parsed), false);
});

test("buildTemporalEventQueries folds context into every event", () => {
  const parsed = parseTemporalQuery(
    ["Bối cảnh: video nấu ăn về nấm", "E1: cắt nấm", "E2: bắc chảo lên bếp"].join("\n"),
  );
  const queries = buildTemporalEventQueries(parsed);
  assert.equal(queries.length, 2);
  assert.ok(queries[0].includes("video nấu ăn về nấm"));
  assert.ok(queries[0].includes("cắt nấm"));
  assert.ok(queries[1].includes("video nấu ăn về nấm"));
});

test("serializeTemporalQuery round-trips through the parser", () => {
  const text = serializeTemporalQuery({
    context: "video múa lân",
    events: ["lân bắt đầu xoay vòng", "bốn chân chạm đất", "chào ban giám khảo"],
  });
  const parsed = parseTemporalQuery(text);
  assert.equal(parsed.context, "video múa lân");
  assert.deepEqual(parsed.events, ["lân bắt đầu xoay vòng", "bốn chân chạm đất", "chào ban giám khảo"]);
  assert.equal(isRunnableTemporalQuery(parsed), true);
});
