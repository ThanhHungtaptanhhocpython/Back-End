import assert from "node:assert/strict";
import test from "node:test";

import {
  BackendSearchError,
  isTransportError,
  normalizeBackendResponse,
  probeBackend,
  runAgentChat,
  runBackendSearch,
  shouldUseQaDemoFallback,
  temporalRequestBody,
} from "../src/services/backendSearch.js";

const successPayload = {
  success: true,
  data: {
    total_items: 1,
    items: [{ faiss_id: 7, video_id: "cam-01", frame_name: "cam-01_0007", timestamp: 12.5, image: "ZmFrZQ==", final_score: 0.92 }],
  },
};

test("normalizes a FastAPI BaseResponse into a workstation card", () => {
  const result = normalizeBackendResponse(successPayload, { type: "TEXT", latency: 18 });
  assert.equal(result.totalItems, 1);
  assert.equal(result.items[0].faissIndex, 7);
  assert.equal(result.items[0].frameName, "cam-01_0007");
  assert.equal(result.items[0].image, "data:image/webp;base64,ZmFrZQ==");
  assert.equal(result.items[0].score, 0.92);
});

test("detects live QA results that have no local keyframes for VLM evaluation", () => {
  const missingFrames = {
    source: "live",
    meta: { status: "uncertain", answer: "Insufficient evidence.", evaluated_frames: 0 },
  };
  const evaluatedFrames = {
    source: "live",
    meta: { status: "answered", answer: "A red bus.", evaluated_frames: 2 },
  };

  assert.equal(shouldUseQaDemoFallback({ searchType: "QA" }, missingFrames), true);
  assert.equal(shouldUseQaDemoFallback({ searchType: "QA" }, evaluatedFrames), false);
  assert.equal(shouldUseQaDemoFallback({ searchType: "TEXT" }, missingFrames), false);
});


test("normalizes submission frame id separately from FAISS vector id", () => {
  const payload = {
    success: true,
    data: {
      total_items: 1,
      items: [
        {
          vector_id: 277466,
          faiss_id: 277466,
          video_id: "L30_V017",
          global_frame_id: 277466,
          frame_id: "003048",
          frame_path: "L30_a/L30_V017/003048.webp",
        },
      ],
    },
  };

  const result = normalizeBackendResponse(payload, { type: "TEXT", latency: 18 });
  const item = result.items[0];
  assert.equal(item.faissIndex, 277466);
  assert.equal(item.gid, 277466);
  assert.equal(item.globalFrameId, 3048);
  assert.equal(item.submissionFrameId, 3048);
  assert.equal(item.frameKey, "003048");
});
test("routes a text query to FastAPI's users endpoint", async () => {
  let call;
  await runBackendSearch(
    { searchType: "TEXT", query: "forklift", params: { topk: 3 } },
    null,
    {
      config: { baseUrl: "http://localhost:3000", mode: "live" },
      fetchImpl: async (url, init) => {
        call = { url, init };
        return { ok: true, json: async () => successPayload };
      },
    }
  );

  assert.equal(call.url, "http://localhost:3000/users/singletextsearch");
  assert.deepEqual(JSON.parse(call.init.body), { query: "forklift", topk: 3 });
});

test("routes an OCR Text query to the /ocrsearch endpoint", async () => {
  let call;
  await runBackendSearch(
    { searchType: "OCR", query: "north gate", params: { topk: 5 } },
    null,
    {
      config: { baseUrl: "http://localhost:3000", mode: "live" },
      fetchImpl: async (url, init) => {
        call = { url, init };
        return { ok: true, json: async () => successPayload };
      },
    },
  );

  assert.equal(call.url, "http://localhost:3000/users/ocrsearch");
  assert.deepEqual(JSON.parse(call.init.body), { query: "north gate", topk: 5 });
});

test("temporalRequestBody folds context into every event and caps topk at 100", () => {
  const body = temporalRequestBody(
    ["Bối cảnh: video nấu ăn về nấm", "E1: cắt nấm", "E2: bắc chảo lên bếp"].join("\n"),
    250,
  );
  assert.equal(body.topk, 100);
  assert.equal(body.context, "video nấu ăn về nấm");
  assert.equal(body.query.length, 2);
  assert.ok(body.query[0].query.includes("video nấu ăn về nấm"));
  assert.ok(body.query[0].query.includes("cắt nấm"));
});

test("routes a temporal query to /users/temporalsearch with event objects", async () => {
  let call;
  const seqPayload = {
    success: true,
    data: {
      total_items: 1,
      items: [
        {
          video_id: "L21_V001",
          timestamps: [1, 2],
          frames: [
            { event_index: 1, video_key: "L21_V001", frame_key: "000010", submission_frame_id: 10, timestamp: 1 },
            { event_index: 2, video_key: "L21_V001", frame_key: "000040", submission_frame_id: 40, timestamp: 2 },
          ],
        },
      ],
    },
  };
  const result = await runBackendSearch(
    { searchType: "TEMPORAL", query: "E1: a\nE2: b", params: { topk: 50 } },
    null,
    {
      config: { baseUrl: "http://localhost:3000", mode: "live" },
      fetchImpl: async (url, init) => {
        call = { url, init };
        return { ok: true, json: async () => seqPayload };
      },
    },
  );

  assert.equal(call.url, "http://localhost:3000/users/temporalsearch");
  const sent = JSON.parse(call.init.body);
  assert.deepEqual(sent.query, [{ query: "a" }, { query: "b" }]);
  assert.equal(sent.topk, 50);
  assert.equal(result.type, "TEMPORAL");
  assert.equal(result.sequences.length, 1);
  assert.deepEqual(result.sequences[0].frames.map((f) => f.submissionFrameId), [10, 40]);
  assert.deepEqual(result.items, []);
});

test("routes an image pivot as multipart data", async () => {
  let body;
  await runBackendSearch(
    { searchType: "IMAGE", params: { topk: 2, imageFile: null } },
    { faissIndex: 7 },
    {
      config: { baseUrl: "http://localhost:3000/users", mode: "live" },
      fetchImpl: async (url, init) => {
        assert.equal(url, "http://localhost:3000/users/imagesearch");
        body = init.body;
        return { ok: true, json: async () => successPayload };
      },
    }
  );

  assert.equal(body.get("faiss_index"), "7");
  assert.equal(body.get("topk"), "2");
});

test("routes a captured-frame pivot to the capture-similar endpoint without faiss_index", async () => {
  let call;
  await runBackendSearch(
    { searchType: "IMAGE", params: { topk: 30 } },
    {
      captured: true,
      videoKey: "L21_V001",
      submissionFrameId: 351,
      frameKey: "351",
      backend: { video_id: "L21_V001", frame_idx: 351 },
    },
    {
      config: { baseUrl: "http://localhost:3000/users", mode: "live" },
      fetchImpl: async (url, init) => {
        call = { url, init };
        return { ok: true, json: async () => successPayload };
      },
    }
  );

  assert.equal(call.url, "http://localhost:3000/users/videos/captures/L21_V001/351/similar");
  assert.equal(call.init.headers["Content-Type"], "application/json");
  assert.deepEqual(JSON.parse(call.init.body), { topk: 30 });
  // A captured frame_idx is per-video; it must never be sent as faiss_index.
  assert.ok(!/faiss_index/.test(String(call.init.body)));
});

test("captured pivot with frame_idx 0 still targets the capture-similar endpoint", async () => {
  let call;
  await runBackendSearch(
    { searchType: "IMAGE", params: { topk: 10 } },
    { captured: true, videoKey: "L21_V001", submissionFrameId: 0 },
    {
      config: { baseUrl: "http://localhost:3000", mode: "live" },
      fetchImpl: async (url, init) => {
        call = { url, init };
        return { ok: true, json: async () => successPayload };
      },
    }
  );
  assert.equal(call.url, "http://localhost:3000/users/videos/captures/L21_V001/0/similar");
});

test("a non-captured keyframe pivot still sends faiss_index to imagesearch", async () => {
  let url;
  let body;
  await runBackendSearch(
    { searchType: "IMAGE", params: { topk: 5 } },
    { faissIndex: 277466, submissionFrameId: 3048, globalFrameId: 3048 },
    {
      config: { baseUrl: "http://localhost:3000/users", mode: "live" },
      fetchImpl: async (u, init) => {
        url = u;
        body = init.body;
        return { ok: true, json: async () => successPayload };
      },
    }
  );

  assert.equal(url, "http://localhost:3000/users/imagesearch");
  assert.equal(body.get("faiss_index"), "277466");
  assert.equal(body.get("topk"), "5");
});

test("surfaces retrieval_backend provenance on a normalized card", () => {
  const payload = {
    success: true,
    data: {
      total_items: 1,
      items: [{ vector_id: 5, video_id: "L21_V001", frame_id: "kf_0002", retrieval_backend: "jina_clip_v2" }],
    },
  };
  const result = normalizeBackendResponse(payload, { type: "TEXT", latency: 1 });
  assert.equal(result.items[0].retrievalBackend, "jina_clip_v2");
});

test("a card with no retrieval_backend defaults to beit3 (old BEiT3 result shape)", () => {
  const payload = {
    success: true,
    data: { total_items: 1, items: [{ vector_id: 9, faiss_id: 9, video_id: "L21_V001", frame_id: "003048" }] },
  };
  const result = normalizeBackendResponse(payload, { type: "TEXT", latency: 1 });
  assert.equal(result.items[0].retrievalBackend, "beit3");
  // and the raw payload the card echoes has no such key
  assert.equal("retrieval_backend" in result.items[0].backend, false);
});

test("an image pivot by id sends its backend provenance alongside faiss_index", async () => {
  let body;
  await runBackendSearch(
    { searchType: "IMAGE", params: { topk: 4 } },
    { faissIndex: 991, retrievalBackend: "jina_clip_v2" },
    {
      config: { baseUrl: "http://localhost:3000/users", mode: "live" },
      fetchImpl: async (url, init) => {
        body = init.body;
        return { ok: true, json: async () => successPayload };
      },
    }
  );
  assert.equal(body.get("faiss_index"), "991");
  assert.equal(body.get("retrieval_backend"), "jina_clip_v2");
});

test("an id pivot from a BEiT3 card (no provenance field) still sends retrieval_backend=beit3", async () => {
  let body;
  await runBackendSearch(
    { searchType: "IMAGE", params: { topk: 5 } },
    { faissIndex: 277466, submissionFrameId: 3048 },
    {
      config: { baseUrl: "http://localhost:3000/users", mode: "live" },
      fetchImpl: async (url, init) => {
        body = init.body;
        return { ok: true, json: async () => successPayload };
      },
    }
  );
  assert.equal(body.get("faiss_index"), "277466");
  assert.equal(body.get("retrieval_backend"), "beit3");
});

test("an uploaded-image search carries no faiss_index and no provenance", async () => {
  let body;
  const blob = new Blob(["x"], { type: "image/png" });
  await runBackendSearch(
    { searchType: "IMAGE", params: { topk: 4, imageFile: blob } },
    null,
    {
      config: { baseUrl: "http://localhost:3000/users", mode: "live" },
      fetchImpl: async (url, init) => {
        body = init.body;
        return { ok: true, json: async () => successPayload };
      },
    }
  );
  assert.equal(body.get("faiss_index"), null);
  assert.equal(body.get("retrieval_backend"), null);
});

test("does not classify valid HTTP failures as transport failures", async () => {
  await assert.rejects(
    runBackendSearch(
      { searchType: "QA", query: "invalid", params: { topk: 1 } },
      null,
      { config: { baseUrl: "http://localhost:3000", mode: "auto" }, fetchImpl: async () => ({ ok: false, status: 422, json: async () => ({ message: "topk invalid" }) }) }
    ),
    (error) => error instanceof BackendSearchError && !isTransportError(error) && error.status === 422
  );
});

test("probe reports live FastAPI and demo-safe unavailable states", async () => {
  const online = await probeBackend({ config: { mode: "live", baseUrl: "http://localhost:3000/users" }, fetchImpl: async (url) => ({ ok: url === "http://localhost:3000/health", status: 200 }) });
  const offline = await probeBackend({ config: { mode: "auto", baseUrl: "http://localhost:3000" }, fetchImpl: async () => { throw new Error("offline"); } });
  assert.deepEqual(online, { backend: "online", demo: false, note: "FASTAPI" });
  assert.deepEqual(offline, { backend: "offline", demo: true, note: "FASTAPI UNAVAILABLE" });
});

test("routes copilot turns to FastAPI's conversational agent endpoint", async () => {
  let call;
  const result = await runAgentChat(
    { sessionId: "s1", message: "find the red bus", topk: 12 },
    {
      config: { baseUrl: "http://localhost:3000/users", mode: "live" },
      fetchImpl: async (url, init) => {
        call = { url, init };
        return { ok: true, status: 200, json: async () => ({ success: true, session_id: "s1", response: "ok", data: null }) };
      },
    }
  );

  assert.equal(call.url, "http://localhost:3000/chat/conversational_kis");
  assert.deepEqual(JSON.parse(call.init.body), { session_id: "s1", message: "find the red bus", topk: 12 });
  assert.deepEqual(result, { sessionId: "s1", response: "ok", data: null, mode: "AGENT LIVE", source: "live" });
});
test("normalizes OCR results into keyframe image URLs", () => {
  const payload = {
    success: true,
    data: {
      total_items: 1,
      items: [
        {
          faiss_id: 181,
          video_id: "L25_V041",
          frame_name: "181.jpg",
          split: "L25",
          global_frame_id: 181,
          timestamp: 682.18,
          ocr_text: "remember",
        },
      ],
    },
  };

  const result = normalizeBackendResponse(payload, { type: "OCR", latency: 66 }, "http://localhost:3000/users");
  assert.equal(result.items[0].image, "http://localhost:3000/keyframes/L25/L25_V041/181.jpg");
  assert.equal(result.items[0].ocrText, "remember");
});
test("deduplicates repeated backend keyframes before rendering", () => {
  const payload = {
    success: true,
    data: {
      total_items: 3,
      items: [
        { video_id: "L23_V024", frame_id: "007845", frame_name: "L23_V024_007845", final_score: 0.9 },
        { video_id: "L23_V024", frame_id: "007845", frame_name: "L23_V024_007845", final_score: 0.88 },
        { video_id: "L22_V001", frame_id: "021336", frame_name: "L22_V001_021336", final_score: 0.8 },
      ],
    },
  };

  const result = normalizeBackendResponse(payload, { type: "TEXT", latency: 12 });
  assert.equal(result.totalItems, 2);
  assert.deepEqual(result.items.map((item) => item.frameName), ["L23_V024_007845", "L22_V001_021336"]);
  assert.deepEqual(result.items.map((item) => item.rank), [1, 2]);
});

test("normalizes agent verification score and metadata", () => {
  const payload = {
    success: true,
    data: {
      total_items: 1,
      items: [
        {
          video_id: "V1",
          frame_id: "0010",
          frame_name: "V1_0010",
          score: 0.42,
          verification_score: 0.87,
          reason: "matched 3 checklist items",
          agent_verification: { method: "light_no_vlm", sources: ["direct"] },
        },
      ],
    },
  };

  const result = normalizeBackendResponse(payload, { type: "AGENT", latency: 22 });
  assert.equal(result.items[0].score, 0.87);
  assert.equal(result.items[0].reason, "matched 3 checklist items");
  assert.deepEqual(result.items[0].agentVerification, { method: "light_no_vlm", sources: ["direct"] });
});
