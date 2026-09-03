import { toTimecode } from "../shared/format.js";
import { buildTemporalEventQueries, MAX_TEMPORAL_TOPK, parseTemporalQuery } from "../shared/temporalQuery.js";
import { normalizeTemporalResponse } from "../shared/temporalNormalize.js";

const SEARCH_ENDPOINTS = Object.freeze({
  TEXT: "singletextsearch",
  QA: "qnasearch",
  IMAGE: "imagesearch",
  TEMPORAL: "temporalsearch",
  OCR: "ocrsearch",
  ASR: "asrsearch",
  // Compatibility alias: the "OCR Text" type is "OCR" and hits /ocrsearch. The
  // backend still exposes /ocrandodsearch (same OCR service) for old callers.
  "OCR+OD": "ocrandodsearch",
  MULTIMODAL: "multimodalsearch",
  AGENT: "agentsearch",
});

const SEARCH_MODES = new Set(["demo", "auto", "live"]);

export class BackendSearchError extends Error {
  constructor(message, { kind = "response", status, cause } = {}) {
    super(message, { cause });
    this.name = "BackendSearchError";
    this.kind = kind;
    this.status = status;
  }
}

function viteEnv() {
  return import.meta.env || {};
}

function positiveInteger(value, fallback = 24) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function finiteNumber(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function firstDefined(...values) {
  return values.find((value) => value !== undefined && value !== null && value !== "");
}

function normaliseScore(value, rank, total) {
  const score = Number(value);
  if (Number.isFinite(score)) return Math.max(0, Math.min(1, score));
  return total > 0 ? Math.max(0, 1 - (rank - 1) / total) : 0;
}
function dedupeItemsById(items) {
  const seen = new Set();
  const deduped = [];
  for (const item of Array.isArray(items) ? items : []) {
    const id = String(item?.id || "");
    const key = id || `${item?.videoKey || ""}:${item?.frameName || item?.frameKey || ""}:${item?.image || ""}`;
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(item);
  }
  return deduped;
}

function imageSource(value) {
  if (!value || typeof value !== "string") return "";
  if (/^(data:|blob:|https?:\/\/|\/)/i.test(value)) return value;
  return `data:image/webp;base64,${value}`;
}

function temporalTopk(value) {
  const parsed = Number.parseInt(value, 10);
  const safe = Number.isFinite(parsed) && parsed > 0 ? parsed : 100;
  return Math.min(safe, MAX_TEMPORAL_TOPK);
}

/**
 * Build the temporal request body from raw query text via the shared parser.
 * `context` is folded into every event query (it is not an event of its own),
 * and also passed alongside for a context-aware backend.
 */
export function temporalRequestBody(query, topk) {
  const parsed = parseTemporalQuery(query);
  const folded = buildTemporalEventQueries(parsed);
  const events = (folded.length ? folded : parsed.events).map((text) => ({ query: text }));
  return {
    query: events.length ? events : [{ query: String(query || "").trim() }],
    topk: temporalTopk(topk),
    context: parsed.context || "",
  };
}

function endpointFor(searchType) {
  const endpoint = SEARCH_ENDPOINTS[searchType || "TEXT"];
  if (!endpoint) {
    throw new BackendSearchError(`Unsupported search type: ${searchType || "unknown"}.`, { kind: "request" });
  }
  return endpoint;
}

function apiUrl(baseUrl, endpoint) {
  const trimmed = baseUrl.replace(/\/+$/, "");
  const usersBase = /\/users$/i.test(trimmed) ? trimmed : `${trimmed}/users`;
  return `${usersBase}/${endpoint}`;
}

function healthUrl(baseUrl) {
  return `${baseUrl.replace(/\/users$/i, "").replace(/\/+$/, "")}/health`;
}

function requestFor(tab, pivot) {
  const type = tab?.searchType || "TEXT";
  const params = tab?.params || {};
  const topk = positiveInteger(params.topk);
  const query = String(tab?.query || "").trim();
  const endpoint = endpointFor(type);

  if (type !== "IMAGE" && !query) {
    throw new BackendSearchError("Search query is empty.", { kind: "request" });
  }
  if (type === "IMAGE") {
    const effectivePivot = pivot || tab?.pivotItem;

    // A captured frame has no global FAISS vector id: its frame_idx is a
    // per-video index. Re-encode its exact extracted still with BEiT3 through
    // the capture-specific endpoint instead of mis-sending frame_idx as
    // faiss_index (which would search from an unrelated corpus vector).
    if (effectivePivot?.captured) {
      const videoId = firstDefined(
        effectivePivot.videoKey,
        effectivePivot.video_id,
        effectivePivot.backend?.video_id
      );
      const frameIdx = Number(
        firstDefined(
          effectivePivot.submissionFrameId,
          effectivePivot.backend?.frame_idx,
          effectivePivot.globalFrameId,
          effectivePivot.frameKey
        )
      );
      if (videoId === undefined || !Number.isFinite(frameIdx)) {
        throw new BackendSearchError("This captured frame has no video id / frame index to search from.", { kind: "request" });
      }
      return {
        endpoint: `videos/captures/${encodeURIComponent(videoId)}/${encodeURIComponent(frameIdx)}/similar`,
        body: { topk },
        headers: { "Content-Type": "application/json" },
      };
    }

    const body = new FormData();
    const image = params.imageFile;
    const faissIndex = firstDefined(
      effectivePivot?.faissIndex,
      effectivePivot?.vector_id,
      effectivePivot?.globalFrameId,
      effectivePivot?.gid,
      effectivePivot?.faiss_id,
      effectivePivot?.faiss_idx
    );
    if (image instanceof Blob) body.append("image", image, image.name || "reference-image");
    if (faissIndex !== undefined && faissIndex !== null) {
      body.append("faiss_index", String(faissIndex));
      // Provenance is REQUIRED for an id pivot so the backend can reject a
      // stale id after a backend switch (BEiT3 <-> Jina have independent
      // vector-id spaces). A card from the default BEiT3 backend carries no
      // `retrieval_backend`, so an absent value means `beit3`. (An uploaded
      // image carries no id and never reaches this branch.)
      const provenance =
        firstDefined(
          effectivePivot?.retrievalBackend,
          effectivePivot?.retrieval_backend,
          effectivePivot?.backend?.retrieval_backend
        ) || "beit3";
      body.append("retrieval_backend", String(provenance));
    }
    body.append("topk", String(topk));
    return { endpoint, body };
  }

  if (type === "TEMPORAL") {
    return {
      endpoint,
      body: { ...temporalRequestBody(query, params.topk), cascaded: Boolean(params.cascaded) },
      headers: { "Content-Type": "application/json" },
    };
  }

  return {
    endpoint,
    body: { query, topk },
    headers: { "Content-Type": "application/json" },
  };
}

/** Read and validate Vite search settings without making a request. */
export function getSearchConfig(env = viteEnv()) {
  const configuredMode = String(env.VITE_SEARCH_MODE || "auto").trim().toLowerCase();
  const mode = SEARCH_MODES.has(configuredMode) ? configuredMode : "auto";
  const baseUrl = String(env.VITE_SEARCH_API_BASE_URL || "").trim().replace(/\/+$/, "");
  return { mode, baseUrl };
}

/** Convert a FastAPI result record into the card shape consumed by ResultCard. */
export function normalizeBackendItem(item, rank, total, baseUrl = "") {
  const raw = item && typeof item === "object" ? item : {};
  const faissIndex = firstDefined(raw.faiss_index, raw.faiss_id, raw.faiss_idx, raw.nearest_faiss_id, raw.vector_id);
  const frameKey = firstDefined(raw.frame_key, raw.frame_id, raw.n, raw.global_frame_id, raw.id, faissIndex, rank);
  const videoKey = String(firstDefined(raw.video_key, raw.video_id, raw.videoKey, "unknown-video"));
  const folderKey = String(firstDefined(raw.folder_key, raw.folderKey, raw.namespace, raw.split, "UNKNOWN"));
  const timestamp = finiteNumber(raw.timestamp);
  const fps = positiveInteger(raw.fps, 25);
  const scoreValue = firstDefined(raw.verification_score, raw.final_score, raw.normalized_score, raw.score, raw._score);
  const frameName = String(firstDefined(raw.frame_name, raw.frameName, `${videoKey}_${frameKey}`));
  const submissionFrameId = finiteNumber(firstDefined(raw.submission_frame_id, raw.submissionFrameId, raw.frame_id, raw.frame_key, raw.n, frameKey), rank);

  let framePath = firstDefined(raw.frame_path, raw.framePath);
  let resolvedImage = firstDefined(raw.image, raw.thumbnail, raw.image_url);
  if (!resolvedImage && !framePath && videoKey && videoKey !== "unknown-video") {
    const split = String(firstDefined(raw.split, raw.folder_key, raw.folderKey, raw.namespace, videoKey.split("_")[0], "UNKNOWN"));
    let frameFile = String(firstDefined(raw.frame_id, raw.frame_idx, raw.keyframe_number, raw.frame_name, frameKey)).trim();
    const prefix = `${videoKey}_`;
    if (frameFile.startsWith(prefix)) frameFile = frameFile.slice(prefix.length);
    if (!/\.(webp|jpe?g|png)$/i.test(frameFile)) frameFile = `${frameFile}.webp`;
    framePath = `${split}/${videoKey}/${frameFile}`;
  }
  if (!resolvedImage && framePath) {
    if (baseUrl) {
      const imageBaseUrl = baseUrl.replace(/\/users$/i, "").replace(/\/+$/, "");
      resolvedImage = `${imageBaseUrl}/keyframes/${framePath.replace(/^\/+/, "")}`;
    } else {
      resolvedImage = `/keyframes/${framePath.replace(/^\/+/, "")}`;
    }
  }

  // Keyframe ID must be uniquely tied to the actual video keyframe across all searches (NOT position-dependent rank)
  const uniqueId = String(
    firstDefined(
      raw.global_frame_id,
      raw.frame_name,
      framePath,
      videoKey !== "unknown-video" && frameKey !== undefined ? `${videoKey}_${frameKey}` : undefined,
      raw.vector_id !== undefined ? `vec-${raw.vector_id}` : undefined,
      faissIndex !== undefined ? `faiss-${faissIndex}` : undefined,
      raw.id,
      `frame-${videoKey}-${rank}`
    )
  );

  return {
    id: uniqueId,
    gid: finiteNumber(firstDefined(raw.global_frame_id, raw.id, rank), rank),
    globalFrameId: submissionFrameId,
    submissionFrameId,
    folderKey,
    videoKey,
    camera: String(firstDefined(raw.camera, raw.camera_id, "BACKEND")),
    frameKey: String(frameKey),
    frameName,
    timestamp,
    timestampSource: raw.timestamp_source,
    timestampMatchedFrameIdx: raw.timestamp_matched_frame_idx,
    timestampFrameIdxDelta: raw.timestamp_frame_idx_delta,
    sourceFrameIdx: raw.source_frame_idx,
    keyframeNumber: raw.keyframe_number,
    timecode: String(firstDefined(raw.timecode, toTimecode(timestamp, fps))),
    fps,
    width: finiteNumber(raw.width),
    height: finiteNumber(raw.height),
    image: imageSource(resolvedImage),
    link: String(firstDefined(raw.link, raw.youtube_url, raw.youtubeUrl, raw.video_url, raw.videoUrl, raw.url) ?? ""),
    real: true,
    faissIndex: faissIndex === undefined ? undefined : finiteNumber(faissIndex),
    // Which retrieval backend produced this card, sent back with an
    // image-pivot-by-id request so the API can reject a stale cross-backend
    // id. BEiT3 (the default) result rows carry no `retrieval_backend`, so an
    // absent value is interpreted as `beit3`.
    retrievalBackend: firstDefined(raw.retrieval_backend, raw.retrievalBackend) || "beit3",
    score: normaliseScore(scoreValue, rank, total),
    rank,
    answer: raw.answer,
    reason: firstDefined(raw.reason, raw.agent_reason),
    verificationScore: raw.verification_score,
    agentVerification: raw.agent_verification,
    agentMatchedChecks: raw.agent_matched_checks,
    agentMissingChecks: raw.agent_missing_checks,
    ocrText: firstDefined(raw.ocr_text, raw.ocrText),
    backend: raw,
    ranking: {
      score: raw.score,
      normalizedScore: raw.normalized_score,
      finalScore: raw.final_score,
      scoreBreakdown: raw.score_breakdown,
      faissIndex,
    },
  };
}

/** Normalize FastAPI's BaseResponse envelope into the stable workstation result contract. */
export function normalizeBackendResponse(payload, { type, latency }, baseUrl = "") {
  if (!payload || typeof payload !== "object") {
    throw new BackendSearchError("Backend returned an invalid response.");
  }
  if (payload.success !== true) {
    throw new BackendSearchError(payload.message || "Backend search was not successful.");
  }
  if (!payload.data || !Array.isArray(payload.data.items)) {
    throw new BackendSearchError("Backend response did not include a result list.");
  }

  const rawTotalItems = positiveInteger(payload.data.total_items, payload.data.items.length);
  const items = dedupeItemsById(
    payload.data.items.map((item, index) => normalizeBackendItem(item, index + 1, rawTotalItems, baseUrl))
  ).map((item, index) => ({ ...item, rank: index + 1 }));
  return {
    items,
    totalItems: items.length,
    meta: payload.data.meta || null,
    latency,
    type,
    mode: "FASTAPI LIVE",
    source: "live",
  };
}

/**
 * A live Q&A request can retrieve vector matches while still having no local
 * image files for the VLM to inspect. In auto mode that state should use the
 * local demo dataset so the complete answer UI remains testable while
 * keyframes are being downloaded.
 */
export function shouldUseQaDemoFallback(tab, result) {
  if (tab?.searchType !== "QA" || result?.source !== "live") return false;
  const evaluatedFrames = Number(result?.meta?.evaluated_frames);
  return Number.isFinite(evaluatedFrames) && evaluatedFrames <= 0;
}

function cleanChatBaseUrl(baseUrl) {
  return String(baseUrl || "").trim().replace(/\/users$/i, "").replace(/\/+$/, "");
}

function normalizePlanQueries(queries = []) {
  return Array.isArray(queries) ? queries.map((query) => ({
    kind: String(query?.kind || "query"),
    query: String(query?.query || "").trim(),
    queryEn: String(query?.query_en || query?.queryEn || query?.query || "").trim(),
  })).filter((query) => query.query || query.queryEn) : [];
}

export function normalizeAgentSearchResponse(payload, { latency, type = "AGENT" }, baseUrl = "") {
  const base = normalizeBackendResponse(payload, { type, latency }, baseUrl);
  const plan = payload?.plan || payload?.data?.plan || {};
  return {
    ...base,
    plan,
    response: String(payload?.response || payload?.message || "").trim(),
    queriesUsed: normalizePlanQueries(plan.expanded_queries || payload?.data?.queries_used || []),
    routing: plan.routing || payload?.data?.routing || {},
    mode: "FASTAPI AGENT",
  };
}

export function isTransportError(error) {
  return error instanceof BackendSearchError && error.kind === "transport";
}

export async function runBackendAgentSearch(tab, { config = getSearchConfig(), fetchImpl = globalThis.fetch } = {}) {
  if (!config.baseUrl) {
    throw new BackendSearchError("VITE_SEARCH_API_BASE_URL is required for Agent Search.", { kind: "config" });
  }
  if (typeof fetchImpl !== "function") {
    throw new BackendSearchError("Fetch is unavailable in this environment.", { kind: "transport" });
  }

  const params = tab?.params || {};
  const query = String(tab?.query || "").trim();
  if (!query) {
    throw new BackendSearchError("Agent Search query is empty.", { kind: "request" });
  }

  const startedAt = Date.now();
  let response;
  try {
    response = await fetchImpl(apiUrl(config.baseUrl, "agentsearch"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, topk: positiveInteger(params.topk, 100) }),
    });
  } catch (cause) {
    throw new BackendSearchError("FastAPI Agent Search service is unavailable.", { kind: "transport", cause });
  }

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new BackendSearchError(payload?.message || payload?.detail || `Backend Agent Search failed with HTTP ${response.status}.`, {
      kind: "response",
      status: response.status,
    });
  }

  return normalizeAgentSearchResponse(payload, { latency: Date.now() - startedAt }, config.baseUrl);
}

export async function runAgentChat({ sessionId, message, topk = 100, endpoint = "conversational_kis" }, { config = getSearchConfig(), fetchImpl = globalThis.fetch } = {}) {
  if (!config.baseUrl) {
    throw new BackendSearchError("VITE_SEARCH_API_BASE_URL is required for agent chat.", { kind: "config" });
  }
  if (typeof fetchImpl !== "function") {
    throw new BackendSearchError("Fetch is unavailable in this environment.", { kind: "transport" });
  }

  const baseUrl = cleanChatBaseUrl(config.baseUrl);
  const response = await fetchImpl(`${baseUrl}/chat/${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message, topk }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new BackendSearchError(payload?.message || payload?.detail || `Backend agent chat failed with HTTP ${response.status}.`, {
      kind: "response",
      status: response.status,
    });
  }
  return {
    sessionId: payload?.session_id || sessionId,
    response: String(payload?.response || ""),
    data: payload?.data || null,
    mode: endpoint === "agent_search" ? "AGENT SEARCH LIVE" : "AGENT LIVE",
    source: "live",
  };
}

/** Execute one FastAPI search request. Callers decide whether a transport error may fall back. */
export async function runBackendSearch(tab, pivot, { config = getSearchConfig(), fetchImpl = globalThis.fetch } = {}) {
  if (!config.baseUrl) {
    throw new BackendSearchError("VITE_SEARCH_API_BASE_URL is required for a live search.", { kind: "config" });
  }
  if (typeof fetchImpl !== "function") {
    throw new BackendSearchError("Fetch is unavailable in this environment.", { kind: "transport" });
  }

  const request = requestFor(tab, pivot);
  const startedAt = Date.now();
  let response;
  try {
    response = await fetchImpl(apiUrl(config.baseUrl, request.endpoint), {
      method: "POST",
      headers: request.headers,
      body: request.headers ? JSON.stringify(request.body) : request.body,
    });
  } catch (cause) {
    throw new BackendSearchError("FastAPI search service is unavailable.", { kind: "transport", cause });
  }

  let payload;
  try {
    payload = await response.json();
  } catch (cause) {
    throw new BackendSearchError(`Backend returned an unreadable HTTP ${response.status} response.`, {
      kind: "response",
      status: response.status,
      cause,
    });
  }
  if (!response.ok) {
    throw new BackendSearchError(payload?.message || `Backend search failed with HTTP ${response.status}.`, {
      kind: "response",
      status: response.status,
    });
  }

  const latency = Date.now() - startedAt;
  if ((tab?.searchType || "TEXT") === "TEMPORAL") {
    const normalized = normalizeTemporalResponse(payload, { latency });
    // Keep the flat-result contract populated so generic callers stay safe.
    return { ...normalized, items: [], mode: "FASTAPI LIVE", source: "live" };
  }

  return normalizeBackendResponse(
    payload,
    {
      type: tab?.searchType || "TEXT",
      latency,
    },
    config.baseUrl
  );
}

/** Probe FastAPI without sending a search request, keeping demo mode explicit. */
export async function probeBackend({ config = getSearchConfig(), fetchImpl = globalThis.fetch } = {}) {
  if (config.mode === "demo") {
    return { backend: "offline", demo: true, note: "LOCAL MOCK" };
  }
  if (!config.baseUrl) {
    return { backend: "offline", demo: true, note: "NO API URL" };
  }
  if (typeof fetchImpl !== "function") {
    return { backend: "offline", demo: true, note: "FETCH UNAVAILABLE" };
  }

  try {
    const response = await fetchImpl(healthUrl(config.baseUrl));
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return { backend: "online", demo: false, note: "FASTAPI" };
  } catch {
    return { backend: "offline", demo: true, note: "FASTAPI UNAVAILABLE" };
  }
}

/** Fetch sequential keyframes for a specific video to power the Review timeline strip. */
export async function fetchVideoTimeline(videoId, aroundFrameId = null, limit = 60, { config = getSearchConfig(), fetchImpl = globalThis.fetch } = {}) {
  if (!videoId || videoId === "unknown-video") return [];
  const queryParams = new URLSearchParams();
  if (aroundFrameId) queryParams.set("around", String(aroundFrameId));
  if (limit) queryParams.set("limit", String(limit));

  const url = `${config.baseUrl ? config.baseUrl.replace(/\/+$/, "") : ""}/users/video_keyframes/${encodeURIComponent(videoId)}?${queryParams.toString()}`;
  try {
    const response = await fetchImpl(url);
    if (!response.ok) return [];
    const payload = await response.json();
    if (!payload?.data?.items) return [];
    const baseUrl = config.baseUrl || "";
    return payload.data.items.map((item, idx) => normalizeBackendItem(item, idx + 1, payload.data.items.length, baseUrl));
  } catch (err) {
    console.warn("Failed to fetch video timeline:", err);
    return [];
  }
}

export { SEARCH_ENDPOINTS };
