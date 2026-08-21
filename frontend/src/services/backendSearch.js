import { toTimecode } from "../shared/format.js";

const SEARCH_ENDPOINTS = Object.freeze({
  TEXT: "singletextsearch",
  QA: "qnasearch",
  IMAGE: "imagesearch",
  TEMPORAL: "temporalsearch",
  OCR: "ocrsearch",
  ASR: "asrsearch",
  "OCR+OD": "ocrandodsearch",
  MULTIMODAL: "multimodalsearch",
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

function imageSource(value) {
  if (!value || typeof value !== "string") return "";
  if (/^(data:|blob:|https?:\/\/|\/)/i.test(value)) return value;
  return `data:image/webp;base64,${value}`;
}

function temporalEvents(query) {
  const events = String(query || "")
    .split(/\n+/)
    .map((event) => event.trim())
    .filter(Boolean)
    .map((event) => ({ query: event }));
  return events.length ? events : [{ query: "" }];
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

function chatUrl(baseUrl) {
  return `${baseUrl.replace(/\/users$/i, "").replace(/\/+$/, "")}/chat/conversational_kis`;
}

function requestFor(tab, pivot) {
  const type = tab?.searchType || "TEXT";
  const params = tab?.params || {};
  const topk = positiveInteger(params.topk);
  const query = String(tab?.query || "").trim();
  const endpoint = endpointFor(type);

  if (type === "IMAGE") {
    const body = new FormData();
    const image = params.imageFile;
    const effectivePivot = pivot || tab?.pivotItem;
    const faissIndex = firstDefined(
      effectivePivot?.faissIndex,
      effectivePivot?.vector_id,
      effectivePivot?.backend?.vector_id,
      effectivePivot?.faiss_id_clip,
      effectivePivot?.faiss_id,
      effectivePivot?.faiss_idx,
      effectivePivot?.backend?.faiss_id,
      effectivePivot?.backend?.faiss_idx,
      effectivePivot?.gid
    );
    if (image instanceof Blob) body.append("image", image, image.name || "reference-image");
    if (faissIndex !== undefined && faissIndex !== null) body.append("faiss_index", String(faissIndex));
    body.append("topk", String(topk));
    body.append("clip", String(Boolean(params.clip)));
    body.append("clipv2", String(Boolean(params.clipv2)));
    return { endpoint, body };
  }

  if (type === "TEMPORAL") {
    return {
      endpoint,
      body: { query: temporalEvents(query), topk, cascaded: Boolean(params.cascaded) },
      headers: { "Content-Type": "application/json" },
    };
  }

  return {
    endpoint,
    body: { query, topk, clip: Boolean(params.clip), clipv2: Boolean(params.clipv2) },
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
  const faissIndex = firstDefined(raw.faiss_index, raw.faiss_id_clip, raw.faiss_id, raw.faiss_idx, raw.nearest_faiss_id, raw.vector_id);
  const submissionFrameId = firstDefined(raw.submission_frame_id, raw.frame_idx, raw.frame_key, raw.frame_id, raw.global_frame_id);
  const frameKey = firstDefined(submissionFrameId, raw.id, faissIndex, rank);
  const videoKey = String(firstDefined(raw.video_key, raw.video_id, raw.videoKey, "unknown-video"));
  const folderKey = String(firstDefined(raw.folder_key, raw.folderKey, raw.namespace, raw.split, "UNKNOWN"));
  const timestamp = finiteNumber(raw.timestamp);
  const fps = positiveInteger(raw.fps, 25);
  const scoreValue = firstDefined(raw.final_score, raw.normalized_score, raw.score, raw._score);
  const frameName = String(firstDefined(raw.frame_name, raw.frameName, `${videoKey}_${frameKey}`));

  const framePath = firstDefined(raw.frame_path, raw.framePath);
  let resolvedImage = firstDefined(raw.image, raw.thumbnail, raw.image_url);
  if (!resolvedImage && framePath) {
    if (baseUrl) {
      resolvedImage = `${baseUrl.replace(/\/+$/, "")}/keyframes/${framePath.replace(/^\/+/, "")}`;
    } else {
      resolvedImage = `/keyframes/${framePath.replace(/^\/+/, "")}`;
    }
  }

  // Keyframe ID must be uniquely tied to the actual video keyframe across all searches (NOT position-dependent rank)
  const uniqueId = String(
    firstDefined(
      videoKey !== "unknown-video" && submissionFrameId !== undefined ? `${videoKey}_${submissionFrameId}` : undefined,
      raw.frame_name,
      framePath,
      raw.global_frame_id,
      raw.vector_id !== undefined ? `vec-${raw.vector_id}` : undefined,
      faissIndex !== undefined ? `faiss-${faissIndex}` : undefined,
      raw.id,
      `frame-${videoKey}-${rank}`
    )
  );

  return {
    id: uniqueId,
    gid: finiteNumber(firstDefined(faissIndex, raw.id, rank), rank),
    globalFrameId: finiteNumber(firstDefined(submissionFrameId, raw.id, rank), rank),
    submissionFrameId: finiteNumber(firstDefined(submissionFrameId, raw.id, rank), rank),
    folderKey,
    videoKey,
    camera: String(firstDefined(raw.camera, raw.camera_id, "BACKEND")),
    frameKey: String(frameKey),
    frameName,
    timestamp,
    timecode: String(firstDefined(raw.timecode, toTimecode(timestamp, fps))),
    fps,
    width: finiteNumber(raw.width),
    height: finiteNumber(raw.height),
    image: imageSource(resolvedImage),
    link: String(firstDefined(raw.link, raw.video_url, "")),
    real: true,
    faissIndex: faissIndex === undefined ? undefined : finiteNumber(faissIndex),
    score: normaliseScore(scoreValue, rank, total),
    rank,
    answer: raw.answer,
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

  const totalItems = positiveInteger(payload.data.total_items, payload.data.items.length);
  return {
    items: payload.data.items.map((item, index) => normalizeBackendItem(item, index + 1, totalItems, baseUrl)),
    totalItems,
    latency,
    type,
    mode: "FASTAPI LIVE",
    source: "live",
  };
}

export function isTransportError(error) {
  return error instanceof BackendSearchError && error.kind === "transport";
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

  return normalizeBackendResponse(
    payload,
    {
      type: tab?.searchType || "TEXT",
      latency: Date.now() - startedAt,
    },
    config.baseUrl
  );
}

/** Execute one conversational KIS turn through the FastAPI agent router. */
export async function runAgentChat(
  { sessionId, message, topk = 100 },
  { config = getSearchConfig(), fetchImpl = globalThis.fetch } = {}
) {
  if (!config.baseUrl) {
    throw new BackendSearchError("VITE_SEARCH_API_BASE_URL is required for live agent chat.", { kind: "config" });
  }
  if (typeof fetchImpl !== "function") {
    throw new BackendSearchError("Fetch is unavailable in this environment.", { kind: "transport" });
  }

  let response;
  try {
    response = await fetchImpl(chatUrl(config.baseUrl), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        message,
        topk: positiveInteger(topk, 100),
      }),
    });
  } catch (cause) {
    throw new BackendSearchError("FastAPI agent service is unavailable.", { kind: "transport", cause });
  }

  let payload;
  try {
    payload = await response.json();
  } catch (cause) {
    throw new BackendSearchError(`Agent returned an unreadable HTTP ${response.status} response.`, {
      kind: "response",
      status: response.status,
      cause,
    });
  }

  if (!response.ok || payload?.success !== true) {
    throw new BackendSearchError(payload?.response || payload?.message || `Agent failed with HTTP ${response.status}.`, {
      kind: "response",
      status: response.status,
    });
  }

  return {
    sessionId: payload.session_id || sessionId,
    response: String(payload.response || ""),
    data: payload.data ?? null,
    mode: "AGENT LIVE",
    source: "live",
  };
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

