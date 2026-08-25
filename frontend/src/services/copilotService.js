import { getSearchConfig, normalizeBackendItem } from "./backendSearch.js";
import { mockChatReply } from "../mocks/copilot.js";

function cleanBaseUrl(baseUrl) {
  return String(baseUrl || "").trim().replace(/\/users$/i, "").replace(/\/+$/, "");
}

function buildChatMessage(text, frames = []) {
  const prompt = String(text || "").trim();
  if (!prompt) return "";
  const frame = Array.isArray(frames) && frames.length ? frames[0] : null;
  if (!frame) return prompt;

  const context = [
    "Ground the answer on this frame when relevant.",
    `video_id: ${frame.videoKey || frame.backend?.video_id || "unknown-video"}`,
    `frame_id: ${frame.frameKey || frame.backend?.frame_id || frame.id || "unknown-frame"}`,
    `timecode: ${frame.timecode || "unknown-timecode"}`,
    frame.ocrText ? `ocr_text: ${frame.ocrText}` : null,
    Array.isArray(frame.odClasses) && frame.odClasses.length ? `detected_objects: ${frame.odClasses.join(", ")}` : null,
    `user_question: ${prompt}`,
  ].filter(Boolean);

  return context.join("\n");
}

function getSessionStorage() {
  return globalThis.sessionStorage || globalThis.localStorage || null;
}

function getSessionId() {
  const storage = getSessionStorage();
  const existing = storage?.getItem("aic_chat_session_id");
  if (existing) return existing;
  const created = `ws-${globalThis.crypto?.randomUUID?.() || Date.now()}`;
  try {
    storage?.setItem("aic_chat_session_id", created);
  } catch {
    // ignore storage failures
  }
  return created;
}

function normaliseIntentText(text) {
  return String(text || "")
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/\u0111/g, "d");
}

function wantsDeepSearch(text) {
  const prompt = normaliseIntentText(text);
  return [
    "tim sau",
    "tim ky",
    "dao sau",
    "deep search",
    "khong tim duoc",
    "chua tim duoc",
    "chua tim thay",
    "khong tim thay",
    "tu thu nhieu huong",
    "retrieval agent",
  ].some((marker) => prompt.includes(marker));
}

function stripLeadingSearchIntent(text) {
  return String(text || "")
    .replace(/^\s*(t\u00f4i|toi)?\s*(ch\u01b0a|chua|kh\u00f4ng|khong)?\s*(t\u00ecm|tim)\s*(\u0111\u01b0\u1ee3c|duoc)?\s*/i, "")
    .replace(/^\s*(c\u1ea3nh|canh|khung h\u00ecnh|khung hinh|frame|clip|video)\s*(n\u00e0y|nay)?\s*:?\s*/i, "")
    .trim();
}

function cleanDeepSearchPrompt(text) {
  let prompt = String(text || "").trim();
  if (!prompt) return "";

  const colonIndex = prompt.indexOf(":");
  if (colonIndex >= 0 && wantsDeepSearch(prompt.slice(0, colonIndex))) {
    prompt = prompt.slice(colonIndex + 1).trim();
  }

  prompt = prompt
    .replace(/\s*(h\u00e3y|hay)\s*(t\u00ecm|tim)\s*(s\u00e2u|sau|k\u1ef9|ky).*$/i, "")
    .replace(/\s*(t\u1ef1|tu)\s*(th\u1eed|thu)\s*nhi\u1ec1u\s*h\u01b0\u1edbng.*$/i, "")
    .trim();

  prompt = stripLeadingSearchIntent(prompt);
  return prompt || String(text || "").trim();
}

function formatQueries(queries = []) {
  return queries
    .map((query) => {
      if (typeof query === "string") return { kind: "query", query, queryEn: query };
      const raw = String(query?.query || "").trim();
      const english = String(query?.query_en || query?.queryEn || raw).trim();
      if (!raw && !english) return null;
      return {
        kind: String(query?.kind || "query"),
        query: raw || english,
        queryEn: english || raw,
      };
    })
    .filter(Boolean);
}

function normalizeChatFrames(payload, baseUrl) {
  const rawFrames = Array.isArray(payload?.data?.frames) ? payload.data.frames : [];
  return rawFrames.map((item, index) => normalizeBackendItem(item, index + 1, rawFrames.length, baseUrl));
}
function normalizeDeepFrames(payload, baseUrl) {
  const rawFrames = Array.isArray(payload?.data?.frames) ? payload.data.frames : [];
  return rawFrames.map((item, index) => normalizeBackendItem(item, index + 1, rawFrames.length, baseUrl));
}

export async function askCopilot(text, frames = [], { fetchImpl = globalThis.fetch } = {}) {
  const prompt = String(text || "").trim();
  if (!prompt) {
    return { text: "", demo: false };
  }

  const config = getSearchConfig();
  const baseUrl = cleanBaseUrl(config.baseUrl) || "http://127.0.0.1:3000";
  if (config.mode === "demo" || typeof fetchImpl !== "function") {
    return { text: mockChatReply(prompt, frames), demo: true };
  }

  const deep = wantsDeepSearch(prompt);
  const deepPrompt = deep ? cleanDeepSearchPrompt(prompt) : prompt;

  try {
    const endpoint = `${baseUrl}/chat/${deep ? "deep_keyframe_search" : "conversational_kis"}`;
    const response = await fetchImpl(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: getSessionId(),
        message: deep ? deepPrompt : buildChatMessage(prompt, frames),
        topk: deep ? 24 : 100,
        per_query: 36,
      }),
    });

    const payload = await response.json().catch(() => ({}));
    const responseText = String(payload?.response || payload?.detail || payload?.error || payload?.message || "").trim();

    if (!response.ok) {
      return {
        text: responseText || `Backend chat failed with HTTP ${response.status}.`,
        demo: false,
        error: responseText || `HTTP ${response.status}`,
      };
    }

    const deepFrames = deep ? normalizeDeepFrames(payload, baseUrl) : [];
    const chatFrames = deep ? [] : normalizeChatFrames(payload, baseUrl);
    const resultFrames = deep ? deepFrames : chatFrames;
    const queriesUsed = formatQueries(payload?.data?.queries_used || []);
    const totalCandidates = Number(payload?.data?.total_candidates || 0);
    const frameSummary = deepFrames.length ? `\n\nAdded ${deepFrames.length} keyframes to a Deep Search results tab.` : "";
    const candidateSummary = totalCandidates ? ` Candidates before dedup: ${totalCandidates}.` : "";

    if (responseText || resultFrames.length) {
      return {
        text: `${responseText || "Deep keyframe search completed."}${candidateSummary}${frameSummary}`,
        demo: false,
        frames: resultFrames,
        queriesUsed,
        searchQuery: deep ? deepPrompt : undefined,
        mode: payload?.data?.mode,
        error: payload?.success === true ? undefined : responseText,
      };
    }

    throw new Error("Backend chat returned an empty response.");
  } catch (error) {
    console.warn("Chat backend transport error, falling back to mock:", error);
    return { text: mockChatReply(prompt, frames), demo: true, error: error instanceof Error ? error.message : String(error) };
  }
}