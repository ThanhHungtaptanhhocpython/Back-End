import { getSearchConfig, normalizeBackendItem, normalizeBackendResponse } from "./backendSearch.js";
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


function wantsAgentSearch(text, frames = []) {
  if (Array.isArray(frames) && frames.length) return false;
  const prompt = normaliseIntentText(text);
  const searchTerms = ["tim", "find", "search", "canh", "khung hinh", "frame", "clip", "video", "vach dich", "xe dap", "nguoi", "xe", "bien", "duong"];
  const questionTerms = ["la gi", "tai sao", "vi sao", "nhu the nao", "explain", "what", "why", "how"];
  return searchTerms.some((term) => prompt.includes(term)) && !questionTerms.some((term) => prompt.includes(term));
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
  const agentSearch = !deep && wantsAgentSearch(prompt, frames);
  const deepPrompt = deep ? cleanDeepSearchPrompt(prompt) : prompt;

  try {
    const endpointName = deep ? "deep_keyframe_search" : agentSearch ? "agent_search" : "conversational_kis";
    const endpoint = `${baseUrl}/chat/${endpointName}`;
    const response = await fetchImpl(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: getSessionId(),
        message: deep || agentSearch ? deepPrompt : buildChatMessage(prompt, frames),
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
    const agentFrames = agentSearch ? normalizeChatFrames(payload, baseUrl) : [];
    const chatFrames = deep || agentSearch ? [] : normalizeChatFrames(payload, baseUrl);
    const resultFrames = deep ? deepFrames : agentSearch ? agentFrames : chatFrames;
    const queriesUsed = formatQueries(payload?.data?.queries_used || []);
    const totalCandidates = Number(payload?.data?.total_candidates || 0);
    const frameSummary = deepFrames.length ? `\n\nAdded ${deepFrames.length} keyframes to a Deep Search results tab.` : agentFrames.length ? `\n\nAdded ${agentFrames.length} keyframes to an Agent Search results tab.` : "";
    const candidateSummary = totalCandidates ? ` Candidates before dedup: ${totalCandidates}.` : "";

    if (responseText || resultFrames.length) {
      return {
        text: `${responseText || "Deep keyframe search completed."}${candidateSummary}${frameSummary}`,
        demo: false,
        frames: resultFrames,
        queriesUsed,
        searchQuery: deep || agentSearch ? deepPrompt : undefined,
        mode: payload?.data?.mode,
        routing: payload?.data?.routing || payload?.data?.plan?.routing || {},
        searchPlan: payload?.data?.plan || {},
        error: payload?.success === true ? undefined : responseText,
      };
    }

    throw new Error("Backend chat returned an empty response.");
  } catch (error) {
    console.warn("Chat backend transport error, falling back to mock:", error);
    return { text: mockChatReply(prompt, frames), demo: true, error: error instanceof Error ? error.message : String(error) };
  }
}

export async function askGroundedQa(text, { fetchImpl = globalThis.fetch, topk = 24 } = {}) {
  const prompt = String(text || "").trim();
  if (!prompt) return { text: "", frames: [], demo: false };

  const config = getSearchConfig();
  const baseUrl = cleanBaseUrl(config.baseUrl) || "http://127.0.0.1:3000";
  if (config.mode === "demo" || typeof fetchImpl !== "function") {
    return { text: mockChatReply(prompt, []), frames: [], demo: true };
  }

  try {
    const startedAt = Date.now();
    const response = await fetchImpl(`${baseUrl}/users/qnasearch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: prompt, topk }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload?.message || payload?.detail || `Grounded Q&A failed with HTTP ${response.status}.`);
    }

    const normalized = normalizeBackendResponse(
      payload,
      { type: "QA", latency: Date.now() - startedAt },
      baseUrl
    );
    const meta = normalized.meta || {};
    const evaluated = normalized.items.filter((frame) => frame.backend?.qa_evidence_id);
    const supporting = evaluated.filter((frame) => frame.backend?.qa_supporting);
    const sourceFrames = supporting.length ? supporting : evaluated;
    const confidence = Number(meta.confidence);
    const confidenceLine = Number.isFinite(confidence)
      ? `\n\nConfidence: ${Math.round(confidence * 100)}% (${meta.status || "unknown"})`
      : "";

    return {
      text: `${String(meta.answer || payload?.message || "No grounded answer returned.").trim()}${confidenceLine}`,
      frames: sourceFrames,
      allFrames: normalized.items,
      meta,
      mode: "grounded_qa",
      demo: false,
    };
  } catch (error) {
    console.warn("Grounded Q&A request failed:", error);
    return {
      text: "Grounded Q&A is unavailable, so no answer was generated.",
      frames: [],
      demo: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}


