import { getSearchConfig } from "./backendSearch.js";
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

  try {
    const endpoint = `${baseUrl}/chat/conversational_kis`;
    const response = await fetchImpl(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: getSessionId(),
        message: buildChatMessage(prompt, frames),
        topk: 100,
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

    if (responseText) {
      return {
        text: responseText,
        demo: false,
        error: payload?.success === true ? undefined : responseText,
      };
    }

    throw new Error("Backend chat returned an empty response.");
  } catch (error) {
    console.warn("Chat backend transport error, falling back to mock:", error);
    return { text: mockChatReply(prompt, frames), demo: true, error: error instanceof Error ? error.message : String(error) };
  }
}
