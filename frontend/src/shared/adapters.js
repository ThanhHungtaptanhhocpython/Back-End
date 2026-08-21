/**
 * ADAPTER BOUNDARY - stable interfaces used by feature code.
 *
 * Feature components should import from here (never from mocks directly)
 * so a real provider/backend can be swapped in without touching UI code.
 * Signatures must stay stable:
 *   - translateText(text, dir)  -> string        (dir: "en-vi" | "vi-en")
 *   - askCopilot(text, frames)  -> Promise<{text, demo, source, mode}>
 *   - runSearch(tab, pivot)     -> Promise<{items, totalItems, latency, type, mode}>
 */
export { translateText } from "../services/translateService.js";

import { mockChatReply } from "../mocks/copilot.js";
import { getSearchConfig, isTransportError, runAgentChat, runBackendSearch } from "../services/backendSearch.js";
import { mockSearch } from "../mocks/searchEngine.js";

function frameValue(frame, ...keys) {
  for (const key of keys) {
    const value = frame?.[key];
    if (value !== undefined && value !== null && value !== "") return value;
  }
  return undefined;
}

function frameContext(frames = []) {
  return frames
    .filter(Boolean)
    .slice(0, 5)
    .map((frame, index) => {
      const video = frameValue(frame, "videoKey", "video_id", "video_key") || "unknown-video";
      const frameId = frameValue(frame, "submissionFrameId", "frameKey", "globalFrameId", "frame_id") || "unknown-frame";
      const time = frameValue(frame, "timecode", "timestamp") || "unknown-time";
      const ocr = frameValue(frame, "ocrText", "ocr_text");
      const answer = frameValue(frame, "answer");
      return [
        `Frame ${index + 1}: video=${video}, frame=${frameId}, time=${time}, rank=${frame.rank ?? "unknown"}`,
        ocr ? `OCR: ${ocr}` : "",
        answer ? `Known answer: ${answer}` : "",
      ]
        .filter(Boolean)
        .join("\n");
    })
    .join("\n");
}

function agentMessage(text, frames) {
  const context = frameContext(frames);
  if (!context) return text;
  return `${text}\n\nContext from selected frame(s):\n${context}`;
}

function demoCopilot(text, frames, mode = "DEMO") {
  return {
    text: mockChatReply(text, frames),
    demo: true,
    source: "demo",
    mode,
    data: null,
  };
}

export async function askCopilot(text, frames = [], options = {}) {
  const config = getSearchConfig();
  if (config.mode === "demo" || (config.mode === "auto" && !config.baseUrl)) {
    return demoCopilot(text, frames);
  }

  try {
    const result = await runAgentChat(
      {
        sessionId: options.sessionId,
        message: agentMessage(text, frames),
        topk: options.topk,
      },
      { config }
    );
    return {
      text: result.response,
      demo: false,
      source: result.source,
      mode: result.mode,
      data: result.data,
    };
  } catch (error) {
    if (config.mode === "auto" && isTransportError(error)) {
      return demoCopilot(text, frames, "FALLBACK DEMO");
    }
    throw error;
  }
}

/**
 * Stable workstation search boundary.
 *
 * `demo` always uses local results. `auto` attempts FastAPI when a base URL is
 * configured and falls back only when the request cannot reach that service.
 * `live` surfaces every backend/configuration error to the caller.
 */
export async function runSearch(tab, pivot) {
  const config = getSearchConfig();
  if (config.mode === "demo" || (config.mode === "auto" && !config.baseUrl)) {
    return mockSearch(tab, pivot);
  }

  try {
    return await runBackendSearch(tab, pivot, { config });
  } catch (error) {
    if (config.mode === "auto" && isTransportError(error)) {
      const fallback = await mockSearch(tab, pivot);
      return {
        ...fallback,
        mode: "FALLBACK DEMO - FASTAPI UNAVAILABLE",
        source: "fallback",
        fallbackReason: error.message,
      };
    }
    throw error;
  }
}

export { probeBackend, fetchVideoTimeline } from "../services/backendSearch.js";