/**
 * ADAPTER BOUNDARY - stable interfaces used by feature code.
 *
 * Feature components should import from here (never from the mocks directly)
 * so a real provider/backend can be swapped in without touching UI code.
 * Signatures must stay stable:
 *   - translateText(text, dir)  -> string        (dir: "en-vi" | "vi-en")
 *   - askCopilot(text, frames)  -> Promise<{text, demo}>
 *   - runSearch(tab, pivot)     -> Promise<{items, totalItems, latency, type, mode}>
 */
export { translateText } from "../services/translateService.js";
export { askCopilot, askGroundedQa } from "../services/copilotService.js";

import { getSearchConfig, isTransportError, runBackendAgentSearch, runBackendSearch } from "../services/backendSearch.js";
import { mockSearch } from "../mocks/searchEngine.js";

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


export async function runAgentSearch(tab, pivot = null) {
  const config = getSearchConfig();
  const demoAgent = async (reason = "LOCAL MOCK AGENT") => {
    const fallback = await mockSearch({ ...tab, searchType: "TEXT" }, pivot);
    return {
      ...fallback,
      type: "AGENT",
      mode: reason,
      source: config.mode === "demo" ? "demo" : "fallback",
      plan: {
        original_query: tab?.query || "",
        expanded_queries: [{ kind: "visual", query: tab?.query || "", query_en: tab?.query || "" }],
        routing: { visual: 1, ocr: 0, asr: 0 },
      },
      queriesUsed: [{ kind: "visual", query: tab?.query || "", queryEn: tab?.query || "" }],
      routing: { visual: 1, ocr: 0, asr: 0 },
      response: "Agent Search is using local mock results because FastAPI is unavailable.",
    };
  };

  if (config.mode === "demo" || (config.mode === "auto" && !config.baseUrl)) {
    return demoAgent();
  }

  try {
    return await runBackendAgentSearch(tab, { config });
  } catch (error) {
    if (config.mode === "auto" && isTransportError(error)) {
      return demoAgent("FALLBACK DEMO - FASTAPI AGENT UNAVAILABLE");
    }
    throw error;
  }
}
