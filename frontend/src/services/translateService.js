import { getSearchConfig } from "./backendSearch.js";

/**
 * @typedef {"ok" | "backend_unreachable" | "provider_unavailable" | "invalid_input" | "empty"} TranslationStatus
 *
 * - ok                  a real (or same-language identity) translation
 * - backend_unreachable  the request never reached a working backend
 * - provider_unavailable the backend answered but no translator produced text
 * - invalid_input        the backend rejected the text as empty / invalid
 * - empty                nothing was sent (blank input) -- handled client-side
 */

function translateEndpoint() {
  const { baseUrl } = getSearchConfig();
  const cleanBase = (baseUrl || "").replace(/\/users$/i, "").replace(/\/+$/, "");
  return cleanBase ? `${cleanBase}/users/translate` : "/users/translate";
}

async function requestBackendTranslation(input, from_lang, to_lang, fetchImpl = globalThis.fetch) {
  const response = await fetchImpl(translateEndpoint(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: input, from_lang, to_lang }),
  });

  // The backend now returns structured failures (400 / 503) with a JSON body;
  // read it regardless of status instead of throwing on !ok.
  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  return { ok: Boolean(response.ok), status: Number(response.status) || 0, body: body || {} };
}

function sameText(a, b) {
  return String(a).trim().toLowerCase() === String(b).trim().toLowerCase();
}

/**
 * Translate text through the backend, returning a structured outcome. The
 * original text is preserved on every failure and is never reported as a live
 * translation.
 *
 * @param {string} text
 * @param {"vi-en" | "en-vi"} dir
 * @param {{ fetchImpl?: typeof fetch }} [options]
 * @returns {Promise<{ text: string, live: boolean, provider: string, status: TranslationStatus }>}
 */
export async function translateTextDetailed(text, dir = "vi-en", options = {}) {
  const input = String(text || "").trim();
  if (!input) {
    return { text: "", live: false, provider: "none", status: "empty" };
  }

  const from_lang = dir === "vi-en" ? "vi" : "en";
  const to_lang = dir === "vi-en" ? "en" : "vi";

  let res;
  try {
    res = await requestBackendTranslation(input, from_lang, to_lang, options.fetchImpl);
  } catch (error) {
    // A throw here means the request never got an HTTP response (offline, DNS,
    // CORS, connection reset): the backend itself was not reachable.
    console.warn("Translation backend unreachable:", error?.name || "network error");
    return { text: input, live: false, provider: "none", status: "backend_unreachable" };
  }

  const data = res.body || {};
  const backendCode = data.error_code || data.status;

  if (!res.ok) {
    if (backendCode === "invalid_input" || res.status === 400 || res.status === 422) {
      return { text: input, live: false, provider: "none", status: "invalid_input" };
    }
    if (backendCode === "provider_unavailable" || res.status === 503) {
      return { text: input, live: false, provider: "none", status: "provider_unavailable" };
    }
    console.warn("Translation backend error:", res.status);
    return { text: input, live: false, provider: "none", status: "backend_unreachable" };
  }

  // 200 OK -- honour an explicit structured failure first.
  if (backendCode === "provider_unavailable") {
    return { text: input, live: false, provider: "none", status: "provider_unavailable" };
  }
  if (backendCode === "invalid_input") {
    return { text: input, live: false, provider: "none", status: "invalid_input" };
  }

  const translated = String(data.translated_text || "").trim();

  // New-contract identity (from_lang === to_lang): a correct, non-live result.
  if (data.status === "ok" && data.translated === false) {
    return { text: translated || input, live: false, provider: data.provider || "identity", status: "ok" };
  }

  if (translated && data.translated !== false && !sameText(translated, input)) {
    return { text: translated, live: true, provider: data.provider || "backend", status: "ok" };
  }

  // 200 OK but the text came back unchanged: a legacy "identity" response or a
  // provider that silently echoed the input. Never present this as a success --
  // a mixed / untranslated query corrupts visual-search embeddings.
  console.warn("Translation returned the original text unchanged; treating as unavailable.");
  return { text: input, live: false, provider: "none", status: "provider_unavailable" };
}

/**
 * Backward-compatible string-only API. Always resolves to a string: the live
 * translation when available, otherwise the original text.
 * @param {string} text
 * @param {"vi-en" | "en-vi"} dir
 * @param {{ fetchImpl?: typeof fetch }} [options]
 * @returns {Promise<string>}
 */
export async function translateText(text, dir = "vi-en", options = {}) {
  const result = await translateTextDetailed(text, dir, options);
  return result.text;
}
