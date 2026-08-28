import { getSearchConfig } from "./backendSearch.js";

async function requestBackendTranslation(input, from_lang, to_lang) {
  const { baseUrl } = getSearchConfig();
  const cleanBase = (baseUrl || "").replace(/\/users$/i, "").replace(/\/+$/, "");
  const endpoint = cleanBase ? `${cleanBase}/users/translate` : "/users/translate";

  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: input, from_lang, to_lang }),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

/**
 * @param {string} text
 * @param {"vi-en" | "en-vi"} dir
 * @returns {Promise<{text: string, live: boolean, provider: string}>}
 */
export async function translateTextDetailed(text, dir = "vi-en") {
  const input = String(text || "").trim();
  if (!input) {
    return { text: "", live: false, provider: "none" };
  }

  const from_lang = dir === "vi-en" ? "vi" : "en";
  const to_lang = dir === "vi-en" ? "en" : "vi";

  try {
    const data = await requestBackendTranslation(input, from_lang, to_lang);
    const translated = String(data?.translated_text || "").trim();
    if (translated && data?.translated !== false && translated !== input) {
      return { text: translated, live: true, provider: data?.provider || "backend" };
    }
    console.warn("Backend translation returned identity text:", data);
  } catch (error) {
    console.warn("Backend translation API error, falling back to local:", error);
  }

  // Never produce a partial dictionary translation: mixed-language text is
  // worse than no translation because it corrupts visual-search embeddings.
  return { text: input, live: false, provider: "unavailable" };
}

/**
 * Backward-compatible string-only API.
 * @param {string} text
 * @param {"vi-en" | "en-vi"} dir
 * @returns {Promise<string>}
 */
export async function translateText(text, dir = "vi-en") {
  const result = await translateTextDetailed(text, dir);
  return result.text;
}
