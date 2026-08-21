import { getSearchConfig } from "./backendSearch.js";
import { demoTranslate } from "../mocks/copilot.js";

/**
 * Translates text via the backend FastAPI /users/translate endpoint.
 * Falls back to local demo dictionary if offline/unreachable.
 *
 * @param {string} text
 * @param {"vi-en" | "en-vi"} dir
 * @returns {Promise<string>}
 */
export async function translateText(text, dir = "vi-en") {
  if (!text || !text.trim()) {
    return "";
  }

  const from_lang = dir === "vi-en" ? "vi" : "en";
  const to_lang = dir === "vi-en" ? "en" : "vi";

  try {
    const { baseUrl } = getSearchConfig();
    const cleanBase = (baseUrl || "").replace(/\/users$/i, "").replace(/\/+$/, "");
    const endpoint = cleanBase ? `${cleanBase}/users/translate` : "/users/translate";

    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text.trim(),
        from_lang,
        to_lang,
      }),
    });

    if (response.ok) {
      const data = await response.json();
      if (data && data.translated_text) {
        return data.translated_text;
      }
    }
  } catch (error) {
    console.warn("Backend translation API error, falling back to local:", error);
  }

  return demoTranslate(text, dir);
}
