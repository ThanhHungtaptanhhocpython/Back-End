/**
 * Decide what a translated query does when the user picks a destination in the
 * Translate panel's "Use as query" menu.
 *
 * Every destination opens a NEW tab; the caller never mutates the tab currently
 * on screen. `Text` / `Q&A` run straight away. `Temporal` runs the existing
 * temporal parser: two or more events run immediately, a single event is
 * pre-filled as Event 1 and left for the user to complete (no request is sent).
 */

import {
  isRunnableTemporalQuery,
  parseTemporalQuery,
  serializeTemporalQuery,
} from "./temporalQuery.js";

/** Destinations offered by the "Use as query" menu (OCR is intentionally absent). */
export const TRANSLATION_TARGETS = Object.freeze([
  { value: "TEXT", label: "Text" },
  { value: "QA", label: "Q&A" },
  { value: "TEMPORAL", label: "Temporal" },
]);

const TAB_LABEL = { TEXT: "Text", QA: "Q&A", TEMPORAL: "Temporal" };

/**
 * Whether the destination menu should be enabled: there must be output text and
 * it must be a live translation or one the user has edited.
 * @param {{ text?: string, live?: boolean, edited?: boolean }} state
 */
export function canTargetTranslation({ text, live, edited } = {}) {
  return Boolean(String(text || "").trim()) && (Boolean(live) || Boolean(edited));
}

/**
 * @param {string} text         the translated (or user-edited) query
 * @param {"TEXT"|"QA"|"TEMPORAL"} destination
 * @returns {null | {
 *   searchType: "TEXT"|"QA"|"TEMPORAL",
 *   tabLabel: string,
 *   query: string,
 *   run: boolean,               run the search immediately
 *   needsSecondEvent: boolean,  temporal: only one event -> prefill E1, focus for E2
 *   note: string,
 * }}
 */
export function planTranslationTarget(text, destination) {
  const query = String(text || "").trim();
  const target = String(destination || "").toUpperCase();
  if (!query || !TAB_LABEL[target]) return null;

  if (target === "TEXT" || target === "QA") {
    return {
      searchType: target,
      tabLabel: TAB_LABEL[target],
      query,
      run: true,
      needsSecondEvent: false,
      note: `Translated query opened in a new ${TAB_LABEL[target]} tab`,
    };
  }

  // TEMPORAL: reuse the existing parser (connectors, "E1:/E2:", numbered lists...).
  const parsed = parseTemporalQuery(query);
  if (isRunnableTemporalQuery(parsed)) {
    return {
      searchType: "TEMPORAL",
      tabLabel: TAB_LABEL.TEMPORAL,
      query: serializeTemporalQuery(parsed),
      run: true,
      needsSecondEvent: false,
      note: `Temporal query with ${parsed.events.length} events opened in a new tab`,
    };
  }

  const firstEvent = String(parsed.events[0] || query).trim();
  return {
    searchType: "TEMPORAL",
    tabLabel: TAB_LABEL.TEMPORAL,
    query: serializeTemporalQuery({ context: parsed.context, events: [firstEvent] }),
    run: false,
    needsSecondEvent: true,
    note: "Added Event 1 to a new Temporal tab - add Event 2 before running",
  };
}
