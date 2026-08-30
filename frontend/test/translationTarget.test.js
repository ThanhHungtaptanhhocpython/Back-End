import assert from "node:assert/strict";
import test from "node:test";

import {
  canTargetTranslation,
  planTranslationTarget,
  TRANSLATION_TARGETS,
} from "../src/shared/translationTarget.js";
import { parseTemporalQuery } from "../src/shared/temporalQuery.js";
import { SEARCH_TYPES } from "../src/shared/constants.js";
import { SEARCH_ENDPOINTS } from "../src/services/backendSearch.js";

test("the destination menu offers only Text, Q&A and Temporal", () => {
  assert.deepEqual(
    TRANSLATION_TARGETS.map((t) => t.value),
    ["TEXT", "QA", "TEMPORAL"],
  );
});

test("canTargetTranslation gates the menu on a valid or edited translation", () => {
  assert.equal(canTargetTranslation({ text: "the woman in pink", live: true, edited: false }), true);
  assert.equal(canTargetTranslation({ text: "the woman in pink", live: false, edited: true }), true);
  assert.equal(canTargetTranslation({ text: "the woman in pink", live: false, edited: false }), false);
  assert.equal(canTargetTranslation({ text: "   ", live: true, edited: true }), false);
  assert.equal(canTargetTranslation({}), false);
});

test("Text destination opens a new Text tab and runs immediately", () => {
  const plan = planTranslationTarget("the woman wearing a pink shirt", "TEXT");
  assert.deepEqual(plan, {
    searchType: "TEXT",
    tabLabel: "Text",
    query: "the woman wearing a pink shirt",
    run: true,
    needsSecondEvent: false,
    note: "Translated query opened in a new Text tab",
  });
});

test("Q&A destination opens a new Q&A tab and runs immediately", () => {
  const plan = planTranslationTarget("is there a red bus at the station", "QA");
  assert.equal(plan.searchType, "QA");
  assert.equal(plan.tabLabel, "Q&A");
  assert.equal(plan.run, true);
  assert.equal(plan.needsSecondEvent, false);
});

test("Temporal destination splits a connector query and runs", () => {
  const plan = planTranslationTarget(
    "a man opens the door then walks into the room then sits down",
    "TEMPORAL",
  );
  assert.equal(plan.searchType, "TEMPORAL");
  assert.equal(plan.run, true);
  assert.equal(plan.needsSecondEvent, false);
  assert.deepEqual(parseTemporalQuery(plan.query).events, [
    "a man opens the door",
    "walks into the room",
    "sits down",
  ]);
});

test("Temporal destination with a single event prefills E1 and does not run", () => {
  const plan = planTranslationTarget("the woman wearing a pink shirt", "TEMPORAL");
  assert.equal(plan.searchType, "TEMPORAL");
  assert.equal(plan.run, false);
  assert.equal(plan.needsSecondEvent, true);
  assert.equal(plan.query, "E1: the woman wearing a pink shirt");
  assert.deepEqual(parseTemporalQuery(plan.query).events, ["the woman wearing a pink shirt"]);
});

test("blank text or an unknown destination yields no plan", () => {
  assert.equal(planTranslationTarget("   ", "TEXT"), null);
  assert.equal(planTranslationTarget("hello", "OCR"), null);
  assert.equal(planTranslationTarget("hello", ""), null);
});

test('"OCR Text" is the OCR type and routes to the OCR endpoint', () => {
  const ocr = SEARCH_TYPES.find((t) => t.label === "OCR Text");
  assert.equal(ocr.value, "OCR");
  assert.equal(SEARCH_ENDPOINTS.OCR, "ocrsearch");
  // Old value kept only as a compatibility alias.
  assert.equal(SEARCH_ENDPOINTS["OCR+OD"], "ocrandodsearch");
  assert.equal(
    SEARCH_TYPES.some((t) => t.value === "OCR+OD"),
    false,
  );
});
