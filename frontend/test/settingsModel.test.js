import assert from "node:assert/strict";
import test from "node:test";

import {
  buildSavePayload,
  dirtyKeys,
  fieldMatches,
  formatBytes,
  indexFields,
  initialFormState,
  isHiddenByCondition,
  partitionAiFields,
  validateAll,
  validateField,
  visibleGroups,
} from "../src/features/settings/settingsModel.js";

const SCHEMA = {
  groups: [
    {
      group: "Server",
      help: "server help",
      fields: [
        { key: "PORT", label: "Port", kind: "int", secret: false, minimum: 1, maximum: 65535, advanced: false },
        { key: "HOST", label: "Host", kind: "str", secret: false, advanced: false },
        { key: "SRC_DIR", label: "Src Dir", kind: "path", secret: false, locked: true, advanced: false },
        { key: "ENV", label: "Env", kind: "choice", secret: false, choices: ["development", "production"], advanced: false },
        { key: "CHAT_HISTORY_MESSAGES", label: "Chat History", kind: "int", secret: false, advanced: true },
      ],
    },
    {
      group: "AI",
      help: "ai help",
      fields: [
        { key: "AI_GATEWAY_ENABLED", label: "Gateway", kind: "bool", secret: false, advanced: false },
        { key: "OPENROUTER_BASE_URL", label: "Base URL", kind: "url", secret: false, advanced: true },
        { key: "PLAYBACK_OFFSETS_JSON", label: "Offsets", kind: "json_object", secret: false, advanced: true },
        { key: "OPENROUTER_API_KEY", label: "Key", kind: "secret", secret: true, advanced: false },
      ],
    },
  ],
};

const CONFIG = {
  values: { PORT: "3000", HOST: "0.0.0.0", SRC_DIR: "/app/src", ENV: "development",
            AI_GATEWAY_ENABLED: "false", OPENROUTER_BASE_URL: "https://openrouter.ai/api/v1",
            PLAYBACK_OFFSETS_JSON: "", OPENROUTER_API_KEY: "********" },
  secrets: { OPENROUTER_API_KEY: true },
};

test("indexFields flattens groups", () => {
  const fields = indexFields(SCHEMA);
  assert.equal(Object.keys(fields).length, 9);
  assert.equal(fields.PORT.kind, "int");
});

test("initialFormState stringifies values and blanks secrets", () => {
  const { values, secrets } = initialFormState(SCHEMA, CONFIG);
  assert.equal(values.PORT, "3000");
  assert.equal(values.OPENROUTER_API_KEY, undefined); // secret, not a plain value
  assert.deepEqual(secrets.OPENROUTER_API_KEY, { value: "", clear: false, configured: true });
});

test("validateField enforces kind + range, blank always ok", () => {
  const port = SCHEMA.groups[0].fields[0];
  assert.equal(validateField(port, ""), "");
  assert.equal(validateField(port, "8080"), "");
  assert.match(validateField(port, "70000"), /<= 65535/);
  assert.match(validateField(port, "abc"), /whole number/);

  const env = SCHEMA.groups[0].fields[3];
  assert.match(validateField(env, "staging"), /one of/);

  const url = SCHEMA.groups[1].fields[1];
  assert.match(validateField(url, "ftp://x"), /http/);

  const json = SCHEMA.groups[1].fields[2];
  assert.match(validateField(json, "[1,2]"), /object/);
  assert.equal(validateField(json, '{"a":1}'), "");
});

test("validateAll skips secret and locked fields", () => {
  const fields = indexFields(SCHEMA);
  const errors = validateAll(fields, { PORT: "-1", SRC_DIR: "", OPENROUTER_API_KEY: "x" });
  assert.deepEqual(Object.keys(errors), ["PORT"]);
});

test("dirtyKeys detects changes and ignores locked fields", () => {
  const fields = indexFields(SCHEMA);
  const changed = dirtyKeys(fields, CONFIG.values, {
    ...CONFIG.values,
    PORT: "4000",
    SRC_DIR: "/somewhere/else",
  });
  assert.deepEqual(changed, ["PORT"]);
});

test("buildSavePayload drops empty numerics, keeps empty strings, splits secrets", () => {
  const fields = indexFields(SCHEMA);
  const values = {
    PORT: "", // int -> dropped
    HOST: "", // str -> kept (explicit clear)
    ENV: "production",
    AI_GATEWAY_ENABLED: "true",
    OPENROUTER_BASE_URL: "https://openrouter.ai/api/v1",
    PLAYBACK_OFFSETS_JSON: "",
  };
  const secrets = {
    OPENROUTER_API_KEY: { value: "  ", clear: false, configured: true }, // blank -> keep
  };
  const payload = buildSavePayload(fields, values, secrets, "note");
  assert.equal("PORT" in payload.values, false);
  assert.equal(payload.values.HOST, "");
  assert.equal(payload.values.ENV, "production");
  assert.deepEqual(payload.secret_set, {});
  assert.deepEqual(payload.secret_clear, []);
  assert.equal(payload.note, "note");

  const withSecret = buildSavePayload(
    fields,
    values,
    { OPENROUTER_API_KEY: { value: "sk-new", clear: false, configured: true } },
  );
  assert.deepEqual(withSecret.secret_set, { OPENROUTER_API_KEY: "sk-new" });

  const cleared = buildSavePayload(
    fields,
    values,
    { OPENROUTER_API_KEY: { value: "", clear: true, configured: true } },
  );
  assert.deepEqual(cleared.secret_clear, ["OPENROUTER_API_KEY"]);
});

test("buildSavePayload with onlyKeys sends just the changed non-secret fields", () => {
  const fields = indexFields(SCHEMA);
  const values = { PORT: "4000", HOST: "1.2.3.4", ENV: "production", AI_GATEWAY_ENABLED: "true" };
  const payload = buildSavePayload(fields, values, {}, "", { onlyKeys: ["PORT", "AI_GATEWAY_ENABLED"] });
  assert.deepEqual(Object.keys(payload.values).sort(), ["AI_GATEWAY_ENABLED", "PORT"]);
  assert.equal(payload.values.PORT, "4000");
});

const AI_SCHEMA = {
  groups: [
    {
      group: "AI",
      fields: [
        { key: "AI_GATEWAY_ENABLED", kind: "bool" },
        { key: "AI_TEXT_PRIORITY", kind: "csv" },
        { key: "AI_VISION_PRIORITY", kind: "csv" },
        { key: "AI_LOCAL_FALLBACK_ENABLED", kind: "bool" },
        { key: "AI_GATEWAY_MAX_TOKENS", kind: "int" },
        { key: "NIM_ENABLED", kind: "bool" },
        { key: "NIM_API_KEY", kind: "secret", secret: true },
        { key: "NIM_TEXT_MODEL", kind: "str" },
        { key: "KILO_ENABLED", kind: "bool" },
        { key: "KILO_API_KEY", kind: "secret", secret: true },
        { key: "OPENROUTER_ENABLED", kind: "bool" },
        { key: "OPENROUTER_API_KEY", kind: "secret", secret: true },
        { key: "OPENROUTER_MODEL", kind: "str" },
        { key: "OPENROUTER_SITE_URL", kind: "url" }, // legacy
        { key: "LLM_PROVIDER", kind: "choice" }, // legacy
        { key: "ANTHROPIC_API_KEY", kind: "secret", secret: true }, // legacy
      ],
    },
    {
      group: "Agent/VLM",
      fields: [
        { key: "AGENT_LLM_ENABLED", kind: "bool" },
        { key: "AGENT_LLM_MODEL", kind: "str" },
        { key: "AGENT_VISUAL_QUERY_LIMIT", kind: "int" },
        { key: "AGENT_VLM_ENABLED", kind: "bool" },
        { key: "AGENT_VLM_BATCH_SIZE", kind: "int" },
        { key: "KIS_VQA_RERANK", kind: "bool" },
      ],
    },
    {
      group: "TRAKE",
      fields: [
        { key: "TRAKE_OCR_ENABLED", kind: "bool" },
        { key: "TRAKE_BEAM_WIDTH", kind: "int" },
        { key: "TRAKE_VLM_ENABLED", kind: "bool" },
        { key: "TRAKE_ENABLE_VQA", kind: "bool" },
      ],
    },
    {
      group: "Q&A",
      fields: [
        { key: "QA_RETRIEVAL_POOL", kind: "int" },
        { key: "QA_VLM_ENABLED", kind: "bool" },
        { key: "QA_MAX_TOKENS", kind: "int" },
      ],
    },
    { group: "Server", fields: [{ key: "PORT", kind: "int" }] },
  ],
};

test("partitionAiFields splits gateway / provider cards / legacy / inference", () => {
  const p = partitionAiFields(AI_SCHEMA);
  assert.deepEqual(p.gateway.map((f) => f.key), [
    "AI_GATEWAY_ENABLED", "AI_TEXT_PRIORITY", "AI_VISION_PRIORITY",
    "AI_LOCAL_FALLBACK_ENABLED", "AI_GATEWAY_MAX_TOKENS",
  ]);
  assert.deepEqual(p.providers.nim.map((f) => f.key), ["NIM_ENABLED", "NIM_API_KEY", "NIM_TEXT_MODEL"]);
  assert.deepEqual(p.providers.kilo.map((f) => f.key), ["KILO_ENABLED", "KILO_API_KEY"]);
  assert.deepEqual(p.providers.openrouter.map((f) => f.key), [
    "OPENROUTER_ENABLED", "OPENROUTER_API_KEY", "OPENROUTER_MODEL",
  ]);
  assert.deepEqual(p.legacy.map((f) => f.key).sort(), [
    "ANTHROPIC_API_KEY", "LLM_PROVIDER", "OPENROUTER_SITE_URL",
  ]);
  assert.deepEqual(p.inference.planner.map((f) => f.key), ["AGENT_LLM_ENABLED", "AGENT_LLM_MODEL", "AGENT_VISUAL_QUERY_LIMIT"]);
  assert.deepEqual(p.inference.vlm.map((f) => f.key), ["AGENT_VLM_ENABLED", "AGENT_VLM_BATCH_SIZE"]);
  assert.deepEqual(p.inference.kis.map((f) => f.key), ["KIS_VQA_RERANK"]);
  assert.deepEqual(p.inference.trake.map((f) => f.key), ["TRAKE_VLM_ENABLED", "TRAKE_ENABLE_VQA"]);
  assert.deepEqual(p.inference.qa.map((f) => f.key), ["QA_VLM_ENABLED", "QA_MAX_TOKENS"]);
});

test("visibleGroups drops the AI + Agent/VLM groups and the moved TRAKE/Q&A keys", () => {
  const groups = visibleGroups(AI_SCHEMA, {
    excludeGroups: ["AI", "Agent/VLM"],
    excludeKeys: ["TRAKE_VLM_ENABLED", "TRAKE_ENABLE_VQA", "QA_VLM_ENABLED", "QA_MAX_TOKENS"],
  });
  assert.deepEqual(groups.map((g) => g.group), ["TRAKE", "Q&A", "Server"]);
  assert.deepEqual(groups.find((g) => g.group === "TRAKE").fields.map((f) => f.key), [
    "TRAKE_OCR_ENABLED", "TRAKE_BEAM_WIDTH",
  ]);
  assert.deepEqual(groups.find((g) => g.group === "Q&A").fields.map((f) => f.key), ["QA_RETRIEVAL_POOL"]);
});

test("formatBytes is human readable", () => {
  assert.equal(formatBytes(512), "512 B");
  assert.equal(formatBytes(2048), "2 KB");
  assert.equal(formatBytes(5 * 1024 * 1024), "5 MB");
});

test("fieldMatches searches key, label and help", () => {
  const spec = { key: "OPENROUTER_API_KEY", label: "Key", help: "server-side only" };
  assert.equal(fieldMatches(spec, ""), true);
  assert.equal(fieldMatches(spec, "openrouter"), true);
  assert.equal(fieldMatches(spec, "server-side"), true);
  assert.equal(fieldMatches(spec, "azure"), false);
});

test("visibleGroups hides advanced fields by default", () => {
  const groups = visibleGroups(SCHEMA, {});
  const server = groups.find((g) => g.group === "Server");
  assert.deepEqual(server.fields.map((f) => f.key).sort(), ["ENV", "HOST", "PORT", "SRC_DIR"]);
  assert.equal(server.hiddenCount, 1); // CHAT_HISTORY_MESSAGES
  assert.equal(server.help, "server help");
});

test("visibleGroups reveals everything with showAdvanced", () => {
  const groups = visibleGroups(SCHEMA, { showAdvanced: true });
  assert.equal(groups.find((g) => g.group === "AI").fields.length, 4);
});

test("a search reveals advanced matches and drops non-matching groups", () => {
  const groups = visibleGroups(SCHEMA, { query: "offsets" });
  assert.equal(groups.length, 1);
  assert.deepEqual(groups[0].fields.map((f) => f.key), ["PLAYBACK_OFFSETS_JSON"]);
});

test("modifiedOnly limits to changed keys and shows advanced ones", () => {
  const groups = visibleGroups(SCHEMA, { modifiedOnly: true, changedKeys: ["OPENROUTER_BASE_URL"] });
  assert.equal(groups.length, 1);
  assert.deepEqual(groups[0].fields.map((f) => f.key), ["OPENROUTER_BASE_URL"]);
});

test("isHiddenByCondition matches hide_when against current values", () => {
  const spec = { key: "AGENT_LLM_MODEL", hide_when: { AI_GATEWAY_ENABLED: "true" } };
  assert.equal(isHiddenByCondition(spec, { AI_GATEWAY_ENABLED: "true" }), true);
  assert.equal(isHiddenByCondition(spec, { AI_GATEWAY_ENABLED: "false" }), false);
  assert.equal(isHiddenByCondition({ key: "X" }, {}), false);
});

test("visibleGroups drops fields whose hide_when currently matches, even under search", () => {
  const schema = {
    groups: [
      {
        group: "AI",
        fields: [
          { key: "AGENT_LLM_ENABLED", label: "planner", kind: "bool", advanced: false },
          {
            key: "AGENT_LLM_MODEL",
            label: "planner model",
            kind: "str",
            advanced: false,
            hide_when: { AI_GATEWAY_ENABLED: "true" },
          },
        ],
      },
    ],
  };
  const gatewayOn = visibleGroups(schema, { values: { AI_GATEWAY_ENABLED: "true" }, query: "planner" });
  assert.deepEqual(gatewayOn[0].fields.map((f) => f.key), ["AGENT_LLM_ENABLED"]);

  const gatewayOff = visibleGroups(schema, { values: { AI_GATEWAY_ENABLED: "false" } });
  assert.deepEqual(gatewayOff[0].fields.map((f) => f.key), ["AGENT_LLM_ENABLED", "AGENT_LLM_MODEL"]);
});
