import assert from "node:assert/strict";
import test from "node:test";

import {
  buildSavePayload,
  dirtyKeys,
  formatBytes,
  indexFields,
  initialFormState,
  validateAll,
  validateField,
} from "../src/features/settings/settingsModel.js";

const SCHEMA = {
  groups: [
    {
      group: "Server",
      fields: [
        { key: "PORT", kind: "int", secret: false, minimum: 1, maximum: 65535 },
        { key: "HOST", kind: "str", secret: false },
        { key: "SRC_DIR", kind: "path", secret: false, locked: true },
        { key: "ENV", kind: "choice", secret: false, choices: ["development", "production"] },
      ],
    },
    {
      group: "AI",
      fields: [
        { key: "AI_GATEWAY_ENABLED", kind: "bool", secret: false },
        { key: "OPENROUTER_BASE_URL", kind: "url", secret: false },
        { key: "PLAYBACK_OFFSETS_JSON", kind: "json_object", secret: false },
        { key: "OPENROUTER_API_KEY", kind: "secret", secret: true },
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
  assert.equal(Object.keys(fields).length, 8);
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

test("formatBytes is human readable", () => {
  assert.equal(formatBytes(512), "512 B");
  assert.equal(formatBytes(2048), "2 KB");
  assert.equal(formatBytes(5 * 1024 * 1024), "5 MB");
});
