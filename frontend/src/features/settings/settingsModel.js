/**
 * Pure helpers shared by the Settings UI (no React, no DOM) so the behaviour
 * that matters -- initial values, dirty diff, client-side validation, the save
 * payload -- is unit-testable with `node:test`.
 */

export const SKIP_EMPTY_KINDS = new Set(["bool", "int", "float", "choice"]);

/** Flatten `schema.groups[*].fields[*]` into `{ KEY: fieldSpec }`. */
export function indexFields(schema) {
  const out = {};
  for (const group of schema?.groups || []) {
    for (const field of group.fields || []) out[field.key] = field;
  }
  return out;
}

/**
 * Build the editable form state from `/settings/config`.
 * Non-secret values come through verbatim; secret fields become "" (blank means
 * "keep current") and carry a `__configured` flag from `config.secrets`.
 */
export function initialFormState(schema, config) {
  const fields = indexFields(schema);
  const values = {};
  const secrets = {};
  for (const [key, spec] of Object.entries(fields)) {
    if (spec.secret) {
      secrets[key] = { value: "", clear: false, configured: Boolean(config?.secrets?.[key]) };
    } else {
      values[key] = String(config?.values?.[key] ?? "");
    }
  }
  return { values, secrets };
}

/** Keys whose non-secret value differs from the loaded config. */
export function dirtyKeys(fields, loadedValues, currentValues) {
  const changed = [];
  for (const key of Object.keys(currentValues || {})) {
    if (fields[key]?.locked) continue;
    if (String(loadedValues?.[key] ?? "") !== String(currentValues[key] ?? "")) changed.push(key);
  }
  return changed;
}

const URL_RE = /^https?:\/\/.+/i;

/** Client-side mirror of the backend field validation. Returns "" when ok. */
export function validateField(spec, raw) {
  const text = String(raw ?? "").trim();
  if (text === "") return "";
  switch (spec.kind) {
    case "int": {
      if (!/^-?\d+$/.test(text)) return "must be a whole number";
      return rangeError(spec, Number(text));
    }
    case "float": {
      if (!Number.isFinite(Number(text))) return "must be a number";
      return rangeError(spec, Number(text));
    }
    case "bool":
      return ["true", "false"].includes(text.toLowerCase()) ? "" : "must be true or false";
    case "choice":
      return !spec.choices || spec.choices.includes(text)
        ? ""
        : `must be one of: ${spec.choices.join(", ")}`;
    case "url":
      return URL_RE.test(text) ? "" : "must be an http(s) URL";
    case "json":
    case "json_object": {
      try {
        const parsed = JSON.parse(text);
        if (spec.kind === "json_object" && (typeof parsed !== "object" || Array.isArray(parsed)))
          return "must be a JSON object";
        return "";
      } catch {
        return "invalid JSON";
      }
    }
    default:
      return "";
  }
}

function rangeError(spec, value) {
  if (spec.minimum != null && value < spec.minimum) return `must be >= ${spec.minimum}`;
  if (spec.maximum != null && value > spec.maximum) return `must be <= ${spec.maximum}`;
  return "";
}

/** Validate every editable field; returns `{ KEY: message }` for failures only. */
export function validateAll(fields, values) {
  const errors = {};
  for (const [key, spec] of Object.entries(fields)) {
    if (spec.secret || spec.locked) continue;
    const message = validateField(spec, values[key]);
    if (message) errors[key] = message;
  }
  return errors;
}

/**
 * Build the POST /settings/config body.
 * - `values`: non-secret, non-locked fields (empty numeric/bool/choice dropped
 *   so the backend keeps its default). With `onlyKeys`, restricted to those
 *   keys — the store then records just what the user actually changed and code
 *   defaults keep applying to everything else.
 * - `secret_set`: secrets with a non-blank new value.
 * - `secret_clear`: secrets explicitly marked for deletion.
 */
export function buildSavePayload(fields, values, secrets, note = "", { onlyKeys } = {}) {
  const limit = onlyKeys ? new Set(onlyKeys) : null;
  const outValues = {};
  for (const [key, spec] of Object.entries(fields)) {
    if (spec.secret || spec.locked) continue;
    if (limit && !limit.has(key)) continue;
    const text = String(values[key] ?? "").trim();
    if (text === "" && SKIP_EMPTY_KINDS.has(spec.kind)) continue;
    outValues[key] = text;
  }
  const secret_set = {};
  const secret_clear = [];
  for (const [key, entry] of Object.entries(secrets || {})) {
    if (entry?.clear) secret_clear.push(key);
    else if (String(entry?.value ?? "").trim() !== "") secret_set[key] = entry.value;
  }
  return { values: outValues, secret_set, secret_clear, note };
}

/** Case-insensitive match of a field against a search query (key/label/help). */
export function fieldMatches(spec, query) {
  const q = String(query || "").trim().toLowerCase();
  if (!q) return true;
  return (
    spec.key.toLowerCase().includes(q) ||
    String(spec.label || "").toLowerCase().includes(q) ||
    String(spec.help || "").toLowerCase().includes(q)
  );
}

/** True when `spec.hide_when` matches the current form values (field N/A right now). */
export function isHiddenByCondition(spec, values = {}) {
  const cond = spec.hide_when;
  if (!cond) return false;
  return Object.entries(cond).every(
    ([key, val]) => String(values[key] ?? "").toLowerCase() === String(val).toLowerCase(),
  );
}

/**
 * Groups with their fields filtered by the toolbar controls.
 * `{ query, showAdvanced, modifiedOnly, changedKeys, values }`
 *   -> `[{ group, help, fields, hiddenCount }]`
 * (only groups with at least one visible field are returned).
 */
export function visibleGroups(
  schema,
  {
    query = "",
    showAdvanced = false,
    modifiedOnly = false,
    changedKeys = [],
    values = {},
    excludeGroups = [],
    excludeKeys = [],
  } = {},
) {
  const changed = new Set(changedKeys);
  const skip = new Set(excludeGroups);
  const skipKeys = new Set(excludeKeys);
  const searching = String(query || "").trim() !== "";
  const out = [];
  for (const group of schema?.groups || []) {
    if (skip.has(group.group)) continue;
    let hidden = 0;
    const fields = (group.fields || []).filter((spec) => {
      if (skipKeys.has(spec.key)) return false;
      // A field made irrelevant by another toggle is always hidden.
      if (isHiddenByCondition(spec, values)) return false;
      if (modifiedOnly && !changed.has(spec.key)) return false;
      if (!fieldMatches(spec, query)) return false;
      // A search or "modified only" reveals advanced fields too.
      if (spec.advanced && !showAdvanced && !searching && !modifiedOnly) {
        hidden += 1;
        return false;
      }
      return true;
    });
    if (fields.length) out.push({ group: group.group, help: group.help || "", fields, hiddenCount: hidden });
  }
  return out;
}

// -- AI Providers tab -----------------------------------------------------
export const AI_GATEWAY_KEYS = [
  "AI_GATEWAY_ENABLED",
  "AI_TEXT_PRIORITY",
  "AI_VISION_PRIORITY",
  "AI_LOCAL_FALLBACK_ENABLED",
  "AI_GATEWAY_MAX_TOKENS",
];

// AI-group keys that belong to the legacy pre-gateway path, not a provider card.
const AI_LEGACY_KEYS = new Set([
  "LLM_PROVIDER",
  "OPENAI_API_KEY", "OPENAI_MODEL",
  "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "ANTHROPIC_MAX_TOKENS",
  "NVIDIA_API_KEY", "NVIDIA_MODEL", "NVIDIA_MAX_TOKENS", "NVIDIA_TOP_P",
  "GOOGLE_API_KEY", "GOOGLE_MODEL",
  "OPENROUTER_SITE_URL", "OPENROUTER_APP_NAME", "OPENROUTER_MAX_TOKENS",
  "OPENROUTER_TRANSLATE_MODEL", "OPENROUTER_TRANSLATE_MAX_TOKENS",
]);

const AI_PROVIDER_PREFIX = {
  nim: "NIM_",
  cerebras: "CEREBRAS_",
  groq: "GROQ_",
  openrouter: "OPENROUTER_",
  gemini: "GEMINI_",
  cloudflare: "CLOUDFLARE_",
  kilo: "KILO_",
};

// The whole "Agent/VLM" group is LLM-planner / VLM-verifier config, plus these
// VLM toggles that otherwise sit in TRAKE / Q&A. All of it moves to the AI
// Providers tab as an "AI inference" section.
export const AI_INFERENCE_GROUP = "Agent/VLM";
export const AI_INFERENCE_EXTRA_KEYS = [
  "TRAKE_VLM_ENABLED",
  "TRAKE_VLM_MAX_SEQUENCES",
  "TRAKE_ENABLE_VQA",
  "TRAKE_VQA_MAX_SEQUENCES",
  "QA_VLM_ENABLED",
  "QA_MAX_TOKENS",
];

/**
 * Split the schema for the AI Providers tab:
 *   { gateway, providers: {id:[specs]}, legacy,
 *     inference: { planner:[], vlm:[], kis:[], trake:[], qa:[] } }
 */
export function partitionAiFields(schema) {
  const groups = schema?.groups || [];
  const ai = groups.find((g) => g.group === "AI");
  const fields = ai ? ai.fields : [];
  const byKey = Object.fromEntries(fields.map((f) => [f.key, f]));

  const gateway = AI_GATEWAY_KEYS.map((k) => byKey[k]).filter(Boolean);

  const providers = {};
  const claimed = new Set(AI_GATEWAY_KEYS);
  for (const [id, prefix] of Object.entries(AI_PROVIDER_PREFIX)) {
    const specs = fields.filter((f) => f.key.startsWith(prefix) && !AI_LEGACY_KEYS.has(f.key));
    if (specs.length) {
      providers[id] = specs;
      specs.forEach((s) => claimed.add(s.key));
    }
  }
  const legacy = fields.filter((f) => !claimed.has(f.key));

  const agentGroup = groups.find((g) => g.group === AI_INFERENCE_GROUP);
  const agentFields = agentGroup ? agentGroup.fields : [];
  const extraByKey = {};
  for (const group of groups) {
    for (const f of group.fields || []) {
      if (AI_INFERENCE_EXTRA_KEYS.includes(f.key)) extraByKey[f.key] = f;
    }
  }
  const inference = {
    planner: agentFields.filter((f) => f.key.startsWith("AGENT_LLM_") || f.key === "AGENT_VISUAL_QUERY_LIMIT"),
    vlm: agentFields.filter((f) => f.key.startsWith("AGENT_VLM_")),
    kis: agentFields.filter((f) => f.key.startsWith("KIS_")),
    trake: AI_INFERENCE_EXTRA_KEYS.filter((k) => k.startsWith("TRAKE_")).map((k) => extraByKey[k]).filter(Boolean),
    qa: AI_INFERENCE_EXTRA_KEYS.filter((k) => k.startsWith("QA_")).map((k) => extraByKey[k]).filter(Boolean),
  };

  return { gateway, providers, legacy, inference };
}

export function formatBytes(bytes) {
  const n = Number(bytes) || 0;
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = n / 1024;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  const rounded = Math.round(value * 10) / 10;
  const text = Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
  return `${text} ${units[i]}`;
}

export const RESTART_STATE_LABEL = {
  idle: "Idle",
  restarting: "Restarting…",
  "polling-health": "Waiting for health…",
  healthy: "Healthy",
  "rolling-back": "Rolling back…",
  "rollback-complete": "Rolled back",
  failed: "Failed — manual fix needed",
};
