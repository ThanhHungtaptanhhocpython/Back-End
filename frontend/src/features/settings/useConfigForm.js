import { useEffect, useMemo, useState } from "react";
import { message } from "antd";

import { saveConfig, validateConfig } from "../../services/settingsApi.js";
import { buildSavePayload, dirtyKeys, indexFields, validateAll } from "./settingsModel.js";

/**
 * Shared editable-config state for the Settings tabs. One instance lives in
 * SettingsPage so edits made in the Configuration tab and the AI Providers tab
 * are the same unsaved-changes set and a single Save persists everything.
 */
export function useConfigForm(schema, config) {
  const fields = useMemo(() => indexFields(schema), [schema]);

  const [values, setValues] = useState({});
  const [secrets, setSecrets] = useState({});
  const [errors, setErrors] = useState({});
  const [serverErrors, setServerErrors] = useState({});
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!schema || !config) return;
    const nextValues = {};
    const nextSecrets = {};
    for (const spec of Object.values(indexFields(schema))) {
      if (spec.secret) {
        nextSecrets[spec.key] = {
          value: "",
          clear: false,
          configured: Boolean(config.secrets?.[spec.key]),
        };
      } else {
        nextValues[spec.key] = String(config.values?.[spec.key] ?? "");
      }
    }
    setValues(nextValues);
    setSecrets(nextSecrets);
    setErrors({});
    setServerErrors({});
  }, [schema, config]);

  const loaded = config?.values || {};
  const changed = dirtyKeys(fields, loaded, values);
  const secretChanged = Object.values(secrets).some((s) => s.clear || s.value.trim() !== "");
  const dirty = changed.length > 0 || secretChanged;
  const allErrors = { ...errors, ...serverErrors };

  function setValue(key, next) {
    setValues((v) => ({ ...v, [key]: next }));
    setServerErrors((e) => (e[key] ? { ...e, [key]: undefined } : e));
  }
  function setSecret(key, entry) {
    setSecrets((s) => ({ ...s, [key]: entry }));
  }

  async function runValidate() {
    const local = validateAll(fields, values);
    setErrors(local);
    try {
      const res = await validateConfig({
        values: Object.fromEntries(changed.map((k) => [k, values[k]])),
      });
      setServerErrors(res.errors || {});
      const ok = res.ok && !Object.keys(local).length;
      if (ok) message.success("Configuration is valid.");
      return ok;
    } catch (err) {
      message.error(err.message);
      return false;
    }
  }

  async function runSave(onSaved) {
    const local = validateAll(fields, values);
    setErrors(local);
    if (Object.keys(local).length) {
      message.error("Fix the highlighted fields first.");
      return;
    }
    setBusy(true);
    try {
      const payload = buildSavePayload(fields, values, secrets, "Updated via Settings UI", {
        onlyKeys: changed,
      });
      const res = await saveConfig(payload);
      if (!res.ok) {
        setServerErrors(res.errors || {});
        message.error(res.detail || "Save rejected.");
        return;
      }
      setServerErrors({});
      message.success(`Saved as revision ${res.revision_id}. Restart to apply.`);
      onSaved?.(res);
    } catch (err) {
      if (err.body?.errors) setServerErrors(err.body.errors);
      message.error(err.message);
    } finally {
      setBusy(false);
    }
  }

  return {
    fields,
    values,
    secrets,
    errors: allErrors,
    serverErrors,
    changed,
    secretChanged,
    dirty,
    busy,
    setValue,
    setSecret,
    runValidate,
    runSave,
  };
}
