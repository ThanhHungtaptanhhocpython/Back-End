import { useMemo, useState } from "react";
import { Alert, Anchor, Button, Space, Spin, Tag, message } from "antd";

import { saveConfig, validateConfig } from "../../services/settingsApi.js";
import FieldRow from "./FieldRow.jsx";
import {
  buildSavePayload,
  dirtyKeys,
  indexFields,
  validateAll,
} from "./settingsModel.js";

export default function ConfigTab({ schema, config, loading, onSaved }) {
  const fields = useMemo(() => indexFields(schema), [schema]);
  const [values, setValues] = useState(() =>
    Object.fromEntries(
      Object.values(fields)
        .filter((f) => !f.secret)
        .map((f) => [f.key, String(config?.values?.[f.key] ?? "")]),
    ),
  );
  const [secrets, setSecrets] = useState(() =>
    Object.fromEntries(
      Object.values(fields)
        .filter((f) => f.secret)
        .map((f) => [f.key, { value: "", clear: false, configured: Boolean(config?.secrets?.[f.key]) }]),
    ),
  );
  const [errors, setErrors] = useState({});
  const [serverErrors, setServerErrors] = useState({});
  const [busy, setBusy] = useState(false);

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
      const res = await validateConfig({ values: pickChanged(changed, values) });
      setServerErrors(res.errors || {});
      if (res.ok && !Object.keys(local).length) message.success("Configuration is valid.");
      return res.ok && !Object.keys(local).length;
    } catch (err) {
      message.error(err.message);
      return false;
    }
  }

  async function runSave() {
    const local = validateAll(fields, values);
    setErrors(local);
    if (Object.keys(local).length) {
      message.error("Fix the highlighted fields first.");
      return;
    }
    setBusy(true);
    try {
      const payload = buildSavePayload(fields, values, secrets, "Updated via Settings UI");
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

  if (loading) return <Spin style={{ display: "block", margin: "48px auto" }} />;

  const groups = schema?.groups || [];

  return (
    <div className="set-config">
      <div className="set-config-toolbar">
        <Space>
          <Button onClick={runValidate}>Validate</Button>
          <Button type="primary" loading={busy} disabled={!dirty} onClick={runSave}>
            Save{dirty ? ` (${changed.length + (secretChanged ? 1 : 0)})` : ""}
          </Button>
        </Space>
        {dirty ? (
          <Tag color="orange">unsaved changes — a restart is required to apply</Tag>
        ) : (
          <Tag color="default">no changes</Tag>
        )}
      </div>

      {Object.keys(serverErrors).length > 0 && (
        <Alert
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
          message="The backend rejected some values"
          description={Object.entries(serverErrors)
            .filter(([, v]) => v)
            .map(([k, v]) => `${k}: ${v}`)
            .join(" · ")}
        />
      )}

      <div className="set-config-body">
        <Anchor
          className="set-config-nav"
          affix={false}
          items={groups.map((g) => ({ key: g.group, href: `#g-${slug(g.group)}`, title: g.group }))}
        />
        <div className="set-config-groups">
          {groups.map((group) => (
            <section key={group.group} id={`g-${slug(group.group)}`} className="set-group">
              <h3 className="set-group-title">{group.group}</h3>
              {group.fields.map((spec) => (
                <FieldRow
                  key={spec.key}
                  spec={spec}
                  value={values[spec.key]}
                  secretEntry={secrets[spec.key]}
                  error={allErrors[spec.key]}
                  onChange={(next) => setValue(spec.key, next)}
                  onSecret={(entry) => setSecret(spec.key, entry)}
                />
              ))}
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}

function pickChanged(keys, values) {
  return Object.fromEntries(keys.map((k) => [k, values[k]]));
}

function slug(text) {
  return String(text).toLowerCase().replace(/[^a-z0-9]+/g, "-");
}
