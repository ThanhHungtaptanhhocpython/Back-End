import { useEffect, useMemo, useState } from "react";
import { Alert, Button, Collapse, Empty, Input, Space, Spin, Switch, Tag, Tooltip, message } from "antd";

import { saveConfig, validateConfig } from "../../services/settingsApi.js";
import FieldRow from "./FieldRow.jsx";
import {
  buildSavePayload,
  dirtyKeys,
  indexFields,
  validateAll,
  visibleGroups,
} from "./settingsModel.js";

const DEFAULT_OPEN = ["Server", "AI"];

export default function ConfigTab({ schema, config, loading, onSaved }) {
  const fields = useMemo(() => indexFields(schema), [schema]);

  const [values, setValues] = useState({});
  const [secrets, setSecrets] = useState({});
  const [errors, setErrors] = useState({});
  const [serverErrors, setServerErrors] = useState({});
  const [busy, setBusy] = useState(false);

  // toolbar
  const [query, setQuery] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [modifiedOnly, setModifiedOnly] = useState(false);

  // Re-seed the form whenever the loaded schema/config changes (initial load
  // AND after a save/restore refresh). A useState initializer would capture the
  // pre-fetch empty data and never update.
  useEffect(() => {
    if (!schema || !config) return;
    const nextValues = {};
    const nextSecrets = {};
    for (const spec of Object.values(indexFields(schema))) {
      if (spec.secret) {
        nextSecrets[spec.key] = { value: "", clear: false, configured: Boolean(config.secrets?.[spec.key]) };
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

  const groups = useMemo(
    () => visibleGroups(schema, { query, showAdvanced, modifiedOnly, changedKeys: changed, values }),
    [schema, query, showAdvanced, modifiedOnly, changed, values],
  );
  const totalHidden = groups.reduce((n, g) => n + g.hiddenCount, 0);
  const openKeys = query || modifiedOnly ? groups.map((g) => g.group) : DEFAULT_OPEN;

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
      const res = await validateConfig({ values: Object.fromEntries(changed.map((k) => [k, values[k]])) });
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
      const res = await saveConfig(buildSavePayload(fields, values, secrets, "Updated via Settings UI"));
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

  if (loading || !schema || !config) return <Spin style={{ display: "block", margin: "48px auto" }} />;

  const changeCount = changed.length + (secretChanged ? 1 : 0);

  return (
    <div className="set-config">
      <div className="set-config-toolbar">
        <Input.Search
          allowClear
          placeholder="Search settings…"
          style={{ maxWidth: 280 }}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <Tooltip title="Show tuning parameters and rarely-changed fields">
          <span className="set-toggle">
            <Switch size="small" checked={showAdvanced} onChange={setShowAdvanced} /> Advanced
          </span>
        </Tooltip>
        <Tooltip title="Only fields you have changed in this session">
          <span className="set-toggle">
            <Switch size="small" checked={modifiedOnly} onChange={setModifiedOnly} /> Modified only
          </span>
        </Tooltip>
        <span className="set-flex-spacer" />
        {dirty ? (
          <Tag color="orange">{changeCount} unsaved — restart required to apply</Tag>
        ) : (
          <Tag color="default">no changes</Tag>
        )}
        <Space>
          <Button onClick={runValidate}>Validate</Button>
          <Button type="primary" loading={busy} disabled={!dirty} onClick={runSave}>
            Save
          </Button>
        </Space>
      </div>

      {!showAdvanced && !query && !modifiedOnly && totalHidden > 0 && (
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 12 }}
          message={
            <>
              Showing the common settings. {totalHidden} advanced fields are hidden —{" "}
              <a onClick={() => setShowAdvanced(true)}>show all</a> or search by name.
            </>
          }
        />
      )}

      {Object.keys(serverErrors).some((k) => serverErrors[k]) && (
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

      {groups.length === 0 ? (
        <Empty description={modifiedOnly ? "No fields changed yet." : "No settings match your search."} />
      ) : (
        <Collapse
          defaultActiveKey={openKeys}
          activeKey={query || modifiedOnly ? openKeys : undefined}
          items={groups.map((group) => ({
            key: group.group,
            label: (
              <div className="set-group-head">
                <span className="set-group-title">{group.group}</span>
                <span className="set-dim">
                  {group.fields.length} field{group.fields.length === 1 ? "" : "s"}
                  {group.hiddenCount ? ` · ${group.hiddenCount} advanced hidden` : ""}
                </span>
              </div>
            ),
            children: (
              <>
                {group.help && <div className="set-group-help">{group.help}</div>}
                {group.fields.map((spec) => (
                  <FieldRow
                    key={spec.key}
                    spec={spec}
                    value={values[spec.key]}
                    secretEntry={secrets[spec.key]}
                    resolved={config.resolved?.[spec.key]}
                    error={allErrors[spec.key]}
                    onChange={(next) => setValue(spec.key, next)}
                    onSecret={(entry) => setSecret(spec.key, entry)}
                  />
                ))}
              </>
            ),
          }))}
        />
      )}
    </div>
  );
}
