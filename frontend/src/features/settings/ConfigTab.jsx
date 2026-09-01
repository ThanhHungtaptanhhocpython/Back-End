import { useMemo, useState } from "react";
import { Alert, Collapse, Empty, Input, Switch, Tooltip } from "antd";

import FieldRow from "./FieldRow.jsx";
import { AI_INFERENCE_EXTRA_KEYS, AI_INFERENCE_GROUP, visibleGroups } from "./settingsModel.js";

const DEFAULT_OPEN = ["Server", "Data/Media"];
// AI provider + inference config lives on the AI Providers tab.
const HIDDEN_GROUPS = ["AI", AI_INFERENCE_GROUP];

export default function ConfigTab({ schema, config, form }) {
  const { values, secrets, errors, changed, setValue, setSecret } = form;

  const [query, setQuery] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [modifiedOnly, setModifiedOnly] = useState(false);

  const groups = useMemo(
    () =>
      visibleGroups(schema, {
        query,
        showAdvanced,
        modifiedOnly,
        changedKeys: changed,
        values,
        excludeGroups: HIDDEN_GROUPS,
        excludeKeys: AI_INFERENCE_EXTRA_KEYS,
      }),
    [schema, query, showAdvanced, modifiedOnly, changed, values],
  );
  const totalHidden = groups.reduce((n, g) => n + g.hiddenCount, 0);
  const openKeys = query || modifiedOnly ? groups.map((g) => g.group) : DEFAULT_OPEN;

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
        <span className="set-dim" style={{ fontSize: 12 }}>
          AI provider &amp; inference settings are on the <b>AI Providers</b> tab.
        </span>
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
                    error={errors[spec.key]}
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
