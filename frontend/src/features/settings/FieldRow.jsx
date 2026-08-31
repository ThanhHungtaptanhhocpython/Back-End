import { Checkbox, Input, InputNumber, Select, Switch, Tag, Tooltip } from "antd";

const { TextArea, Password } = Input;

/** One configuration field: label + help + the widget for its kind. */
export default function FieldRow({ spec, value, error, secretEntry, resolved, onChange, onSecret }) {
  const widget = spec.secret
    ? renderSecret(spec, secretEntry, onSecret)
    : renderValue(spec, value, onChange);

  const showResolved =
    spec.kind === "path" && !spec.locked && resolved && String(value ?? "").trim() === "";

  return (
    <div className={`set-field${error ? " set-field--error" : ""}`}>
      <div className="set-field-head">
        <label className="set-field-label">
          {spec.label}
          {spec.locked && (
            <Tooltip title="Derived from the install location; cannot be edited.">
              <Tag className="set-tag" color="default">locked</Tag>
            </Tooltip>
          )}
          {!spec.has_runtime_flow && (
            <Tooltip title="Declared for the UI but not wired to a runtime code path yet.">
              <Tag className="set-tag" color="warning">no runtime flow</Tag>
            </Tooltip>
          )}
          {spec.secret && secretEntry?.configured && !secretEntry?.clear && (
            <Tag className="set-tag" color="success">configured</Tag>
          )}
        </label>
        <code className="set-field-key">{spec.key}</code>
      </div>
      {spec.help && <div className="set-field-help">{spec.help}</div>}
      <div className="set-field-widget">{widget}</div>
      {showResolved && (
        <div className={`set-field-resolved${resolved.exists ? "" : " missing"}`}>
          Auto — resolves to <code>{resolved.path}</code>{" "}
          {resolved.exists ? "· found ✓" : "· not found on disk ✗"}
        </div>
      )}
      {error && <div className="set-field-msg">{error}</div>}
    </div>
  );
}

function renderValue(spec, value, onChange) {
  const common = { disabled: spec.locked, style: { width: "100%" } };
  switch (spec.kind) {
    case "bool":
      return (
        <Switch
          checked={String(value).toLowerCase() === "true"}
          disabled={spec.locked}
          onChange={(checked) => onChange(checked ? "true" : "false")}
        />
      );
    case "int":
    case "float":
      return (
        <InputNumber
          {...common}
          value={value === "" || value == null ? null : Number(value)}
          min={spec.minimum ?? undefined}
          max={spec.maximum ?? undefined}
          step={spec.kind === "float" ? 0.1 : 1}
          onChange={(next) => onChange(next == null ? "" : String(next))}
        />
      );
    case "choice":
      return (
        <Select
          {...common}
          value={value || undefined}
          allowClear
          options={(spec.choices || []).map((c) => ({ value: c, label: c }))}
          onChange={(next) => onChange(next ?? "")}
        />
      );
    case "json":
    case "json_object":
      return (
        <TextArea
          {...common}
          value={value}
          autoSize={{ minRows: 2, maxRows: 8 }}
          placeholder={spec.placeholder}
          onChange={(e) => onChange(e.target.value)}
        />
      );
    case "csv":
      return (
        <TextArea
          {...common}
          value={value}
          autoSize={{ minRows: 1, maxRows: 4 }}
          placeholder={spec.placeholder}
          onChange={(e) => onChange(e.target.value)}
        />
      );
    default:
      return (
        <Input
          {...common}
          value={value}
          placeholder={spec.placeholder}
          onChange={(e) => onChange(e.target.value)}
        />
      );
  }
}

function renderSecret(spec, entry, onSecret) {
  const current = entry || { value: "", clear: false, configured: false };
  return (
    <div className="set-secret">
      <Password
        style={{ width: "100%" }}
        value={current.value}
        disabled={current.clear}
        placeholder={
          current.configured ? "•••••••• (leave blank to keep)" : "not configured"
        }
        onChange={(e) => onSecret({ ...current, value: e.target.value })}
      />
      {current.configured && (
        <Checkbox
          checked={current.clear}
          onChange={(e) => onSecret({ ...current, clear: e.target.checked, value: "" })}
        >
          Delete this secret
        </Checkbox>
      )}
    </div>
  );
}
