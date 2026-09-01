import { useEffect, useMemo, useState } from "react";
import { Alert, Button, Collapse, Space, Spin, Tag, Typography, message } from "antd";

import { discoverModels, fetchProviders, testProvider } from "../../services/settingsApi.js";
import FieldRow from "./FieldRow.jsx";
import { partitionAiFields } from "./settingsModel.js";

const { Text, Paragraph } = Typography;

function ProviderExplainer({ gatewayEnabled }) {
  return (
    <Collapse
      className="set-explainer"
      defaultActiveKey={gatewayEnabled ? [] : ["what"]}
      items={[
        {
          key: "what",
          label: "What do AI providers do for this app?",
          children: (
            <div className="set-explainer-body">
              <Paragraph>
                The app runs <b>fully without any AI provider</b>: visual search (BEiT3),
                keyframe browsing, OCR/ASR (Elasticsearch), TRAKE and CSV export all work
                locally. Providers are optional and only power the language / vision steps
                below. When the gateway is off, or every provider in a chain fails, the app
                falls back to local behaviour — never a hard error.
              </Paragraph>
              <Paragraph>
                <b>Text chain</b> — <code>AI_TEXT_PRIORITY</code>. Tried in order for query
                translation (Vietnamese → English) and the Agent Search planner. Failure →
                Google Translate / the local planner.
              </Paragraph>
              <Paragraph>
                <b>Vision chain</b> — <code>AI_VISION_PRIORITY</code>. Tried in order for
                grounded video Q&amp;A, the Agent Search VLM frame verifier, and the TRAKE
                verifier. Failure → a non-VLM result with a clear status / retrieval order
                kept.
              </Paragraph>
              <Paragraph type="secondary" style={{ marginBottom: 0 }}>
                Fallover triggers on timeout, rate limit / quota, unknown model, or upstream
                error. API keys stay server-side. Model IDs are never hard-coded — set them
                per provider below and confirm with <b>Test</b>.
              </Paragraph>
            </div>
          ),
        },
      ]}
    />
  );
}

export default function ProvidersTab({ schema, config, form, active, reloadToken }) {
  const { values, secrets, errors, setValue, setSecret } = form;
  const partition = useMemo(() => partitionAiFields(schema), [schema]);

  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [row, setRow] = useState({}); // id -> { testing, result, models, ... }

  async function load() {
    setLoading(true);
    try {
      setStatus(await fetchProviders());
    } catch (err) {
      message.error(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (active) load();
  }, [active, reloadToken]);

  async function runTest(id, mode) {
    setRow((r) => ({ ...r, [id]: { ...r[id], testing: true } }));
    try {
      const result = await testProvider(id, mode);
      setRow((r) => ({ ...r, [id]: { ...r[id], testing: false, result } }));
    } catch (err) {
      setRow((r) => ({ ...r, [id]: { ...r[id], testing: false, result: { ok: false, detail: err.message } } }));
    }
  }

  async function runDiscover(id) {
    setRow((r) => ({ ...r, [id]: { ...r[id], loadingModels: true } }));
    try {
      const res = await discoverModels(id);
      setRow((r) => ({
        ...r,
        [id]: { ...r[id], loadingModels: false, models: res.models || [], modelsError: res.ok ? "" : res.detail },
      }));
    } catch (err) {
      setRow((r) => ({ ...r, [id]: { ...r[id], loadingModels: false, modelsError: err.message } }));
    }
  }

  if (loading || !status || !schema) return <Spin style={{ display: "block", margin: "48px auto" }} />;

  const statusById = Object.fromEntries((status.providers || []).map((p) => [p.id, p]));
  const orderedIds = (status.providers || []).map((p) => p.id).filter((id) => partition.providers[id]);

  function renderFields(specs) {
    return (specs || []).map((spec) => (
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
    ));
  }

  const providerItems = orderedIds.map((id) => {
    const st = statusById[id] || {};
    const rs = row[id] || {};
    return {
      key: id,
      label: (
        <div className="set-provider-head">
          <span>
            <Text strong>{st.label || id}</Text> <code className="set-dim">{id}</code>
          </span>
          <Space size={4} wrap onClick={(e) => e.stopPropagation()}>
            <Tag color={st.enabled ? "blue" : "default"}>{st.enabled ? "enabled" : "off"}</Tag>
            <Tag color={st.configured ? "success" : "default"}>{st.configured ? "key set" : "no key"}</Tag>
            {(st.missing_requirements || []).map((m) => (
              <Tag key={m} color="warning">{m}</Tag>
            ))}
            {st.in_text_chain && <Tag color="geekblue">text #{(st.text_priority_index ?? 0) + 1}</Tag>}
            {st.in_vision_chain && <Tag color="purple">vision #{(st.vision_priority_index ?? 0) + 1}</Tag>}
          </Space>
        </div>
      ),
      children: (
        <>
          {renderFields(partition.providers[id])}
          <div className="set-provider-actions">
            <Space wrap>
              <Button size="small" loading={rs.testing} onClick={() => runTest(id, "text")}>Test text</Button>
              <Button size="small" loading={rs.testing} onClick={() => runTest(id, "vision")}>Test vision</Button>
              <Button size="small" loading={rs.loadingModels} onClick={() => runDiscover(id)}>Discover models</Button>
            </Space>
            {rs.result && (
              <div className={`set-test ${rs.result.ok ? "ok" : "bad"}`}>
                {rs.result.ok
                  ? `ok · ${rs.result.model} · ${rs.result.latency_ms}ms · "${rs.result.sample}"`
                  : `${rs.result.category || "error"} · ${rs.result.detail || ""}`}
              </div>
            )}
            {rs.models && (
              <div className="set-dim set-test">
                {rs.models.length} models{rs.modelsError ? ` · ${rs.modelsError}` : ""}
                {rs.models.length > 0 &&
                  `: ${rs.models.slice(0, 8).join(", ")}${rs.models.length > 8 ? "…" : ""}`}
              </div>
            )}
          </div>
        </>
      ),
    };
  });

  const openProviderKeys = orderedIds.filter((id) => {
    const st = statusById[id] || {};
    return st.enabled || st.configured;
  });

  const inf = partition.inference || {};
  const infSections = [
    ["Agent Search planner (text)", inf.planner],
    ["Agent Search VLM verifier (vision)", inf.vlm],
    ["Deep keyframe search — VQA rerank", inf.kis],
    ["TRAKE verifier (vision)", inf.trake],
    ["Grounded Q&A (vision)", inf.qa],
  ].filter(([, specs]) => specs && specs.length);
  const infCount = infSections.reduce((n, [, specs]) => n + specs.length, 0);
  const inferenceItems = infCount
    ? [
        {
          key: "inference",
          label: `AI inference — which features call the chains, and how (${infCount} settings)`,
          children: infSections.map(([title, specs]) => (
            <div key={title} className="set-inf-section">
              <h4 className="set-inf-title">{title}</h4>
              {renderFields(specs)}
            </div>
          )),
        },
      ]
    : [];

  return (
    <div className="set-providers">
      <ProviderExplainer gatewayEnabled={status.gateway_enabled} />

      <Alert
        type={status.gateway_enabled ? "success" : "info"}
        showIcon
        style={{ marginBottom: 16 }}
        message={
          status.gateway_enabled
            ? "AI gateway is ON."
            : "AI gateway is OFF — enable it below to route translation / Agent / VLM through these providers."
        }
        description={
          <Space direction="vertical" size={2}>
            <span>Text chain: {(status.text_chain || []).join(" → ") || "(empty)"}</span>
            <span>Vision chain: {(status.vision_chain || []).join(" → ") || "(empty)"}</span>
            <span className="set-dim">
              Edits here need Save + Restart before a provider becomes active; Test works right after Save.
            </span>
          </Space>
        }
      />

      <section className="set-group" style={{ marginBottom: 16 }}>
        <h3 className="set-group-title">Gateway</h3>
        <div className="set-group-help">
          Master switch and the two ordered fallback lists (comma-separated provider ids).
        </div>
        {renderFields(partition.gateway)}
      </section>

      <div className="set-dim" style={{ margin: "4px 0 6px", fontSize: 12 }}>
        Providers (click to expand)
      </div>
      <Collapse className="set-provider-list" defaultActiveKey={openProviderKeys} items={providerItems} />

      {inferenceItems.length > 0 && (
        <Collapse style={{ marginTop: 12 }} items={inferenceItems} />
      )}

      {partition.legacy.length > 0 && (
        <Collapse
          style={{ marginTop: 8 }}
          items={[
            {
              key: "legacy",
              label: `Legacy single-provider settings (${partition.legacy.length}) — used only when the gateway is OFF`,
              children: renderFields(partition.legacy),
            },
          ]}
        />
      )}
    </div>
  );
}
