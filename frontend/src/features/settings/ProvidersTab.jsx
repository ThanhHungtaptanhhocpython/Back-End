import { useEffect, useState } from "react";
import { Alert, Button, Collapse, Space, Spin, Table, Tag, Typography, message } from "antd";

import { discoverModels, fetchProviders, testProvider } from "../../services/settingsApi.js";

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
                <b>Text chain</b> — <code>AI_TEXT_PRIORITY</code>. Tried in order for:
              </Paragraph>
              <ul className="set-explainer-list">
                <li>
                  <b>Query translation</b> (<code>/users/translate</code>): Vietnamese → English
                  for the search box. Order: Google Translate → Text chain → keep your original
                  text (honest failure, never a silent echo).
                </li>
                <li>
                  <b>Agent Search planner</b>: turns a vague description into rich English scene
                  queries + a visual checklist before retrieval. Failure → the deterministic
                  local planner.
                </li>
              </ul>

              <Paragraph>
                <b>Vision chain</b> — <code>AI_VISION_PRIORITY</code>. Tried in order for:
              </Paragraph>
              <ul className="set-explainer-list">
                <li>
                  <b>Grounded video Q&amp;A</b>: reads the retrieved keyframes to answer a
                  question with a confidence score. Failure → a non-VLM “uncertain” result
                  with a clear status.
                </li>
                <li>
                  <b>Agent Search VLM verifier</b>: looks at the top candidate frames and
                  re-scores / reranks them for visual match. Failure → retrieval order kept.
                </li>
                <li>
                  <b>TRAKE verifier</b>: checks ordered-event sequences frame by frame.
                </li>
              </ul>

              <Paragraph type="secondary" style={{ marginBottom: 0 }}>
                Fallover triggers on timeout, rate limit / quota, unknown model, or upstream
                error. API keys stay entirely server-side — this screen only shows
                configured / not-configured. Model IDs are never hard-coded: set them per
                provider and confirm with <b>Test</b>. Free tiers (e.g. NVIDIA NIM) are fine
                for prototyping but rate-limited.
              </Paragraph>

              <Paragraph style={{ marginTop: 12, marginBottom: 0 }}>
                <b>To enable:</b> in <i>Configuration → AI</i>, turn on a provider, add its API
                key + text/vision model, put its id in <code>AI_TEXT_PRIORITY</code> /
                <code> AI_VISION_PRIORITY</code>, set <code>AI_GATEWAY_ENABLED = true</code>,
                then Save &amp; Restart.
              </Paragraph>
            </div>
          ),
        },
      ]}
    />
  );
}

export default function ProvidersTab({ active }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [row, setRow] = useState({}); // id -> { testing, result, models }

  async function load() {
    setLoading(true);
    try {
      setData(await fetchProviders());
    } catch (err) {
      message.error(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (active) load();
  }, [active]);

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
      setRow((r) => ({ ...r, [id]: { ...r[id], loadingModels: false, models: res.models || [], modelsError: res.ok ? "" : res.detail } }));
    } catch (err) {
      setRow((r) => ({ ...r, [id]: { ...r[id], loadingModels: false, modelsError: err.message } }));
    }
  }

  if (loading || !data) return <Spin style={{ display: "block", margin: "48px auto" }} />;

  const columns = [
    {
      title: "Provider",
      dataIndex: "label",
      render: (label, r) => (
        <div>
          <div><Text strong>{label}</Text></div>
          <code className="set-dim">{r.id}</code>
        </div>
      ),
    },
    {
      title: "State",
      render: (_, r) => (
        <Space size={4} wrap>
          <Tag color={r.enabled ? "blue" : "default"}>{r.enabled ? "enabled" : "disabled"}</Tag>
          <Tag color={r.configured ? "success" : "error"}>{r.configured ? "key set" : "no key"}</Tag>
          {r.missing_requirements?.map((m) => <Tag key={m} color="warning">{m}</Tag>)}
          {r.in_text_chain && <Tag color="geekblue">text #{(r.text_priority_index ?? 0) + 1}</Tag>}
          {r.in_vision_chain && <Tag color="purple">vision #{(r.vision_priority_index ?? 0) + 1}</Tag>}
        </Space>
      ),
    },
    {
      title: "Models",
      render: (_, r) => (
        <div>
          <div>text: <code className="set-dim">{r.text_model || "—"}</code></div>
          <div>vision: <code className="set-dim">{r.vision_model || "—"}</code></div>
        </div>
      ),
    },
    {
      title: "Test",
      render: (_, r) => {
        const state = row[r.id] || {};
        return (
          <div>
            <Space>
              <Button size="small" loading={state.testing} onClick={() => runTest(r.id, "text")}>
                Text
              </Button>
              <Button size="small" loading={state.testing} onClick={() => runTest(r.id, "vision")}>
                Vision
              </Button>
              <Button size="small" loading={state.loadingModels} onClick={() => runDiscover(r.id)}>
                Models
              </Button>
            </Space>
            {state.result && (
              <div className={`set-test ${state.result.ok ? "ok" : "bad"}`}>
                {state.result.ok
                  ? `ok · ${state.result.model} · ${state.result.latency_ms}ms · "${state.result.sample}"`
                  : `${state.result.category || "error"} · ${state.result.detail || ""}`}
              </div>
            )}
            {state.models && (
              <div className="set-dim set-test">
                {state.models.length} models{state.modelsError ? ` · ${state.modelsError}` : ""}
                {state.models.length > 0 && `: ${state.models.slice(0, 6).join(", ")}${state.models.length > 6 ? "…" : ""}`}
              </div>
            )}
          </div>
        );
      },
    },
  ];

  return (
    <div>
      <ProviderExplainer gatewayEnabled={data.gateway_enabled} />
      <Alert
        type={data.gateway_enabled ? "success" : "info"}
        showIcon
        style={{ marginBottom: 16 }}
        message={
          data.gateway_enabled
            ? "AI gateway is ON — translation / Agent planner use the Text chain; Q&A / VLM use the Vision chain."
            : "AI gateway is OFF — set AI_GATEWAY_ENABLED in the Configuration tab to use these chains."
        }
        description={
          <Space direction="vertical" size={2}>
            <span>Text chain: {(data.text_chain || []).join(" → ") || "(empty)"}</span>
            <span>Vision chain: {(data.vision_chain || []).join(" → ") || "(empty)"}</span>
            <span className="set-dim">
              Priority order is edited via AI_TEXT_PRIORITY / AI_VISION_PRIORITY in Configuration.
            </span>
          </Space>
        }
      />
      <div style={{ marginBottom: 12 }}>
        <Button onClick={load}>Refresh</Button>
      </div>
      <Table rowKey="id" size="small" pagination={false} columns={columns} dataSource={data.providers || []} />
    </div>
  );
}
