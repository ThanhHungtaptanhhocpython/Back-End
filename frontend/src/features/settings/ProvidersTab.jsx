import { useEffect, useState } from "react";
import { Alert, Button, Space, Spin, Table, Tag, Typography, message } from "antd";

import { discoverModels, fetchProviders, testProvider } from "../../services/settingsApi.js";

const { Text } = Typography;

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
