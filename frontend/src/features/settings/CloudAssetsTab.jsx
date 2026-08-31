import { useEffect, useState } from "react";
import { Alert, Button, Descriptions, Popconfirm, Space, Spin, Table, Tag, message } from "antd";

import {
  clearCloudCache,
  fetchCloudManifest,
  fetchCloudStatus,
  syncCloud,
  testCloud,
} from "../../services/settingsApi.js";
import { formatBytes } from "./settingsModel.js";

export default function CloudAssetsTab({ active }) {
  const [status, setStatus] = useState(null);
  const [manifest, setManifest] = useState(null);
  const [probe, setProbe] = useState(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState("");

  async function loadStatus() {
    setLoading(true);
    try {
      setStatus(await fetchCloudStatus());
    } catch (err) {
      message.error(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (active) loadStatus();
  }, [active]);

  async function runTest() {
    setBusy("test");
    try {
      setProbe(await testCloud());
    } catch (err) {
      setProbe({ ok: false, detail: err.message });
    } finally {
      setBusy("");
    }
  }

  async function loadManifest(refresh) {
    setBusy("manifest");
    try {
      setManifest(await fetchCloudManifest(refresh));
    } catch (err) {
      message.error(err.message);
    } finally {
      setBusy("");
    }
  }

  async function runSync() {
    setBusy("sync");
    try {
      const report = await syncCloud([]);
      message[report.ok ? "success" : "warning"](
        `Sync ${report.ok ? "complete" : "finished with issues"} · version ${report.version}` +
          (report.promoted ? " · promoted" : ""),
      );
      await loadManifest(false);
      await loadStatus();
    } catch (err) {
      message.error(err.message);
    } finally {
      setBusy("");
    }
  }

  async function runClear(scope) {
    setBusy("clear");
    try {
      const res = await clearCloudCache(scope);
      message.success(`Cleared ${scope} · freed ${formatBytes(res.data?.freed_bytes)}`);
      await loadStatus();
      setManifest(null);
    } catch (err) {
      message.error(err.message);
    } finally {
      setBusy("");
    }
  }

  if (loading || !status) return <Spin style={{ display: "block", margin: "48px auto" }} />;

  if (!status.active) {
    return (
      <Alert
        type="info"
        showIcon
        message="Cloud assets are disabled"
        description="Set CLOUD_ASSETS_ENABLED and CLOUD_ASSETS_PROVIDER (azure_blob or s3_compatible) in the Configuration tab, add the credentials, then restart."
      />
    );
  }

  const artColumns = [
    { title: "Artifact", dataIndex: "name" },
    { title: "Container", dataIndex: "container" },
    { title: "Key", dataIndex: "key", render: (k) => <code className="set-dim">{k}</code> },
    { title: "Size", dataIndex: "size", render: formatBytes },
    {
      title: "Cache",
      render: (_, r) =>
        r.cached ? (
          <Tag color={r.verified ? "success" : "warning"}>{r.verified ? "verified" : "cached"}</Tag>
        ) : (
          <Tag color="default">missing</Tag>
        ),
    },
  ];

  return (
    <div className="set-cloud">
      <Descriptions bordered size="small" column={2} style={{ marginBottom: 16 }}>
        <Descriptions.Item label="Provider">{status.provider}</Descriptions.Item>
        <Descriptions.Item label="Manifest key">{status.manifest_key}</Descriptions.Item>
        <Descriptions.Item label="Azure SDK">{String(status.sdk?.azure_blob)}</Descriptions.Item>
        <Descriptions.Item label="S3 SDK (boto3)">{String(status.sdk?.s3_compatible)}</Descriptions.Item>
        <Descriptions.Item label="Artifact cache">
          {formatBytes(status.artifact_cache?.usage_bytes)} · current{" "}
          {status.artifact_cache?.current_version || "—"}
        </Descriptions.Item>
        <Descriptions.Item label="Keyframe cache">
          {formatBytes(status.keyframe_cache?.usage_bytes)} / {formatBytes(status.keyframe_cache?.max_bytes)} ·{" "}
          {status.keyframe_cache?.entries} files
        </Descriptions.Item>
      </Descriptions>

      <Space wrap style={{ marginBottom: 16 }}>
        <Button loading={busy === "test"} onClick={runTest}>Test connection</Button>
        <Button loading={busy === "manifest"} onClick={() => loadManifest(true)}>Load manifest</Button>
        <Popconfirm title="Download and checksum-verify all artifacts?" onConfirm={runSync}>
          <Button type="primary" loading={busy === "sync"}>Sync artifacts</Button>
        </Popconfirm>
        <Popconfirm title="Clear the synced-artifact cache?" onConfirm={() => runClear("artifacts")}>
          <Button danger loading={busy === "clear"}>Clear artifact cache</Button>
        </Popconfirm>
        <Popconfirm title="Clear the on-demand keyframe cache?" onConfirm={() => runClear("keyframes")}>
          <Button danger loading={busy === "clear"}>Clear keyframe cache</Button>
        </Popconfirm>
      </Space>

      {probe && (
        <Alert
          style={{ marginBottom: 16 }}
          type={probe.ok ? "success" : "error"}
          showIcon
          message={probe.ok ? "Connection OK" : "Connection failed"}
          description={
            probe.ok
              ? `containers: ${(probe.containers || []).join(", ") || "—"} · manifest ${
                  probe.manifest_present ? `v${probe.manifest_version}` : "not found"
                }`
              : probe.detail || "unknown error"
          }
        />
      )}

      {manifest && (
        <>
          <div className="set-dim" style={{ marginBottom: 8 }}>
            manifest version <b>{manifest.version}</b> · current synced{" "}
            <b>{manifest.current_version || "none"}</b>
          </div>
          <Table
            rowKey="name"
            size="small"
            pagination={false}
            columns={artColumns}
            dataSource={manifest.artifacts || []}
          />
        </>
      )}
    </div>
  );
}
