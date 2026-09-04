import { useCallback, useEffect, useRef, useState } from "react";
import { Alert, Button, Space, Spin, Tag, Typography, message } from "antd";
import { ReloadOutlined } from "@ant-design/icons";

import { fetchJinaReadiness } from "../../services/settingsApi.js";

const { Paragraph, Text } = Typography;

const STATUS_ALERT = { ok: "success", warn: "warning", miss: "error" };
const STATUS_LABEL = { ok: "OK", warn: "WARNING", miss: "MISSING" };

/**
 * "Can this machine run Jina CLIP v2?" — the GPU / model / index readiness
 * report from GET /settings/jina/readiness, mirrored by
 * `python scripts/check_jina_setup.py`.
 */
export default function JinaReadinessCard({ active, reloadToken }) {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const aliveRef = useRef(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const next = await fetchJinaReadiness();
      if (aliveRef.current) setReport(next);
    } catch (err) {
      if (aliveRef.current) message.error(err.message);
    } finally {
      if (aliveRef.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    aliveRef.current = true;
    if (active) load();
    return () => {
      aliveRef.current = false;
    };
  }, [active, reloadToken, load]);

  if (!active) return null;
  if (loading && !report) return <Spin style={{ display: "block", margin: "24px auto" }} />;
  if (!report) return null;

  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
        <Text strong>Jina CLIP v2 readiness</Text>
        <Tag color={report.ok ? "success" : "error"}>{report.ok ? "READY" : "NOT READY"}</Tag>
        <Tag color="default">active backend: {report.active_backend}</Tag>
        {report.active_backend !== "jina_clip_v2" && (
          <Text type="secondary" style={{ fontSize: 12 }}>
            (Jina is not the active backend right now)
          </Text>
        )}
        <span style={{ flex: 1 }} />
        <Button size="small" icon={<ReloadOutlined />} loading={loading} onClick={load}>
          Recheck
        </Button>
      </div>

      <Space direction="vertical" size={8} style={{ width: "100%" }}>
        {(report.checks || []).map((c) => (
          <Alert
            key={c.id}
            type={STATUS_ALERT[c.status] || "info"}
            showIcon
            message={
              <span>
                <Tag color={STATUS_ALERT[c.status] || "default"} style={{ marginInlineEnd: 8 }}>
                  {STATUS_LABEL[c.status] || c.status}
                </Tag>
                <b>{c.label}</b> — {c.summary}
              </span>
            }
            description={
              (c.detail || c.fix) && (
                <div style={{ marginTop: 4 }}>
                  {c.detail && <div className="set-dim">{c.detail}</div>}
                  {c.fix && (
                    <Paragraph
                      code
                      copyable
                      style={{ marginTop: 6, marginBottom: 0, whiteSpace: "pre-wrap", fontSize: 12 }}
                    >
                      {c.fix}
                    </Paragraph>
                  )}
                </div>
              )
            }
          />
        ))}
      </Space>
    </div>
  );
}
