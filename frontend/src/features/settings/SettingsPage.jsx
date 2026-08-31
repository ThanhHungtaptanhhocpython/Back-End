import { useEffect, useState } from "react";
import { Alert, Button, Result, Tabs, Typography } from "antd";
import { ArrowLeftOutlined } from "@ant-design/icons";

import { fetchConfig, fetchSchema } from "../../services/settingsApi.js";
import ConfigTab from "./ConfigTab.jsx";
import ProvidersTab from "./ProvidersTab.jsx";
import CloudAssetsTab from "./CloudAssetsTab.jsx";
import HistoryTab from "./HistoryTab.jsx";
import RestartControl from "./RestartControl.jsx";
import "./settings.css";

const { Title } = Typography;

export default function SettingsPage({ onClose }) {
  const [schema, setSchema] = useState(null);
  const [config, setConfig] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [tab, setTab] = useState("config");
  const [pendingRevision, setPendingRevision] = useState(null);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    Promise.all([fetchSchema(), fetchConfig()])
      .then(([s, c]) => {
        if (!alive) return;
        setSchema(s);
        setConfig(c);
        setError(null);
      })
      .catch((err) => alive && setError(err))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [reloadKey]);

  function afterMutation(res) {
    if (res?.revision_id) setPendingRevision(res.revision_id);
    setReloadKey((k) => k + 1);
  }

  return (
    <div className="set-page">
      <header className="set-header">
        <div className="set-header-left">
          <Button type="text" icon={<ArrowLeftOutlined />} onClick={onClose}>
            Workstation
          </Button>
          <Title level={3} style={{ margin: 0 }}>Settings</Title>
          {config?.store?.enabled === false && (
            <Alert
              type="warning"
              showIcon
              banner
              message="Runtime config store is disabled — editing .env directly. Saving is unavailable."
            />
          )}
        </div>
        <RestartControl pendingRevision={pendingRevision} />
      </header>

      {config?.store?.enabled &&
        config.store.env_imported &&
        (config.store.revision_count ?? 0) <= 1 && (
          <Alert
            type="info"
            showIcon
            closable
            style={{ marginBottom: 12 }}
            message="These values were imported from your .env on first run."
            description="Edit them here — the SQLite store now takes precedence over .env. Save creates a revision; Restart applies it. History lets you roll back."
          />
        )}

      {error ? (
        <Result
          status={error.kind === "forbidden" ? "403" : "error"}
          title="Could not load settings"
          subTitle={
            error.kind === "forbidden"
              ? "The management API only accepts loopback clients and localhost origins."
              : error.message
          }
          extra={<Button onClick={() => setReloadKey((k) => k + 1)}>Retry</Button>}
        />
      ) : (
        <Tabs
          activeKey={tab}
          onChange={setTab}
          className="set-tabs"
          items={[
            {
              key: "config",
              label: "Configuration",
              children: (
                <ConfigTab
                  key={reloadKey}
                  schema={schema}
                  config={config}
                  loading={loading}
                  onSaved={afterMutation}
                />
              ),
            },
            {
              key: "providers",
              label: "AI Providers",
              children: <ProvidersTab active={tab === "providers"} />,
            },
            {
              key: "cloud",
              label: "Cloud Assets",
              children: <CloudAssetsTab active={tab === "cloud"} />,
            },
            {
              key: "history",
              label: "History",
              children: <HistoryTab active={tab === "history"} onRestored={afterMutation} />,
            },
          ]}
        />
      )}
    </div>
  );
}
