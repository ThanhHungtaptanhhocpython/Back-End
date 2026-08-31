import { useEffect, useState } from "react";
import { Button, Popconfirm, Spin, Table, Tag, message } from "antd";

import { fetchRevisions, restoreRevision } from "../../services/settingsApi.js";

export default function HistoryTab({ active, onRestored }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [restoring, setRestoring] = useState(0);

  async function load() {
    setLoading(true);
    try {
      setData(await fetchRevisions());
    } catch (err) {
      message.error(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (active) load();
  }, [active]);

  async function runRestore(id) {
    setRestoring(id);
    try {
      const res = await restoreRevision(id);
      message.success(`Restored as revision ${res.revision_id}. Restart to apply.`);
      await load();
      onRestored?.(res);
    } catch (err) {
      message.error(err.message);
    } finally {
      setRestoring(0);
    }
  }

  if (loading || !data) return <Spin style={{ display: "block", margin: "48px auto" }} />;

  const columns = [
    { title: "#", dataIndex: "id", width: 60 },
    { title: "When (UTC)", dataIndex: "created_at" },
    { title: "Source", dataIndex: "source", render: (s) => <Tag>{s}</Tag> },
    { title: "Note", dataIndex: "note" },
    { title: "Fields", dataIndex: "field_count", width: 80 },
    { title: "Secrets", dataIndex: "secret_count", width: 80 },
    {
      title: "",
      width: 130,
      render: (_, r) =>
        r.active ? (
          <Tag color="green">active</Tag>
        ) : (
          <Popconfirm
            title={`Restore revision ${r.id}? It becomes a new active revision; a restart applies it.`}
            onConfirm={() => runRestore(r.id)}
          >
            <Button size="small" loading={restoring === r.id}>Restore</Button>
          </Popconfirm>
        ),
    },
  ];

  return (
    <>
      <div className="set-dim" style={{ marginBottom: 8 }}>
        The 10 most recent revisions are kept. Active revision:{" "}
        <b>{data.active_revision_id ?? "—"}</b>
      </div>
      <Table rowKey="id" size="small" pagination={false} columns={columns} dataSource={data.revisions || []} />
    </>
  );
}
