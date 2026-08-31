import { useCallback, useEffect, useMemo, useState } from "react";
import { Alert, Breadcrumb, Button, Input, List, Modal, Space, Spin, Tag } from "antd";
import {
  FileOutlined,
  FolderFilled,
  FolderOpenOutlined,
  HomeOutlined,
  ReloadOutlined,
} from "@ant-design/icons";

import { browseFs } from "../../services/settingsApi.js";
import { formatBytes } from "./settingsModel.js";

/**
 * Server-side location picker. The management API is loopback-only, so it lists
 * directories on the machine the backend runs on — which is where these paths
 * are read.
 *
 * props: open, title, initialPath, pathKind ("dir" | "file" | "any"),
 *        onPick(path), onClose
 */
export default function FilePickerModal({ open, title, initialPath, pathKind = "any", onPick, onClose }) {
  const allowDir = pathKind !== "file";
  const allowFile = pathKind !== "dir";

  const [state, setState] = useState(null); // { sep, path, parent, entries, roots, home, repo_root, error }
  const [loading, setLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState("");
  const [manual, setManual] = useState("");
  const [loadErr, setLoadErr] = useState("");

  const load = useCallback(
    async (path) => {
      setLoading(true);
      setLoadErr("");
      setSelectedFile("");
      try {
        const res = await browseFs(path, { dirsOnly: pathKind === "dir" });
        setState(res);
        setManual(res.path || "");
      } catch (err) {
        setLoadErr(err.message || "could not read that location");
      } finally {
        setLoading(false);
      }
    },
    [pathKind],
  );

  useEffect(() => {
    if (open) load(initialPath || "");
  }, [open, initialPath, load]);

  const crumbs = useMemo(() => {
    if (!state?.path) return [];
    const sep = state.sep || "/";
    const parts = state.path.split(sep).filter(Boolean);
    const out = [];
    let acc = state.path.startsWith(sep) ? sep : "";
    parts.forEach((part, i) => {
      acc = i === 0 && !acc ? part + (sep === "\\" ? sep : "") : acc + part;
      out.push({ label: part || sep, path: acc });
      acc += sep;
    });
    return out;
  }, [state]);

  const atRoots = !state?.path;
  const chosen = selectedFile || (allowDir ? state?.path : "");

  return (
    <Modal
      open={open}
      title={title || "Choose a location"}
      width={720}
      onCancel={onClose}
      footer={[
        <Button key="cancel" onClick={onClose}>Cancel</Button>,
        allowDir && state?.path && (
          <Button key="dir" onClick={() => onPick(state.path)}>
            Use this folder
          </Button>
        ),
        <Button key="ok" type="primary" disabled={!chosen} onClick={() => onPick(chosen)}>
          Select
        </Button>,
      ]}
    >
      <Space style={{ marginBottom: 10 }} wrap>
        <Button size="small" icon={<HomeOutlined />} onClick={() => load(state?.home || "")}>
          Home
        </Button>
        {state?.repo_root && (
          <Button size="small" onClick={() => load(state.repo_root)}>Repo</Button>
        )}
        {(state?.roots || []).map((r) => (
          <Button size="small" key={r.path} onClick={() => load(r.path)}>{r.name}</Button>
        ))}
        <Button
          size="small"
          icon={<ReloadOutlined />}
          onClick={() => load(state?.path || "")}
          disabled={atRoots}
        />
      </Space>

      <Space.Compact style={{ width: "100%", marginBottom: 10 }}>
        <Input
          value={manual}
          placeholder="Type or paste a path, then Go"
          onChange={(e) => setManual(e.target.value)}
          onPressEnter={() => load(manual)}
        />
        <Button onClick={() => load(manual)}>Go</Button>
      </Space.Compact>

      {!atRoots && (
        <Breadcrumb
          style={{ marginBottom: 8 }}
          items={[
            { title: <a onClick={() => load("")}><FolderOpenOutlined /></a> },
            ...crumbs.map((c) => ({ title: <a onClick={() => load(c.path)}>{c.label}</a> })),
          ]}
        />
      )}

      {loadErr && <Alert type="error" showIcon style={{ marginBottom: 8 }} message={loadErr} />}
      {state?.error && <Alert type="warning" showIcon style={{ marginBottom: 8 }} message={state.error} />}

      <div className="set-picker-list">
        {loading ? (
          <Spin style={{ display: "block", margin: "40px auto" }} />
        ) : atRoots ? (
          <List
            size="small"
            dataSource={state?.roots || []}
            renderItem={(r) => (
              <List.Item className="set-picker-row" onClick={() => load(r.path)}>
                <FolderFilled className="set-dim" /> <span>{r.name}</span>
              </List.Item>
            )}
          />
        ) : (
          <List
            size="small"
            locale={{ emptyText: "empty folder" }}
            dataSource={state?.entries || []}
            renderItem={(e) => {
              const isSel = !e.is_dir && selectedFile === e.path;
              return (
                <List.Item
                  className={`set-picker-row${isSel ? " selected" : ""}`}
                  onClick={() => (e.is_dir ? load(e.path) : allowFile && setSelectedFile(e.path))}
                >
                  {e.is_dir ? <FolderFilled style={{ color: "#eab308" }} /> : <FileOutlined className="set-dim" />}
                  <span className="set-picker-name">{e.name}</span>
                  {!e.is_dir && e.size != null && <Tag className="set-dim">{formatBytes(e.size)}</Tag>}
                </List.Item>
              );
            }}
          />
        )}
      </div>

      <div className="set-dim" style={{ marginTop: 8, fontSize: 12 }}>
        Selected: <code>{chosen || "(none)"}</code>
      </div>
    </Modal>
  );
}
