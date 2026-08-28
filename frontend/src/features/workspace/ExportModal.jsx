import { useEffect, useMemo, useRef, useState } from "react";
import { CloseOutlined, ExportOutlined } from "@ant-design/icons";
import useDialogFocus from "../../hooks/useDialogFocus";
import { fetchVideoTimeline } from "../../shared/adapters";
import {
  buildSubmissionCsv,
  makeSubmissionZip,
  queryTypeFromSearchType,
  sanitizeQueryFileName,
} from "../../shared/submissionExport";

function safeDownloadName(value, fallback) {
  return (String(value || "").trim() || fallback).replace(/[\\/:*?"<>|]+/g, "_");
}

function previewText(files) {
  if (!files.length) return "No frames selected.";
  return files
    .flatMap((file) => {
      const lines = String(file.content || "").split("\n").filter(Boolean);
      return [
        `submission/${sanitizeQueryFileName(file.name, file.queryType)}`,
        ...lines.slice(0, 5),
        ...(lines.length > 5 ? [`... +${lines.length - 5} more`] : []),
      ];
    })
    .join("\n");
}

function itemPriorityKey(item) {
  return String(item?.id ?? item?.backend?.id ?? `${item?.videoKey || item?.backend?.video_id}-${item?.frameKey || item?.backend?.frame_id || item?.submissionFrameId || item?.id}`);
}

function prioritizeKeptItems(items, keptItems, limit = 100) {
  const keptKeys = new Set((keptItems || []).map(itemPriorityKey));
  const seen = new Set();
  const ordered = [];

  [...(keptItems || []), ...(items || [])].forEach((item) => {
    const key = itemPriorityKey(item);
    if (seen.has(key)) return;
    if (ordered.length >= limit) return;
    if (keptKeys.has(key) || !(keptItems || []).length || !(keptItems || []).some((kept) => itemPriorityKey(kept) === key)) {
      seen.add(key);
      ordered.push(item);
    }
  });

  if (ordered.length >= limit) return ordered;

  (items || []).forEach((item) => {
    const key = itemPriorityKey(item);
    if (seen.has(key) || ordered.length >= limit) return;
    seen.add(key);
    ordered.push(item);
  });

  return ordered;
}

export default function ExportModal({
  open,
  items = [],
  searchItems = [],
  keptItems = [],
  tabs = [],
  searchType = "TEXT",
  defaultSource = "results",
  onClose,
  toast,
}) {
  const inferredType = queryTypeFromSearchType(searchType);
  const [source, setSource] = useState(defaultSource);
  const [queryType, setQueryType] = useState(inferredType);
  const [csvName, setCsvName] = useState("");
  const [zipName, setZipName] = useState("submission.zip");
  const [answer, setAnswer] = useState("");
  const [resolvedCustomItems, setResolvedCustomItems] = useState([]);
  const [hydratingCustomItems, setHydratingCustomItems] = useState(false);
  const closeRef = useRef(null);
  const dialogRef = useDialogFocus(closeRef, open);

  useEffect(() => {
    if (!open) return;
    setSource(defaultSource);
    setQueryType(inferredType);
    setCsvName(`query-p1-1-${inferredType}.csv`);
    setZipName("submission.zip");
    setAnswer("");
    setResolvedCustomItems(items || []);
    setHydratingCustomItems(false);
  }, [open, defaultSource, inferredType, items]);

  useEffect(() => {
    if (!open || source !== "custom") {
      setResolvedCustomItems(items || []);
      setHydratingCustomItems(false);
      return;
    }

    const selected = items || [];
    if (selected.length === 0 || selected.length >= 100 || queryType === "trake") {
      setResolvedCustomItems(selected);
      setHydratingCustomItems(false);
      return;
    }

    const [seed] = selected;
    const videoId = seed?.videoKey || seed?.backend?.video_id;
    const aroundFrameId = seed?.submissionFrameId || seed?.frameKey || seed?.backend?.frame_id || seed?.id;
    if (!videoId) {
      setResolvedCustomItems(selected);
      setHydratingCustomItems(false);
      return;
    }

    let cancelled = false;
    setHydratingCustomItems(true);

    fetchVideoTimeline(videoId, aroundFrameId, 100)
      .then((timeline) => {
        if (cancelled) return;
        const merged = [];
        const seen = new Set();
        const seedAnswer = seed?.answer ?? seed?.backend?.answer;

        [...selected, ...(timeline || [])].forEach((item) => {
          const key = String(item?.id ?? `${item?.videoKey || item?.backend?.video_id}-${item?.frameKey || item?.backend?.frame_id}`);
          if (seen.has(key)) return;
          seen.add(key);
          if (queryType === "qa" && seedAnswer && !item?.answer && !item?.backend?.answer) {
            merged.push({ ...item, answer: seedAnswer });
            return;
          }
          merged.push(item);
        });

        setResolvedCustomItems(merged.slice(0, 100));
      })
      .catch(() => {
        if (!cancelled) setResolvedCustomItems(selected);
      })
      .finally(() => {
        if (!cancelled) setHydratingCustomItems(false);
      });

    return () => {
      cancelled = true;
    };
  }, [open, source, items, queryType]);

  const files = useMemo(() => {
    if (source === "allTabs") {
      return (tabs || [])
        .map((tab, index) => {
          const type = queryTypeFromSearchType(tab?.searchType);
          const rows = prioritizeKeptItems(tab?.results || [], keptItems || [], 100);
          return {
            name: `query-p1-${index + 1}-${type}.csv`,
            queryType: type,
            content: buildSubmissionCsv(rows, type, answer),
            count: rows.length,
          };
        })
        .filter((file) => file.count > 0);
    }

    const selectedItems = source === "results"
      ? prioritizeKeptItems(searchItems || [], keptItems || [], 100)
      : source === "tray"
        ? (keptItems || []).slice(0, 100)
        : source === "custom"
          ? prioritizeKeptItems(resolvedCustomItems || [], keptItems || [], 100)
          : (resolvedCustomItems || []).slice(0, 100);

    return [{
      name: sanitizeQueryFileName(csvName, queryType),
      queryType,
      content: buildSubmissionCsv(selectedItems, queryType, answer),
      count: selectedItems.length,
    }];
  }, [source, tabs, searchItems, keptItems, resolvedCustomItems, csvName, queryType, answer]);

  const rowCount = files.reduce((sum, file) => sum + file.count, 0);
  const preview = previewText(files);
  const hasQa = files.some((file) => file.queryType === "qa");

  if (!open) return null;

  const downloadBlob = (blob, filename) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const ensureRows = () => {
    if (rowCount === 0) {
      toast.warning("Nothing to export");
      return false;
    }
    return true;
  };

  const downloadCsv = () => {
    if (!ensureRows()) return;
    files.forEach((file) => {
      downloadBlob(
        new Blob([file.content], { type: "text/csv;charset=utf-8" }),
        sanitizeQueryFileName(file.name, file.queryType)
      );
    });
    toast.success(`Exported ${files.length} CSV file(s), ${rowCount} row(s)`);
    onClose();
  };

  const downloadZip = () => {
    if (!ensureRows()) return;
    const blob = makeSubmissionZip(files);
    const filename = safeDownloadName(zipName, "submission.zip");
    downloadBlob(blob, filename.toLowerCase().endsWith(".zip") ? filename : `${filename}.zip`);
    toast.success(`Exported ${files.length} CSV file(s), ${rowCount} row(s) as BTC ZIP`);
    onClose();
  };

  return (
    <div className="ws-overlay" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div ref={dialogRef} className="ws-modal" role="dialog" aria-modal="true" aria-label="Export submission" tabIndex={-1}>
        <div className="ws-modal-head">
          <div className="ws-modal-title">
            <ExportOutlined /> Export Submission
          </div>
          <span className="ws-dim" style={{ fontSize: "11px", letterSpacing: "1px" }}>
            {rowCount} ROWS - {files.length} CSV - {hydratingCustomItems ? "LOADING TIMELINE" : rowCount === 0 ? "EMPTY" : "READY"}
          </span>
          <button ref={closeRef} className="ws-modal-close" onClick={onClose} title="Close (Esc)">
            <CloseOutlined />
          </button>
        </div>
        <div className="ws-modal-body">
          <div className="ws-field">
            <label className="ws-field-label">Export Source</label>
            <div className="ws-exp-fmt">
              <button className={`ws-type ${source === "results" ? "active" : ""}`} onClick={() => setSource("results")} disabled={!searchItems.length}>
                Top 100 Search Results ({Math.min(searchItems.length, 100)})
              </button>
              <button className={`ws-type ${source === "tray" ? "active" : ""}`} onClick={() => setSource("tray")} disabled={!keptItems.length}>
                Selection Tray ({keptItems.length})
              </button>
              <button className={`ws-type ${source === "allTabs" ? "active" : ""}`} onClick={() => setSource("allTabs")} disabled={!tabs.some((tab) => tab.results?.length)}>
                All Query Tabs ({tabs.filter((tab) => tab.results?.length).length})
              </button>
              <button className={`ws-type ${source === "custom" ? "active" : ""}`} onClick={() => setSource("custom")} disabled={!items.length}>
                Current Export ({Math.min((resolvedCustomItems || items || []).length, 100)})
              </button>
            </div>
          </div>

          {source !== "allTabs" ? (
            <div className="ws-field">
              <label className="ws-field-label">Query Type</label>
              <div className="ws-exp-fmt">
                {[
                  ["kis", "KIS"],
                  ["qa", "QA"],
                  ["trake", "TRAKE"],
                ].map(([value, label]) => (
                  <button key={value} className={`ws-type ${queryType === value ? "active" : ""}`} onClick={() => setQueryType(value)}>
                    {label}
                  </button>
                ))}
              </div>
            </div>
          ) : null}

          {hasQa ? (
            <div className="ws-field">
              <label className="ws-field-label">Fallback answer for QA (max 100 chars)</label>
              <input value={answer} onChange={(e) => setAnswer(e.target.value.slice(0, 100))} placeholder="Used when result has no answer" />
            </div>
          ) : null}

          {source !== "allTabs" ? (
            <div className="ws-field">
              <label className="ws-field-label">CSV filename inside submission/</label>
              <input value={csvName} onChange={(e) => setCsvName(e.target.value)} placeholder={`query-p1-1-${queryType}.csv`} />
            </div>
          ) : null}
          <div className="ws-field">
            <label className="ws-field-label">ZIP filename</label>
            <input value={zipName} onChange={(e) => setZipName(e.target.value)} placeholder="submission.zip" />
          </div>
          <div className="ws-field">
            <label className="ws-field-label">Preview (folder + BTC CSV format)</label>
            <div className="ws-exp-preview">{preview}</div>
          </div>
          <div className="ws-runbar">
            <button className="ws-btn primary" onClick={downloadZip} disabled={rowCount === 0}>
              <ExportOutlined /> Download ZIP
            </button>
            <button className="ws-btn" onClick={downloadCsv} disabled={rowCount === 0}>
              Download CSV only
            </button>
            <button className="ws-btn" onClick={onClose}>
              Cancel
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
