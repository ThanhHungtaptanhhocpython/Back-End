import { useCallback, useEffect, useRef, useState } from "react";
import { App as AntApp } from "antd";
import { CloseOutlined, MessageOutlined } from "@ant-design/icons";
import { CARD_W, GAP } from "../../shared/constants";
import { runSearch as runSearchQuery, runAgentSearch as runAgentSearchQuery, askCopilot, askGroundedQa, probeBackend as probeSearchBackend } from "../../shared/adapters";
import { getFramePool } from "../../mocks/searchEngine";
import useClock from "../../hooks/useClock";
import useWorkspaceKeyboard from "../../hooks/useWorkspaceKeyboard";
import Keycap from "../../components/Keycap";
import StatusBar from "./StatusBar";
import ExportModal from "./ExportModal";
import BatchQueryModal from "./BatchQueryModal";
import ShortcutOverlay from "./ShortcutOverlay";
import QueryTabs from "../search/QueryTabs";
import SearchBar from "../search/SearchBar";
import ResultGrid from "../results/ResultGrid";
import QaAnswerPanel from "../results/QaAnswerPanel";
import TemporalStoryboard from "../results/TemporalStoryboard";
import { reindexTemporalSequences } from "../../shared/temporalNormalize";
import { isRunnableTemporalQuery, parseTemporalQuery } from "../../shared/temporalQuery";
import { planTranslationTarget } from "../../shared/translationTarget";
import SelectionTray from "../selection/SelectionTray";
import ReviewOverlay from "../review/ReviewOverlay";
import ChatPanel, { ChatFocus } from "../chat/ChatPanel";

let tabSeq = 0;
function makeTab() {
  tabSeq += 1;
  return {
    key: `q${tabSeq}`,
    label: `Query ${String(tabSeq).padStart(2, "0")}`,
    searchType: "TEXT",
    query: "",
    params: { topk: 100, imageFile: null },
    status: "idle",
    latency: 0,
    results: [],
    sequences: [],
    total: 0,
    meta: null,
    resultSource: null,
    resultMode: null,
  };
}

let chatSeq = 0;

export default function Workstation() {
  const { message } = AntApp.useApp();
  const toast = message;

  const [tabs, setTabs] = useState(() => [makeTab()]);
  const [activeKey, setActiveKey] = useState(() => tabs[0]?.key);
  const [focusedId, setFocusedId] = useState(null);
  const [cols, setCols] = useState(5);
  const [kept, setKept] = useState(() => new Map());
  const [reviewItem, setReviewItem] = useState(null);
  const [reviewTabKey, setReviewTabKey] = useState(null);
  const [reviewReplaceCtx, setReviewReplaceCtx] = useState(null);
  const [exportOpen, setExportOpen] = useState(false);
  const [exportItems, setExportItems] = useState([]);
  const [exportDefaultSource, setExportDefaultSource] = useState("results");
  const [batchOpen, setBatchOpen] = useState(false);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [editingKey, setEditingKey] = useState(null);
  const [backend, setBackend] = useState({ backend: "offline", demo: true, note: "LOCAL MOCK", at: "" });
  const [chatOpen, setChatOpen] = useState(true);
  const [chatWidth, setChatWidth] = useState(500);
  const [chatMsgs, setChatMsgs] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [chatStatus, setChatStatus] = useState("idle");
  const [chatCtxFrame, setChatCtxFrame] = useState(null);
  const [chatFocus, setChatFocus] = useState(false);
  const [chatTab, setChatTab] = useState("qa");
  const [agentSearchMsgs, setAgentSearchMsgs] = useState([]);
  const [agentSearchInput, setAgentSearchInput] = useState("");
  const [agentSearchStatus, setAgentSearchStatus] = useState("idle");

  /* keyboard preferences + transient helpers */
  const [powerUser, setPowerUser] = useState(false);
  const [undoState, setUndoState] = useState(null);
  const undoTimer = useRef(null);
  const invokeFocusStackRef = useRef([]);
  const revealFocusedCardRef = useRef(false);

  const clock = useClock();

  const searchRef = useRef(null);
  const gridRef = useRef(null);
  const trayRef = useRef(null);
  const composerRef = useRef(null);
  const agentComposerRef = useRef(null);
  const cardRefs = useRef({});
  const searchSeqRef = useRef(new Map());

  const activeTab = tabs.find((t) => t.key === activeKey) || tabs[0];
  const focusedItem = activeTab?.results.find((r) => r.id === focusedId) || null;

  /* grid columns measurement */
  useEffect(() => {
    const el = gridRef.current;
    if (!el) return;
    const measure = () => {
      const w = el.clientWidth;
      setCols(Math.max(1, Math.floor((w + GAP) / (CARD_W + GAP))));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  /* ---------- search actions ---------- */
  const runSearch = async (tab, pivotFrame) => {
    const query = String(tab?.query || "").trim();
    if (tab?.searchType !== "IMAGE" && !query) {
      searchSeqRef.current.set(tab.key, (searchSeqRef.current.get(tab.key) || 0) + 1);
      setTabs((prev) => prev.map((t) => (
        t.key === tab.key
          ? { ...t, status: "idle", results: [], sequences: [], total: 0, latency: 0, meta: null, resultSource: null, resultMode: null }
          : t
      )));
      return;
    }
    if (tab?.searchType === "TEMPORAL" && !isRunnableTemporalQuery(parseTemporalQuery(query))) {
      searchSeqRef.current.set(tab.key, (searchSeqRef.current.get(tab.key) || 0) + 1);
      setTabs((prev) => prev.map((t) => (t.key === tab.key ? { ...t, status: "idle", results: [], sequences: [], total: 0, latency: 0 } : t)));
      return;
    }

    const tabKey = tab.key;
    const requestId = (searchSeqRef.current.get(tabKey) || 0) + 1;
    searchSeqRef.current.set(tabKey, requestId);
    const isLatestSearch = () => searchSeqRef.current.get(tabKey) === requestId;
    const effectivePivot = pivotFrame || tab?.pivotItem;

    if (tabKey === activeKey) setFocusedId(null);
    setTabs((prev) => prev.map((t) => (
      t.key === tabKey
        ? { ...t, status: "running", results: [], sequences: [], total: 0, latency: 0, meta: null, resultSource: null, resultMode: null }
        : t
    )));
    try {
      const res = await runSearchQuery(tab, effectivePivot);
      if (!isLatestSearch()) return;

      const isTemporal = res.type === "TEMPORAL" || Array.isArray(res.sequences);
      setTabs((prev) =>
        prev.map((t) =>
          t.key === tabKey
            ? {
                ...t,
                status: "done",
                results: isTemporal ? [] : res.items,
                sequences: isTemporal ? res.sequences || [] : [],
                total: isTemporal ? res.totalItems || (res.sequences || []).length : res.totalItems,
                latency: res.latency,
                meta: res.meta || null,
                resultSource: res.source || null,
                resultMode: res.mode || null,
              }
            : t,
        )
      );
      const qaDemoFallback = res.source === "fallback" && res.type === "QA";
      if (tabKey === activeKey) setFocusedId(isTemporal ? null : res.items?.[0]?.id || null);
      setBackend({
        backend: res.source === "live" || qaDemoFallback ? "online" : "offline",
        demo: res.source !== "live",
        note: res.source === "live" ? "FASTAPI" : qaDemoFallback ? "FASTAPI + QA DEMO FALLBACK" : res.source === "fallback" ? "FASTAPI UNAVAILABLE" : "LOCAL MOCK",
        at: new Date().toLocaleTimeString("en-GB", { hour12: false }),
      });
      toast.success(`${res.type} - ${res.totalItems} frames - ${res.mode} - ${res.latency}ms`);
    } catch (error) {
      if (!isLatestSearch()) return;
      setTabs((prev) => prev.map((t) => (t.key === tabKey ? { ...t, status: "err", meta: null, resultSource: null, resultMode: null } : t)));
      toast.error(error instanceof Error ? error.message : "Search failed");
    }
  };

  /* debounced auto-run on query/type/param changes (IMAGE is explicit - requires a seed) */
  useEffect(() => {
    if (!activeTab || editingKey || activeTab.searchType === "IMAGE" || activeTab.searchType === "AGENT" || activeTab.searchType === "TEMPORAL") return;
    if (!String(activeTab.query || "").trim()) return;
    const id = setTimeout(() => {
      runSearch(activeTab, null);
    }, 420);
    return () => clearTimeout(id);
  }, [activeTab?.query, activeTab?.searchType, activeTab?.params?.topk, activeKey]);

  const runActive = () => {
    const tab = tabs.find((t) => t.key === activeKey);
    if (!tab) return;
    if (tab.searchType === "AGENT") runAgentSearchFromTab(tab);
    else runSearch(tab, tab.pivotItem);
  };


  /* ---------- tab management ---------- */
  const addTab = () => {
    const fresh = makeTab();
    setTabs((prev) => [...prev, fresh]);
    setActiveKey(fresh.key);
    setFocusedId(null);
    toast.info(`New query tab ${fresh.label}`);
  };

  const closeTab = (key) => {
    const idx = tabs.findIndex((t) => t.key === key);
    if (idx === -1) return;
    const next = tabs.filter((t) => t.key !== key);
    if (next.length === 0) {
      const fresh = makeTab();
      setTabs([fresh]);
      setActiveKey(fresh.key);
      setFocusedId(null);
      return;
    }
    let nk = activeKey;
    if (key === activeKey) nk = next[Math.max(0, idx - 1)].key;
    setTabs(next);
    setActiveKey(nk);
    setFocusedId(null);
  };

  const renameTab = (key, label) => {
    setTabs((prev) => prev.map((t) => (t.key === key ? { ...t, label } : t)));
  };

  const patchTab = (patch) => {
    setTabs((prev) => prev.map((t) => (t.key === activeKey ? { ...t, ...patch } : t)));
  };

  /* ---------- frame actions ---------- */
  const toggleKeep = (item) => {
    setKept((prev) => {
      const n = new Map(prev);
      if (n.has(item.id)) {
        n.delete(item.id);
        toast.info("Removed from tray");
      } else {
        n.set(item.id, item);
        toast.success("Kept in tray - Space again to release");
      }
      return n;
    });
  };

  /* Add a captured-frame candidate to the tray, de-duped on video + frame. */
  const captureToTray = (candidate) => {
    const videoKey = candidate?.videoKey;
    const frameIdx = candidate?.submissionFrameId ?? candidate?.backend?.frame_idx;
    let added = false;
    setKept((prev) => {
      const dup = Array.from(prev.values()).some(
        (entry) =>
          entry.id === candidate.id ||
          (entry.videoKey === videoKey &&
            (entry.submissionFrameId ?? entry.backend?.frame_idx) === frameIdx),
      );
      if (dup) return prev;
      const next = new Map(prev);
      next.set(candidate.id, candidate);
      added = true;
      return next;
    });
    if (added) toast.success(`Captured ${candidate.frameName} - added to Selection Tray`);
    else toast.info(`${candidate.frameName} is already in the tray`);
    return added;
  };

  const exclude = (item, tabKey = activeKey) => {
    setTabs((prev) =>
      prev.map((t) =>
        t.key === tabKey
          ? { ...t, results: t.results.filter((r) => r.id !== item.id), total: Math.max(0, t.total - 1) }
          : t
      )
    );
    setKept((prev) => {
      const n = new Map(prev);
      n.delete(item.id);
      return n;
    });
    toast.warning(`Frame ${item.frameName} removed from results`);
  };

  const removeWithUndo = (item, tabKey = activeKey) => {
    const tab = tabs.find((t) => t.key === tabKey);
    const index = tab?.results.findIndex((r) => r.id === item.id) ?? -1;
    const wasKept = kept.has(item.id);
    exclude(item, tabKey);
    if (index >= 0) {
      if (undoTimer.current) clearTimeout(undoTimer.current);
      setUndoState({ item, tabKey, index, wasKept });
      undoTimer.current = setTimeout(() => setUndoState(null), 6000);
    }
    if (focusedId === item.id) {
      const rest = (tab?.results || []).filter((r) => r.id !== item.id);
      setFocusedId(rest.length ? rest[Math.min(index, rest.length - 1)].id : null);
    }
  };

  const undoRemove = () => {
    if (!undoState) return;
    if (undoTimer.current) clearTimeout(undoTimer.current);
    setTabs((prev) =>
      prev.map((t) => {
        if (t.key !== undoState.tabKey) return t;
        if (t.results.some((r) => r.id === undoState.item.id)) return t;
        const insertAt = Math.min(undoState.index, t.results.length);
        const results = [...t.results.slice(0, insertAt), undoState.item, ...t.results.slice(insertAt)];
        return { ...t, results, total: results.length };
      })
    );
    setKept((prev) => {
      const n = new Map(prev);
      if (undoState.wasKept) n.set(undoState.item.id, undoState.item);
      else n.delete(undoState.item.id);
      return n;
    });
    setUndoState(null);
    toast.success(`Frame ${undoState.item.frameName} restored`);
  };

  const pivot = (item) => {
    const fresh = makeTab();
    fresh.searchType = "IMAGE";
    fresh.query = `SIMILAR > ${item.frameName}`;
    setTabs((prev) => [...prev, fresh]);
    setActiveKey(fresh.key);
    setFocusedId(null);
    toast.info(`Image pivot on ${item.frameName}`);
    runSearch(fresh, item);
  };

  const openReview = (item) => {
    rememberFocusTarget();
    setReviewTabKey(activeKey);
    setReviewReplaceCtx(null);
    setReviewItem(item);
  };

  /* Open one storyboard event in the Review player/timeline, carrying the
     bounds a replacement frame must stay inside (same video, between the
     neighbouring events' timestamps). */
  const openTemporalEvent = (sequence, frame) => {
    rememberFocusTarget();
    setReviewTabKey(activeKey);
    const eventIndex = frame.eventIndex;
    const prev = sequence.frames[eventIndex - 2];
    const next = sequence.frames[eventIndex];
    setReviewReplaceCtx({
      tabKey: activeKey,
      sequenceId: sequence.id,
      eventIndex,
      videoKey: sequence.videoKey,
      minTs: prev ? prev.timestamp : Number.NEGATIVE_INFINITY,
      maxTs: next ? next.timestamp : Number.POSITIVE_INFINITY,
    });
    setReviewItem({ ...frame, real: true });
  };

  const handleReplaceEventFrame = (newFrame) => {
    const ctx = reviewReplaceCtx;
    if (!ctx) return;
    const nextId = Number(newFrame?.submissionFrameId ?? newFrame?.backend?.frame_idx);
    if (!Number.isFinite(nextId)) {
      toast.error("That frame has no resolvable frame index — pick another keyframe.");
      return;
    }
    setTabs((prev) =>
      prev.map((t) => {
        if (t.key !== ctx.tabKey) return t;
        const updated = (t.sequences || []).map((seq) => {
          if (seq.id !== ctx.sequenceId) return { ...seq, chosen: false };
          const frames = seq.frames.map((fr) =>
            fr.eventIndex === ctx.eventIndex
              ? {
                  ...fr,
                  submissionFrameId: nextId,
                  globalFrameId: nextId,
                  frameKey: String(newFrame.frameKey ?? nextId),
                  frameName: newFrame.frameName || fr.frameName,
                  timestamp: Number(newFrame.timestamp) || fr.timestamp,
                  timecode: newFrame.timecode || fr.timecode,
                  image: newFrame.image || fr.image,
                  folderKey: newFrame.folderKey || fr.folderKey,
                  unresolved: false,
                }
              : fr,
          );
          const timestamps = frames.map((f) => f.timestamp);
          const orderOk = timestamps.every((v, i) => i === 0 || v >= timestamps[i - 1]);
          const resolved = frames.every((f) => !f.unresolved);
          return { ...seq, frames, timestamps, edited: true, chosen: true, orderOk, sameVideo: true, resolved, valid: orderOk && resolved };
        });
        return { ...t, sequences: reindexTemporalSequences(updated) };
      }),
    );
    toast.success(`Event ${ctx.eventIndex} frame replaced - sequence moved to rank 1`);
  };

  /* "Use this" on a storyboard row: pin exactly one sequence to rank 1 so the
     jittered export wiggles it across the 100-row budget. No frame edit needed. */
  const chooseTemporalSequence = (sequence, chosen) => {
    setTabs((prev) =>
      prev.map((t) => {
        if (t.key !== activeKey) return t;
        const updated = (t.sequences || []).map((seq) => ({
          ...seq,
          chosen: seq.id === sequence.id ? chosen : false,
        }));
        return { ...t, sequences: reindexTemporalSequences(updated) };
      }),
    );
    toast.info(chosen ? `Sequence pinned - Export will wiggle it` : "Sequence unpinned");
  };

  const closeReview = () => {
    setReviewItem(null);
    setReviewReplaceCtx(null);
    restoreFocusTarget();
  };

  const closeChatFocus = () => {
    setChatFocus(false);
    restoreFocusTarget();
  };

  const expandChat = () => {
    rememberFocusTarget();
    setChatFocus(true);
  };

  const reviewNav = (dir) => {
    const tab = tabs.find((t) => t.key === reviewTabKey) || tabs.find((t) => t.key === activeKey);
    const results = tab?.results || [];
    if (!reviewItem) return;
    let seq = results;
    if (!results.some((r) => r.id === reviewItem.id)) {
      const nb = reviewItem.real ? [] : getFramePool().filter((f) => f.videoKey === reviewItem.videoKey).sort((a, b) => a.timestamp - b.timestamp);
      if (nb.length) seq = nb;
    }
    const i = seq.findIndex((x) => x.id === reviewItem.id);
    const ni = i === -1 ? 0 : Math.max(0, Math.min(seq.length - 1, i + dir));
    if (seq[ni] && seq[ni].id !== reviewItem.id) setReviewItem(seq[ni]);
  };

  const keepCurrent = () => {
    if (reviewItem) toggleKeep(reviewItem);
  };

  const removeCurrent = () => {
    if (!reviewItem) return;
    const tab = tabs.find((t) => t.key === reviewTabKey);
    const prevResults = tab?.results || [];
    const before = prevResults.findIndex((r) => r.id === reviewItem.id);
    removeWithUndo(reviewItem, reviewTabKey);
    setFocusedId(null);
    const rest = prevResults.filter((r) => r.id !== reviewItem.id);
    if (rest.length) setReviewItem(rest[Math.min(before, rest.length - 1)]);
    else setReviewItem(null);
  };

  const pivotCurrent = () => {
    if (!reviewItem) return;
    pivot(reviewItem);
    setReviewItem(null);
  };

  const askAboutFrame = (item) => {
    setChatCtxFrame(item);
    setChatOpen(true);
    setChatTab("qa");
    setReviewItem(null);
    setTimeout(() => composerRef.current?.focus(), 50);
  };

  const applyDeepSearchResults = (query, items) => {
    if (!Array.isArray(items) || !items.length) return;
    const fresh = makeTab();
    fresh.label = `Deep Search ${String(tabSeq).padStart(2, "0")}`;
    fresh.searchType = "TEXT";
    fresh.query = query;
    fresh.status = "done";
    fresh.results = items;
    fresh.total = items.length;
    fresh.latency = 0;
    setTabs((prev) => [...prev, fresh]);
    setActiveKey(fresh.key);
    setFocusedId(items[0]?.id || null);
    setBackend({
      backend: "online",
      demo: false,
      note: "FASTAPI DEEP SEARCH",
      at: new Date().toLocaleTimeString("en-GB", { hour12: false }),
    });
    toast.success(`Deep search returned ${items.length} frames`);
  };
  const applyGroundedQaResults = (query, items, meta = null, demo = false) => {
    if (!Array.isArray(items) || !items.length) return;
    const fresh = makeTab();
    fresh.label = `Q&A ${String(tabSeq).padStart(2, "0")}`;
    fresh.searchType = "QA";
    fresh.query = query;
    fresh.status = "done";
    fresh.results = items;
    fresh.total = items.length;
    fresh.latency = 0;
    fresh.meta = meta;
    fresh.resultSource = demo ? "fallback" : "live";
    fresh.resultMode = demo ? "QA DEMO" : "FASTAPI GROUNDED Q&A";
    setTabs((prev) => [...prev, fresh]);
    setActiveKey(fresh.key);
    setFocusedId(items[0]?.id || null);
    setBackend({
      backend: "online",
      demo: false,
      note: "FASTAPI GROUNDED Q&A",
      at: new Date().toLocaleTimeString("en-GB", { hour12: false }),
    });
    toast.success(`Q&A returned ${items.length} source frames`);
  };
  const selectQaCandidate = (candidate) => {
    if (!candidate || !activeKey) return;
    setTabs((prev) => prev.map((tab) => {
      if (tab.key !== activeKey) return tab;
      return {
        ...tab,
        meta: {
          ...(tab.meta || {}),
          answer: candidate.answer,
          confidence: candidate.confidence,
          status: candidate.status,
          reason: candidate.reason,
          supporting_frame_ids: candidate.supporting_frame_ids || [],
          answer_mode: "candidate",
          selected_candidate_id: candidate.candidate_id,
          selected_candidate_video_id: candidate.video_id,
        },
      };
    }));
    const names = new Set([
      ...(Array.isArray(candidate.supporting_frame_names) ? candidate.supporting_frame_names : []),
      candidate.representative_frame_name,
    ].filter(Boolean).map(String));
    const matching = activeTab?.results?.find((frame) => names.has(String(frame?.frameName || "")))
      || activeTab?.results?.find((frame) => String(frame?.videoKey || "") === String(candidate.video_id || ""));
    if (matching) setFocusedId(matching.id);
    toast.info(`Đã chọn ${candidate.video_id}: ${candidate.answer}`);
  };
  const formatAgentSearchMessage = (result, addedLabel) => {
    const queries = Array.isArray(result?.queriesUsed) ? result.queriesUsed : [];
    const routing = result?.routing || result?.plan?.routing || {};
    const queryLines = queries.slice(0, 4).map((query, index) => `${index + 1}. ${query.queryEn || query.query || ""}`);
    const routeLine = ["visual", "ocr", "asr"]
      .map((key) => `${key.toUpperCase()} ${Number(routing[key] || 0).toFixed(1)}`)
      .join(" | ");
    return [
      "Expanded queries:",
      ...(queryLines.length ? queryLines : ["1. " + (result?.plan?.original_query || "")]),
      "",
      "Routing:",
      routeLine,
      "",
      `Results added to ${addedLabel}: ${result?.totalItems || 0} keyframes.`,
    ].join("\n");
  };

  const applyAgentSearchResults = (query, result) => {
    const items = Array.isArray(result?.items) ? result.items : [];
    const fresh = makeTab();
    fresh.searchType = "AGENT";
    fresh.query = query;
    fresh.status = "done";
    fresh.results = items;
    fresh.total = result?.totalItems || items.length;
    fresh.latency = result?.latency || 0;
    setTabs((prev) => [...prev, fresh]);
    setActiveKey(fresh.key);
    setFocusedId(items[0]?.id || null);
    setBackend({
      backend: result?.source === "live" ? "online" : "offline",
      demo: result?.source !== "live",
      note: result?.source === "live" ? "FASTAPI AGENT" : "LOCAL MOCK",
      at: new Date().toLocaleTimeString("en-GB", { hour12: false }),
    });
    toast.success(`Agent Search added ${fresh.total} frames to ${fresh.label}`);
    return fresh.label;
  };

  const runAgentSearchFromTab = async (tab) => {
    const query = String(tab?.query || "").trim();
    if (!query) return;
    try {
      const result = await runAgentSearchQuery(tab);
      applyAgentSearchResults(query, result);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Agent Search failed");
    }
  };

  const sendAgentSearch = async () => {
    const text = agentSearchInput.trim();
    if (!text || agentSearchStatus === "thinking") return;
    const params = activeTab?.params || { topk: 100 };
    const userMsg = { id: `a${++chatSeq}`, role: "user", text, demo: false };
    setAgentSearchMsgs((prev) => [...prev, userMsg]);
    setAgentSearchInput("");
    setAgentSearchStatus("thinking");

    try {
      const result = await runAgentSearchQuery({ searchType: "AGENT", query: text, params });
      const addedLabel = applyAgentSearchResults(text, result);
      const reply = {
        id: `a${++chatSeq}`,
        role: "assistant",
        text: formatAgentSearchMessage(result, addedLabel),
        demo: result?.source !== "live",
        queriesUsed: result?.queriesUsed || [],
        queryTitle: "Expanded queries",
        routing: result?.routing || result?.plan?.routing || {},
        mustHaveChecks: result?.plan?.must_have_checks || result?.plan?.search_plan?.must_have_checks || [],
      };
      setAgentSearchMsgs((prev) => [...prev, reply]);
      setAgentSearchStatus("idle");
    } catch (error) {
      setAgentSearchStatus("err");
      toast.error(error instanceof Error ? error.message : "Agent Search failed");
    }
  };

  /* ---------- copilot actions ---------- */
  const sendChat = async () => {
    const text = chatInput.trim();
    if (!text || chatStatus === "thinking") return;
    const ctx = chatCtxFrame;
    const userMsg = { id: `c${++chatSeq}`, role: "user", text, frames: ctx ? [ctx] : [], demo: false };
    setChatMsgs((prev) => [...prev, userMsg]);
    setChatInput("");
    setChatCtxFrame(null);
    setChatStatus("thinking");
    const frames = userMsg.frames;

    try {
      const result = frames.length ? await askCopilot(text, frames) : await askGroundedQa(text);
      if (Array.isArray(result?.frames) && result.frames.length) {
        if (result?.mode === "agent_search") applyAgentSearchResults(result?.searchQuery || text, { items: result.frames, totalItems: result.frames.length, latency: 0, source: "live" });
        else if (result?.mode === "grounded_qa") {
          const qaFrames = Array.isArray(result?.allFrames) && result.allFrames.length
            ? result.allFrames
            : result.frames;
          applyGroundedQaResults(text, qaFrames, result.meta, result.demo);
        }
        else applyDeepSearchResults(result?.searchQuery || text, result.frames);
      }
      const reply = {
        id: `c${++chatSeq}`,
        role: "assistant",
        text: result?.text || "No response returned.",
        frames: Array.isArray(result?.frames) && result.frames.length ? result.frames : frames,
        demo: Boolean(result?.demo),
        queriesUsed: result?.queriesUsed || [],
        queryTitle: result?.mode === "agent_search" ? "Expanded queries" : "Queries used",
        routing: result?.routing || {},
        mustHaveChecks: result?.searchPlan?.must_have_checks || result?.searchPlan?.search_plan?.must_have_checks || [],
        answerCandidates: result?.meta?.answer_candidates || [],
        allFrames: result?.allFrames || result?.frames || [],
      };
      setChatMsgs((prev) => [...prev, reply]);
      setChatStatus("idle");
    } catch (error) {
      setChatStatus("err");
      toast.error(error instanceof Error ? error.message : "Copilot failed");
    }
  };

  /* Translate panel -> "Use as query" menu. Every destination opens a NEW tab;
     the tab currently on screen and its results are left untouched. */
  const openTranslatedQueryTab = (text, destination) => {
    const plan = planTranslationTarget(text, destination);
    if (!plan) {
      toast.error("Translate a query first, then pick a destination.");
      return;
    }
    const fresh = makeTab();
    fresh.searchType = plan.searchType;
    fresh.query = plan.query;
    fresh.label = `${plan.tabLabel} ${String(tabSeq).padStart(2, "0")}`;
    if (plan.needsSecondEvent) fresh.needsSecondEvent = true;
    setTabs((prev) => [...prev, fresh]);
    setActiveKey(fresh.key);
    setFocusedId(null);
    if (plan.run) {
      runSearch(fresh, null);
    } else {
      toast.info(plan.note);
    }
  };

  const useTranslatedInChat = (text) => {
    setChatTab("qa");
    setChatInput(text.trim());
    setTimeout(() => composerRef.current?.focus(), 30);
  };

  const startChatResize = (e) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = chatWidth;
    const move = (ev) => {
      const w = Math.min(720, Math.max(380, startW + (startX - ev.clientX)));
      setChatWidth(w);
    };
    const up = () => {
      document.removeEventListener("pointermove", move);
      document.removeEventListener("pointerup", up);
    };
    document.addEventListener("pointermove", move);
    document.addEventListener("pointerup", up);
  };

  const removeKept = (id) => {
    setKept((prev) => {
      const n = new Map(prev);
      n.delete(id);
      return n;
    });
  };

  /* ---------- workspace-context actions (keyboard targets) ---------- */
  const rememberFocusTarget = () => {
    const el = document.activeElement;
    invokeFocusStackRef.current.push(el && el !== document.body && el !== document.documentElement ? el : null);
  };

  const restoreFocusTarget = () => {
    const el = invokeFocusStackRef.current.pop();
    if (el && el.isConnected) {
      el.focus({ preventScroll: true });
      return true;
    }
    const tab = tabs.find((t) => t.key === activeKey);
    const id = focusedId || tab?.results[0]?.id;
    const fb = id ? cardRefs.current[id] : null;
    if (fb && fb.isConnected) {
      fb.focus({ preventScroll: true });
      return true;
    }
    return false;
  };

  const openExport = () => {
    rememberFocusTarget();
    const tab = tabs.find((t) => t.key === activeKey);
    setExportItems(Array.from(kept.values()));
    setExportDefaultSource(tab?.results?.length ? "results" : "tray");
    setExportOpen(true);
  };

  const closeExport = () => {
    setExportOpen(false);
    restoreFocusTarget();
  };

  const openExportFromReview = (item) => {
    rememberFocusTarget();
    setExportItems([item]);
    setExportDefaultSource("custom");
    setExportOpen(true);
  };

  const focusQuery = () => searchRef.current?.focus();

  const escWorkspace = () => {
    if (showShortcuts) { setShowShortcuts(false); restoreFocusTarget(); return; }
    if (exportOpen) { setExportOpen(false); restoreFocusTarget(); return; }
    if (focusedId) {
      setFocusedId(null);
      document.activeElement?.blur?.();
    }
  };

  const toggleHelp = () => {
    if (showShortcuts) {
      setShowShortcuts(false);
      restoreFocusTarget();
    } else {
      rememberFocusTarget();
      setShowShortcuts(true);
    }
  };

  const closeHelp = () => {
    setShowShortcuts(false);
    restoreFocusTarget();
  };

  const togglePowerUser = () => setPowerUser((v) => !v);

  /* ---------- keyboard ---------- */
  const moveGridFocus = (code) => {
    const tab = tabs.find((t) => t.key === activeKey);
    const ids = tab?.results.map((r) => r.id) || [];
    if (!ids.length) return;
    let idx = focusedId ? ids.indexOf(focusedId) : -1;
    if (code === "Home") idx = 0;
    else if (code === "End") idx = ids.length - 1;
    else {
      if (idx === -1) idx = 0;
      else if (code === "ArrowRight") idx += 1;
      else if (code === "ArrowLeft") idx -= 1;
      else if (code === "ArrowDown") idx += cols;
      else if (code === "ArrowUp") idx -= cols;
      idx = Math.max(0, Math.min(ids.length - 1, idx));
    }
    revealFocusedCardRef.current = true;
    setFocusedId(ids[idx]);
  };

  const handlersRef = useRef({});
  handlersRef.current = {
    powerUser,
    reviewItem, chatFocus, showShortcuts, exportOpen, focusedItem,
    closeReview, closeChatFocus, closeExport, closeHelp, escWorkspace, openExport, toggleHelp,
    focusQuery, reviewNav, keepCurrent, removeCurrent, pivotCurrent,
    moveGridFocus, openReview, toggleKeep, removeWithUndo, pivot,
  };
  useWorkspaceKeyboard(handlersRef);

  useEffect(() => {
    const el = cardRefs.current[focusedId];
    if (el) {
      el.focus({ preventScroll: true });
      if (revealFocusedCardRef.current) {
        el.scrollIntoView({ block: "nearest", inline: "nearest" });
      }
    }
    revealFocusedCardRef.current = false;
  }, [focusedId]);

  const registerRef = (id, el) => {
    cardRefs.current[id] = el;
  };

  const keptArray = Array.from(kept.values());

  const ping = useCallback(async (showToast = true) => {
    setBackend((previous) => ({ ...previous, checking: true, at: "" }));
    const result = await probeSearchBackend();
    setBackend({
      ...result,
      checking: false,
      at: new Date().toLocaleTimeString("en-GB", { hour12: false }),
    });
    if (showToast) {
      if (result.backend === "online") toast.success("FastAPI is online");
      else toast.warning(`${result.note} - demo search remains available`);
    }
  }, [toast]);

  useEffect(() => {
    void ping(false);
  }, []);

  return (
    <div className="ws-root">
      <StatusBar
        clock={clock}
        backend={backend}
        onPing={ping}
        onShortcuts={toggleHelp}
      />
      <div className="ws-body">
        <QueryTabs
          tabs={tabs}
          activeKey={activeKey}
          onSelect={(k) => { setActiveKey(k); setFocusedId(null); }}
          onAdd={addTab}
          onClose={closeTab}
          onRename={renameTab}
          editingKey={editingKey}
          setEditingKey={setEditingKey}
        />

        <div className="ws-stage" data-chat={chatOpen ? "open" : "closed"} style={{ "--ws-chat-w": `${chatWidth}px` }}>
          <SearchBar
            tab={activeTab}
            onPatch={patchTab}
            onRun={runActive}
            onAgentRun={() => activeTab && runAgentSearchFromTab(activeTab)}
            searchRef={searchRef}
            toast={toast}
          />

          <div className="ws-results" ref={gridRef}>
            <div className="ws-results-head">
              <div className="ws-results-title">Results // {activeTab?.label}</div>
              <div className="ws-results-meta">
                <span>
                  MATCHES <span className="v">{activeTab?.total ?? 0}</span>
                </span>
                <span>
                  LATENCY <span className="v">{activeTab?.latency ?? 0}ms</span>
                </span>
                <span>
                  MODE <span className="mode">{activeTab?.searchType || "TEXT"}</span>
                </span>
                <span>
                  GRID <span className="v">{cols} COL</span>
                </span>
              </div>
            </div>
            <QaAnswerPanel tab={activeTab} onSelectCandidate={selectQaCandidate} />
            <ResultGrid
              tab={activeTab}
              keptMap={kept}
              focusedId={focusedId}
              onFocusItem={setFocusedId}
              onOpen={openReview}
              onToggleKeep={toggleKeep}
              onExclude={removeWithUndo}
              onPivot={pivot}
              registerRef={registerRef}
            />
            {activeTab?.searchType === "TEMPORAL" ? (
              <TemporalStoryboard
                sequences={activeTab?.sequences || []}
                status={activeTab?.status}
                onOpenEvent={openTemporalEvent}
                onChooseSequence={chooseTemporalSequence}
              />
            ) : (
              <ResultGrid
                tab={activeTab}
                keptMap={kept}
                focusedId={focusedId}
                onFocusItem={setFocusedId}
                onOpen={openReview}
                onToggleKeep={toggleKeep}
                onExclude={removeWithUndo}
                onPivot={pivot}
                registerRef={registerRef}
              />
            )}
          </div>

          <ChatPanel
            open={chatOpen}
            width={chatWidth}
            chatTab={chatTab}
            setChatTab={setChatTab}
            messages={chatMsgs}
            status={chatStatus}
            input={chatInput}
            setInput={setChatInput}
            onSend={sendChat}
            ctxFrame={chatCtxFrame}
            onClearCtx={() => setChatCtxFrame(null)}
            composerRef={composerRef}
            onToggleOpen={() => setChatOpen(false)}
            onExpand={expandChat}
            onStartResize={startChatResize}
            onUseInSearch={openTranslatedQueryTab}
            onUseInChat={useTranslatedInChat}
            agentMessages={agentSearchMsgs}
            agentStatus={agentSearchStatus}
            agentInput={agentSearchInput}
            setAgentInput={setAgentSearchInput}
            onAgentSearch={sendAgentSearch}
            agentComposerRef={agentComposerRef}
            onSelectQaCandidate={selectQaCandidate}
          />
        </div>
      </div>

      {!chatOpen ? (
        <button className="ws-chat-reopen" onClick={() => setChatOpen(true)} title="Reopen copilot panel">
          <MessageOutlined /> Ask
        </button>
      ) : null}

      <SelectionTray
        keptItems={keptArray}
        onRemove={removeKept}
        onClear={() => setKept(new Map())}
        onExport={openExport}
        onOpenBatch={() => setBatchOpen(true)}
        onOpen={openReview}
        trayRef={trayRef}
      />

      {undoState ? (
        <div className="ws-undo" role="status" aria-live="polite">
          <span className="ws-undo-text">Removed {undoState.item.frameName}</span>
          <button className="ws-btn small" onClick={undoRemove}>Undo</button>
          <button className="ws-undo-x" onClick={() => setUndoState(null)} title="Dismiss">
            <CloseOutlined />
          </button>
        </div>
      ) : null}

      {reviewItem ? (
        <ReviewOverlay
          item={reviewItem}
          results={
            reviewReplaceCtx
              ? (tabs.find((t) => t.key === reviewTabKey)?.sequences || []).find((s) => s.id === reviewReplaceCtx.sequenceId)?.frames || []
              : tabs.find((t) => t.key === reviewTabKey)?.results || []
          }
          isKept={kept.has(reviewItem.id)}
          onClose={closeReview}
          onNavigate={reviewNav}
          onSelect={setReviewItem}
          onToggleKeep={toggleKeep}
          onRemove={removeCurrent}
          onPivot={pivotCurrent}
          onAsk={askAboutFrame}
          onExportThis={openExportFromReview}
          onCapture={captureToTray}
          replaceCtx={reviewReplaceCtx}
          onReplaceEventFrame={handleReplaceEventFrame}
        />
      ) : null}

      <ExportModal
        open={exportOpen}
        items={exportItems}
        searchItems={activeTab?.results || []}
        sequences={activeTab?.sequences || []}
        keptItems={keptArray}
        tabs={tabs}
        searchType={activeTab?.searchType || "TEXT"}
        defaultSource={exportDefaultSource}
        onClose={closeExport}
        toast={toast}
      />
      <BatchQueryModal open={batchOpen} onClose={() => setBatchOpen(false)} toast={toast} />
      <ShortcutOverlay open={showShortcuts} powerUser={powerUser} onTogglePowerUser={togglePowerUser} onClose={toggleHelp} />
      <ChatFocus open={chatFocus} messages={chatMsgs} onClose={closeChatFocus} />

      <div className="ws-ribbon">
        <span className="ws-ribbon-label">Keyboard</span>
        <span className="ws-key"><Keycap>?</Keycap> help</span>
        <span className="ws-key"><Keycap>Tab</Keycap> move</span>
        <span className="ws-key"><Keycap>Arrows</Keycap> grid</span>
        <span className="ws-key"><Keycap>Enter</Keycap> open</span>
        <span className="ws-key"><Keycap>Space</Keycap> keep</span>
        <span className="ws-key"><Keycap>Del</Keycap> remove</span>
        <span className="ws-key"><Keycap>Esc</Keycap> leave / close</span>
        <span className={`ws-key ${powerUser ? "on" : "off"}`}>
          power: <Keycap>/</Keycap><Keycap>S</Keycap><Keycap>X</Keycap><Keycap>E</Keycap>
        </span>
        <button
          className={`ws-kbd-toggle ${powerUser ? "on" : ""}`}
          onClick={togglePowerUser}
          title={`Power-user shortcuts are ${powerUser ? "on" : "off"} (/ S X E)`}
          aria-label={`Turn power-user shortcuts ${powerUser ? "off" : "on"}`}
          aria-pressed={powerUser}
        >
          <Keycap>PWR</Keycap>
        </button>
      </div>
    </div>
  );
}
