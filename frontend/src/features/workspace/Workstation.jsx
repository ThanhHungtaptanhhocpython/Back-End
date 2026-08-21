import { useCallback, useEffect, useRef, useState } from "react";
import { App as AntApp } from "antd";
import { CloseOutlined, MessageOutlined } from "@ant-design/icons";
import { CARD_W, GAP } from "../../shared/constants";
import { queryTypeFromSearchType } from "../../shared/submissionExport";
import { runSearch as runSearchQuery, askCopilot, probeBackend as probeSearchBackend } from "../../shared/adapters";
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
    params: { topk: 100, clip: true, clipv2: false, imageFile: null },
    status: "running",
    latency: 0,
    results: [],
    total: 0,
  };
}

let chatSeq = 0;

export default function Workstation({ view, onSwitchView }) {
  const { message } = AntApp.useApp();
  const toast = message;

  const [tabs, setTabs] = useState(() => [makeTab()]);
  const [activeKey, setActiveKey] = useState(() => tabs[0]?.key);
  const [focusedId, setFocusedId] = useState(null);
  const [cols, setCols] = useState(5);
  const [kept, setKept] = useState(() => new Map());
  const [reviewItem, setReviewItem] = useState(null);
  const [reviewTabKey, setReviewTabKey] = useState(null);
  const [exportOpen, setExportOpen] = useState(false);
  const [exportItems, setExportItems] = useState([]);
  const [batchOpen, setBatchOpen] = useState(false);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [editingKey, setEditingKey] = useState(null);
  const [backend, setBackend] = useState({ backend: "offline", demo: true, note: "LOCAL MOCK", at: "" });
  const [chatOpen, setChatOpen] = useState(true);
  const [chatWidth, setChatWidth] = useState(400);
  const [chatMsgs, setChatMsgs] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [chatStatus, setChatStatus] = useState("idle");
  const [chatCtxFrame, setChatCtxFrame] = useState(null);
  const [chatFocus, setChatFocus] = useState(false);
  const [chatTab, setChatTab] = useState("qa");

  /* keyboard preferences + transient helpers */
  const [powerUser, setPowerUser] = useState(false);
  const [undoState, setUndoState] = useState(null);
  const undoTimer = useRef(null);
  const invokeFocusStackRef = useRef([]);

  const clock = useClock();

  const searchRef = useRef(null);
  const gridRef = useRef(null);
  const trayRef = useRef(null);
  const composerRef = useRef(null);
  const cardRefs = useRef({});
  const chatSessionRef = useRef(`workspace-${Date.now()}-${Math.random().toString(36).slice(2)}`);

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
    const effectivePivot = pivotFrame || tab?.pivotItem;
    setTabs((prev) => prev.map((t) => (t.key === tab.key ? { ...t, status: "running" } : t)));
    try {
      const res = await runSearchQuery(tab, effectivePivot);
      setTabs((prev) =>
        prev.map((t) => (t.key === tab.key ? { ...t, status: "done", results: res.items, total: res.totalItems, latency: res.latency } : t))
      );
      setBackend({
        backend: res.source === "live" ? "online" : "offline",
        demo: res.source !== "live",
        note: res.source === "live" ? "FASTAPI" : res.source === "fallback" ? "FASTAPI UNAVAILABLE" : "LOCAL MOCK",
        at: new Date().toLocaleTimeString("en-GB", { hour12: false }),
      });
      toast.success(`${res.type} Â· ${res.totalItems} frames Â· ${res.mode} Â· ${res.latency}ms`);
    } catch (error) {
      setTabs((prev) => prev.map((t) => (t.key === tab.key ? { ...t, status: "err" } : t)));
      toast.error(error instanceof Error ? error.message : "Search failed");
    }
  };

  /* debounced auto-run on query/type/param changes (IMAGE is explicit â€” requires a seed) */
  useEffect(() => {
    if (!activeTab || editingKey || activeTab.searchType === "IMAGE") return;
    const id = setTimeout(() => {
      runSearch(activeTab, null);
    }, 420);
    return () => clearTimeout(id);
  }, [activeTab?.query, activeTab?.searchType, activeTab?.params?.topk, activeTab?.params?.clip, activeTab?.params?.clipv2, activeKey]);

  const runActive = () => {
    const tab = tabs.find((t) => t.key === activeKey);
    if (tab) runSearch(tab, tab.pivotItem);
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
        const tab = tabs.find((t) => t.key === activeKey);
        const queryIndex = Math.max(1, tabs.findIndex((t) => t.key === activeKey) + 1);
        n.set(item.id, {
          ...item,
          __submission: {
            key: activeKey,
            queryIndex,
            queryType: queryTypeFromSearchType(tab?.searchType),
          },
        });
        toast.success("Kept in tray â€” Space again to release");
      }
      return n;
    });
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
    setReviewItem(item);
  };

  const closeReview = () => {
    setReviewItem(null);
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
      const result = await askCopilot(text, frames, {
        sessionId: chatSessionRef.current,
        topk: activeTab?.params?.topk ?? 100,
      });
      const reply = {
        id: `c${++chatSeq}`,
        role: "assistant",
        text: result.text,
        frames,
        demo: result.demo,
        mode: result.mode,
        data: result.data,
      };
      setChatMsgs((prev) => [...prev, reply]);
    } catch (error) {
      const errorText = error instanceof Error ? error.message : "Copilot failed";
      setChatMsgs((prev) => [
        ...prev,
        { id: `c${++chatSeq}`, role: "assistant", text: errorText, frames, demo: false, error: true, mode: "ERROR" },
      ]);
      toast.error(errorText);
    } finally {
      setChatStatus("idle");
    }
  };
  const useTranslatedInSearch = (text) => {
    if (!text.trim()) return;
    patchTab({ query: text.trim() });
    runActive();
    toast.info("Query updated from translated text");
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
      const w = Math.min(560, Math.max(320, startW + (startX - ev.clientX)));
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
    setExportItems(Array.from(kept.values()));
    setExportOpen(true);
  };

  const closeExport = () => {
    setExportOpen(false);
    restoreFocusTarget();
  };

  const openExportFromReview = (item) => {
    rememberFocusTarget();
    const tab = tabs.find((t) => t.key === reviewTabKey) || tabs.find((t) => t.key === activeKey);
    const queryIndex = Math.max(1, tabs.findIndex((t) => t.key === tab?.key) + 1);
    setExportItems([{
      ...item,
      __submission: {
        key: tab?.key || activeKey,
        queryIndex,
        queryType: queryTypeFromSearchType(tab?.searchType),
      },
    }]);
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
      el.focus();
      el.scrollIntoView({ block: "nearest", inline: "nearest" });
    }
  }, [focusedId]);

  const registerRef = (id, el) => {
    cardRefs.current[id] = el;
  };

  const keptArray = Array.from(kept.values());

  const ping = useCallback(async () => {
    setBackend((previous) => ({ ...previous, checking: true, at: "" }));
    const result = await probeSearchBackend();
    setBackend({
      ...result,
      checking: false,
      at: new Date().toLocaleTimeString("en-GB", { hour12: false }),
    });
    if (result.backend === "online") toast.success("FastAPI is online");
    else toast.warning(`${result.note} â€” demo search remains available`);
  }, [toast]);

  useEffect(() => {
    void ping();
  }, [ping]);

  return (
    <div className="ws-root">
      <StatusBar
        view={view}
        onSwitchView={onSwitchView}
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
            onUseInSearch={useTranslatedInSearch}
            onUseInChat={useTranslatedInChat}
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
          results={tabs.find((t) => t.key === reviewTabKey)?.results || []}
          isKept={kept.has(reviewItem.id)}
          onClose={closeReview}
          onNavigate={reviewNav}
          onSelect={setReviewItem}
          onToggleKeep={toggleKeep}
          onRemove={removeCurrent}
          onPivot={pivotCurrent}
          onAsk={askAboutFrame}
          onExportThis={openExportFromReview}
        />
      ) : null}

      <ExportModal
        open={exportOpen}
        items={exportItems}
        activeTabResults={activeTab?.results || []}
        tabs={tabs}
        activeKey={activeKey}
        onClose={closeExport}
        toast={toast}
      />
      <BatchQueryModal
        open={batchOpen}
        onClose={() => setBatchOpen(false)}
        toast={toast}
      />
      <ShortcutOverlay open={showShortcuts} powerUser={powerUser} onTogglePowerUser={togglePowerUser} onClose={toggleHelp} />
      <ChatFocus messages={chatMsgs} onClose={closeChatFocus} />

      <div className="ws-ribbon">
        <span className="ws-ribbon-label">Keyboard</span>
        <span className="ws-key"><Keycap>?</Keycap> help</span>
        <span className="ws-key"><Keycap>Tab</Keycap> move</span>
        <span className="ws-key"><Keycap>â†‘â†“â†â†’</Keycap> grid</span>
        <span className="ws-key"><Keycap>â†µ</Keycap> open</span>
        <span className="ws-key"><Keycap>â£</Keycap> keep</span>
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
