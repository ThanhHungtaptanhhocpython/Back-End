import { useEffect, useRef, useState } from "react";
import { Dropdown } from "antd";
import {
  CloseOutlined,
  CopyOutlined,
  DownOutlined,
  EditOutlined,
  ExpandOutlined,
  MessageOutlined,
  SearchOutlined,
  SendOutlined,
  ShrinkOutlined,
  TranslationOutlined,
  VideoCameraOutlined,
} from "@ant-design/icons";
import { translateTextDetailed } from "../../services/translateService";
import { canTargetTranslation, TRANSLATION_TARGETS } from "../../shared/translationTarget";
import useDialogFocus from "../../hooks/useDialogFocus";

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // clipboard unavailable - no-op
  }
}

function QueryList({ queries, title = "Queries used" }) {
  if (!Array.isArray(queries) || !queries.length) return null;
  return (
    <div className="ws-query-list">
      <div className="ws-query-list-title">{title}</div>
      {queries.map((item, index) => {
        const query = String(item?.query || "").trim();
        const queryEn = String(item?.queryEn || item?.query_en || query).trim();
        const kind = String(item?.kind || "query").toUpperCase();
        return (
          <div className="ws-query-item" key={`${kind}-${query}-${index}`}>
            <div className="ws-query-kind">{kind}</div>
            <div className="ws-query-lines">
              {query ? (
                <div className="ws-query-line">
                  <span className="ws-query-lang">VI</span>
                  <span className="ws-query-text">{query}</span>
                  <button className="ws-query-copy" onClick={() => copyText(query)} title="Copy Vietnamese query">
                    <CopyOutlined />
                  </button>
                </div>
              ) : null}
              {queryEn ? (
                <div className="ws-query-line en">
                  <span className="ws-query-lang">EN</span>
                  <span className="ws-query-text">{queryEn}</span>
                  <button className="ws-query-copy" onClick={() => copyText(queryEn)} title="Copy English query">
                    <CopyOutlined />
                  </button>
                </div>
              ) : null}
            </div>
          </div>
        );
      })}
    </div>
  );
}
function inlineMessageParts(text, timestamp, lineKey) {
  const tokens = String(text || "").split(/(\[[^\]]+\]\(https?:\/\/[^)\s]+\)|\*\*[^*]+\*\*)/g);
  return tokens.filter(Boolean).map((token, index) => {
    const link = token.match(/^\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)$/);
    if (link) {
      let href = link[2];
      if (Number.isFinite(timestamp) && /(?:youtube\.com\/watch|youtu\.be\/)/i.test(href)) {
        const separator = href.includes("?") ? "&" : "?";
        href = /[?&]t=/.test(href) ? href : href + separator + "t=" + Math.floor(timestamp) + "s";
      }
      return <a key={lineKey + "-link-" + index} href={href} target="_blank" rel="noreferrer">{link[1]}</a>;
    }
    const strong = token.match(/^\*\*([^*]+)\*\*$/);
    if (strong) return <strong key={lineKey + "-strong-" + index}>{strong[1]}</strong>;
    return <span key={lineKey + "-text-" + index}>{token}</span>;
  });
}

function RoutingSummary({ routing }) {
  if (!routing || typeof routing !== "object") return null;
  const parts = ["visual", "ocr", "asr"].map((key) => ({ key, value: Number(routing[key] || 0) }));
  if (!parts.some((part) => part.value > 0)) return null;
  return (
    <div className="ws-routing">
      {parts.map((part) => (
        <div className="ws-routing-row" key={part.key}>
          <span>{part.key.toUpperCase()}</span>
          <div className="ws-routing-track"><i style={{ width: `${Math.round(part.value * 100)}%` }} /></div>
          <strong>{part.value.toFixed(1)}</strong>
        </div>
      ))}
    </div>
  );
}


function SearchChecklist({ checks, title = "Must-have checks" }) {
  if (!Array.isArray(checks) || !checks.length) return null;
  return (
    <div className="ws-checklist">
      <div className="ws-query-list-title">{title}</div>
      {checks.map((check, index) => (
        <div className="ws-check-row" key={`${check?.id || check?.label || index}-${index}`}>
          <span className="ws-check-index">{String(index + 1).padStart(2, "0")}</span>
          <span className="ws-check-label">{String(check?.label || check?.query_en || check || "")}</span>
        </div>
      ))}
    </div>
  );
}
function InteractiveMessage({ text }) {
  let latestTimestamp = null;
  return (
    <div className="ws-msg-text">
      {String(text || "").split(/\r?\n/).map((rawLine, index) => {
        const secondsMatch = rawLine.match(/\(([0-9]+(?:\.[0-9]+)?)\s*gi(?:â|a)y\)/i);
        if (secondsMatch) latestTimestamp = Number(secondsMatch[1]);
        const heading = rawLine.match(/^#{1,6}\s+(.+)$/);
        const line = heading ? heading[1] : rawLine;
        return (
          <div key={"message-line-" + index} className={heading ? "ws-msg-heading" : "ws-msg-line"}>
            {line ? inlineMessageParts(line, latestTimestamp, index) : <br />}
          </div>
        );
      })}
    </div>
  );
}
/* ---------- Q&A tab ---------- */
function QaTab({ messages, status, input, setInput, onSend, ctxFrame, onClearCtx, composerRef }) {
  const listRef = useRef(null);
  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages.length, status]);

  return (
    <div className="ws-qa">
      {messages.length === 0 ? (
        <div className="ws-qa-empty">
          <MessageOutlined style={{ fontSize: 26, color: "#94a3b8" }} />
          <div>Ask about a frame or the index.</div>
          <div className="ws-qa-empty-sub">Use "Ask" inside a frame review, or type here. Live backend answers are used when available.</div>
        </div>
      ) : (
        <div className="ws-qa-list" ref={listRef}>
          {messages.map((m) => (
            <div key={m.id} className={`ws-msg ${m.role}`}>
              {m.frames && m.frames.length ? (
                <div className="ws-msg-ctx">
                  <VideoCameraOutlined /> grounded on {m.frames.map((f) => f.frameName).join(", ")}
                </div>
              ) : null}
              {m.role === "assistant" ? <span className="ws-demo-badge">{m.demo ? "DEMO" : "LIVE"}</span> : null}
              <InteractiveMessage text={m.text} />
              <RoutingSummary routing={m.routing} />
              <SearchChecklist checks={m.mustHaveChecks} />
              <QueryList queries={m.queriesUsed} title={m.queryTitle || "Queries used"} />
            </div>
          ))}
          {status === "thinking" ? (
            <div className="ws-msg assistant">
              <span className="ws-demo-badge">LIVE...</span>
              <div className="ws-typing"><i /><i /><i /></div>
            </div>
          ) : null}
          {status === "err" ? (
            <div className="ws-qa-error">
              Copilot failed. <button className="ws-btn small" onClick={onSend}>Retry</button>
            </div>
          ) : null}
        </div>
      )}

      <div className="ws-qa-composer">
        {ctxFrame ? (
          <div className="ws-qa-ctxchip">
            <VideoCameraOutlined /> Grounding on {ctxFrame.frameName}
            <button onClick={onClearCtx} title="Clear grounding context"><CloseOutlined /></button>
          </div>
        ) : null}
        <textarea
          ref={composerRef}
          className="ws-qa-input"
          rows={2}
          placeholder="Ask about a frame or the video index..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
              e.preventDefault();
              onSend();
            }
          }}
        />
        <div className="ws-qa-bar">
          <span className="ws-qa-hint">Enter to send - Shift+Enter for newline</span>
          <button className="ws-btn small primary" onClick={onSend} disabled={!input.trim() || status === "thinking"}>
            <SendOutlined /> Send
          </button>
        </div>
      </div>
    </div>
  );
}

/* ---------- Agent Search tab ---------- */
function AgentSearchTab({ messages, status, input, setInput, onSend, composerRef }) {
  const listRef = useRef(null);
  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages.length, status]);

  return (
    <div className="ws-qa ws-agent-search">
      {messages.length === 0 ? (
        <div className="ws-qa-empty">
          <SearchOutlined style={{ fontSize: 26, color: "#94a3b8" }} />
          <div>Describe what you need to find.</div>
          <div className="ws-qa-empty-sub">The coordinator expands the query, routes visual/OCR/ASR, and adds keyframes to the main grid.</div>
        </div>
      ) : (
        <div className="ws-qa-list" ref={listRef}>
          {messages.map((m) => (
            <div key={m.id} className={`ws-msg ${m.role}`}>
              {m.role === "assistant" ? <span className="ws-demo-badge">{m.demo ? "DEMO" : "AGENT"}</span> : null}
              <InteractiveMessage text={m.text} />
              <RoutingSummary routing={m.routing} />
              <SearchChecklist checks={m.mustHaveChecks} />
              <QueryList queries={m.queriesUsed} title={m.queryTitle || "Expanded queries"} />
            </div>
          ))}
          {status === "thinking" ? (
            <div className="ws-msg assistant">
              <span className="ws-demo-badge">AGENT...</span>
              <div className="ws-typing"><i /><i /><i /></div>
            </div>
          ) : null}
          {status === "err" ? (
            <div className="ws-qa-error">
              Agent Search failed. <button className="ws-btn small" onClick={onSend}>Retry</button>
            </div>
          ) : null}
        </div>
      )}

      <div className="ws-qa-composer">
        <textarea
          ref={composerRef}
          className="ws-qa-input"
          rows={3}
          placeholder="Find frames where..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
              e.preventDefault();
              onSend();
            }
          }}
        />
        <div className="ws-qa-bar">
          <span className="ws-qa-hint">Enter to run - Shift+Enter for newline</span>
          <button className="ws-btn small primary" onClick={onSend} disabled={!input.trim() || status === "thinking"}>
            <SearchOutlined /> Agent Search
          </button>
        </div>
      </div>
    </div>
  );
}

/* ---------- translation tab ---------- */
function TranslationPanel({ onUseInSearch, onUseInChat }) {
  const [dir, setDir] = useState("en-vi");
  const [src, setSrc] = useState("");
  const [override, setOverride] = useState(null);
  const [translated, setTranslated] = useState("");
  const [translating, setTranslating] = useState(false);
  const [translationLive, setTranslationLive] = useState(false);
  const [translationProvider, setTranslationProvider] = useState("none");
  const [translationStatus, setTranslationStatus] = useState("");
  const [editing, setEditing] = useState(false);
  const [editBuf, setEditBuf] = useState("");

  const out = override !== null ? override : translated;
  const showOut = src.trim() !== "";
  const canUseAsQuery = canTargetTranslation({ text: out, live: translationLive, edited: override !== null });

  const translationBadge = translating
    ? "LIVE..."
    : override !== null
      ? "EDITED"
      : translationLive
        ? "LIVE"
        : translationStatus === "backend_unreachable"
          ? "OFFLINE"
          : "UNAVAILABLE";

  const translationNote =
    translationLive || override !== null
      ? "Original text is always preserved."
      : translationStatus === "backend_unreachable"
        ? "Translation service could not be reached. The original text was kept and cannot be used as a translated search query."
        : "Translation is unavailable. The original text was kept and cannot be used as a translated search query.";

  useEffect(() => {
    const text = src.trim();
    if (!text || override !== null) {
      setTranslated("");
      setTranslationLive(false);
      setTranslationProvider("none");
      setTranslationStatus("");
      setTranslating(false);
      return;
    }

    let cancelled = false;
    setTranslating(true);
    const timer = window.setTimeout(async () => {
      try {
        const result = await translateTextDetailed(text, dir);
        if (!cancelled) {
          setTranslated(result?.text || "");
          setTranslationLive(Boolean(result?.live));
          setTranslationProvider(result?.provider || "none");
          setTranslationStatus(result?.status || "");
        }
      } catch {
        if (!cancelled) {
          setTranslated("");
          setTranslationProvider("unavailable");
          setTranslationStatus("backend_unreachable");
        }
      } finally {
        if (!cancelled) setTranslating(false);
      }
    }, 350);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [src, dir, override]);

  const setDirection = (d) => {
    setDir(d);
    setOverride(null);
    setEditing(false);
  };

  const copy = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // clipboard unavailable - no-op
    }
  };

  return (
    <div className="ws-tr">
      <div className="ws-tr-dir">
        <button className={dir === "en-vi" ? "active" : ""} onClick={() => setDirection("en-vi")} title="Translate from English to Vietnamese">
          English to Vietnamese
        </button>
        <button className={dir === "vi-en" ? "active" : ""} onClick={() => setDirection("vi-en")} title="Translate from Vietnamese to English">
          Vietnamese to English
        </button>
      </div>

      <label className="ws-param-label" htmlFor="ws-tr-src">Text to translate</label>
      <textarea
        id="ws-tr-src"
        className="ws-tr-src"
        placeholder={dir === "en-vi" ? "Type English text..." : "Nhap tieng Viet..."}
        value={src}
        onChange={(e) => {
          setSrc(e.target.value);
          setOverride(null);
          setEditing(false);
        }}
      />

      {showOut ? (
        <div className="ws-tr-out">
          <div className="ws-tr-orig">
            <div className="ws-tr-head"><span>Original - {dir === "en-vi" ? "English" : "Vietnamese"}</span></div>
            <div className="ws-tr-box">{src}</div>
          </div>
          <div className="ws-tr-trans">
            <div className="ws-tr-head">
              <span>Translated - {dir === "en-vi" ? "Vietnamese" : "English"}</span>
              <span className="ws-demo-badge">
                {translating ? "LIVE..." : translationLive ? `LIVE · ${translationProvider.toUpperCase()}` : override !== null ? "EDITED" : "UNAVAILABLE"}
              </span>
              <span className="ws-demo-badge">{translationBadge}</span>
            </div>
            {editing ? (
              <textarea
                className="ws-tr-edit"
                value={editBuf}
                autoFocus
                onChange={(e) => setEditBuf(e.target.value)}
              />
            ) : (
              <div className="ws-tr-box">{translating && !out ? "Translating..." : out}</div>
            )}
          </div>

          <div className="ws-tr-actions">
            {editing ? (
              <button className="ws-btn small primary" onClick={() => { setOverride(editBuf); setEditing(false); }}>
                Save edit
              </button>
            ) : (
              <button className="ws-btn small" onClick={() => { setEditBuf(out); setEditing(true); }} title="Edit the translated text">
                <EditOutlined /> Edit translation
              </button>
            )}
            <button className="ws-btn small" onClick={() => copy(editing ? editBuf : out)} title="Copy translated text">
              <CopyOutlined /> Copy
            </button>
            <Dropdown
              trigger={["click"]}
              disabled={!canUseAsQuery}
              menu={{
                items: TRANSLATION_TARGETS.map((t) => ({ key: t.value, label: t.label })),
                onClick: ({ key }) => onUseInSearch(editing ? editBuf : out, key),
              }}
            >
              <button
                className="ws-btn small"
                disabled={!canUseAsQuery}
                title={canUseAsQuery ? "Open the translated text in a new Text, Q&A, or Temporal tab" : "Translation is unavailable. Edit it into the target language before searching."}
              >
                <SearchOutlined /> Use as query <DownOutlined />
              </button>
            </Dropdown>
            <button className="ws-btn small" onClick={() => onUseInChat(editing ? editBuf : out)} title="Send translated text to the copilot">
              <MessageOutlined /> Use as prompt
            </button>
          </div>
          <p className="ws-tr-note">{translationNote}</p>
        </div>
      ) : (
        <div className="ws-tr-empty">Enter text above to see a clearly-labelled live translation preview. The original text is never silently replaced.</div>
      )}
    </div>
  );
}

/* ---------- panel shell ---------- */
export default function ChatPanel({
  open, width, chatTab, setChatTab, messages, status, input, setInput,
  onSend, ctxFrame, onClearCtx, composerRef, onToggleOpen, onExpand, onStartResize,
  onUseInSearch, onUseInChat, agentMessages = [], agentStatus = "idle", agentInput = "", setAgentInput = () => {}, onAgentSearch = () => {}, agentComposerRef,
}) {
  if (!open) return null;
  return (
    <div className="ws-chat" style={{ width }}>
      <div className="ws-chat-resize" onPointerDown={onStartResize} title="Drag to resize" />
      <div className="ws-chat-head">
        <div className="ws-chat-tabs">
          <button className={chatTab === "qa" ? "active" : ""} onClick={() => setChatTab("qa")}>
            <MessageOutlined /> Q&A
          </button>
          <button className={chatTab === "agent" ? "active" : ""} onClick={() => setChatTab("agent")}>
            <SearchOutlined /> Agent Search
          </button>
          <button className={chatTab === "tr" ? "active" : ""} onClick={() => setChatTab("tr")}>
            <TranslationOutlined /> Translate
          </button>
        </div>
        <div className="ws-chat-head-actions">
          <button className="ws-icon-btn" onClick={onExpand} title="Focused reading mode (expand conversation)">
            <ExpandOutlined />
          </button>
          <button className="ws-icon-btn" onClick={onToggleOpen} title="Collapse panel">
            <ShrinkOutlined />
          </button>
        </div>
      </div>
      {/* Keep both panels mounted: switching tabs must not discard a translation
          that is being edited or a request that is still in flight. */}
      <div className="ws-chat-pane" hidden={chatTab !== "qa"}>
        <QaTab
          messages={messages}
          status={status}
          input={input}
          setInput={setInput}
          onSend={onSend}
          ctxFrame={ctxFrame}
          onClearCtx={onClearCtx}
          composerRef={composerRef}
        />
      </div>
      <div className="ws-chat-pane" hidden={chatTab !== "agent"}>
        <AgentSearchTab
          messages={agentMessages}
          status={agentStatus}
          input={agentInput}
          setInput={setAgentInput}
          onSend={onAgentSearch}
          composerRef={agentComposerRef}
        />
      </div>
      <div className="ws-chat-pane" hidden={chatTab !== "tr"}>
        <TranslationPanel onUseInSearch={onUseInSearch} onUseInChat={onUseInChat} />
      </div>
      <div className="ws-chat-foot">Copilot uses the backend when available and falls back to mock replies only on transport failure.</div>
    </div>
  );
}

/* ---------- focused reading mode ---------- */
export function ChatFocus({ open, messages, onClose }) {
  const closeRef = useRef(null);
  const dialogRef = useDialogFocus(closeRef, open);
  if (!open || !messages.length) return null;
  return (
    <div className="ws-overlay chat-focus" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div ref={dialogRef} className="ws-chat-focus" role="dialog" aria-modal="true" aria-label="Focused chat reading" tabIndex={-1}>
        <div className="ws-modal-head">
          <div className="ws-modal-title">
            <MessageOutlined /> Focused reading
          </div>
          <button ref={closeRef} className="ws-modal-close" onClick={onClose} title="Close (Esc)">
            <CloseOutlined />
          </button>
        </div>
        <div className="ws-chat-focus-body">
          {messages.map((m) => (
            <div key={m.id} className={`ws-msg focus ${m.role}`}>
              {m.frames && m.frames.length ? (
                <div className="ws-msg-ctx">
                  <VideoCameraOutlined /> grounded on {m.frames.map((f) => f.frameName).join(", ")}
                </div>
              ) : null}
              {m.role === "assistant" ? <span className="ws-demo-badge">{m.demo ? "DEMO" : "LIVE"}</span> : null}
              <InteractiveMessage text={m.text} />
              <RoutingSummary routing={m.routing} />
              <SearchChecklist checks={m.mustHaveChecks} />
              <QueryList queries={m.queriesUsed} title={m.queryTitle || "Queries used"} />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

