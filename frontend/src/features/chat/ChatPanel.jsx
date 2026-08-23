import { useEffect, useRef, useState } from "react";
import {
  CloseOutlined,
  CopyOutlined,
  EditOutlined,
  ExpandOutlined,
  MessageOutlined,
  SearchOutlined,
  SendOutlined,
  ShrinkOutlined,
  TranslationOutlined,
  VideoCameraOutlined,
} from "@ant-design/icons";
import { translateText } from "../../shared/adapters";
import useDialogFocus from "../../hooks/useDialogFocus";

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
              <div className="ws-msg-text">{m.text}</div>
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

/* ---------- translation tab ---------- */
function TranslationPanel({ onUseInSearch, onUseInChat }) {
  const [dir, setDir] = useState("en-vi");
  const [src, setSrc] = useState("");
  const [override, setOverride] = useState(null);
  const [translated, setTranslated] = useState("");
  const [translating, setTranslating] = useState(false);
  const [editing, setEditing] = useState(false);
  const [editBuf, setEditBuf] = useState("");

  const out = override !== null ? override : translated;
  const showOut = src.trim() !== "";

  useEffect(() => {
    const text = src.trim();
    if (!text || override !== null) {
      setTranslated("");
      setTranslating(false);
      return;
    }

    let cancelled = false;
    setTranslating(true);
    const timer = window.setTimeout(async () => {
      try {
        const result = await translateText(text, dir);
        if (!cancelled) setTranslated(result || "");
      } catch {
        if (!cancelled) setTranslated("");
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
              <span className="ws-demo-badge">{translating ? "LIVE..." : "LIVE"}</span>
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
            <button className="ws-btn small" onClick={() => onUseInSearch(editing ? editBuf : out)} title="Use translated text as search query">
              <SearchOutlined /> Use as query
            </button>
            <button className="ws-btn small" onClick={() => onUseInChat(editing ? editBuf : out)} title="Send translated text to the copilot">
              <MessageOutlined /> Use as prompt
            </button>
          </div>
          <p className="ws-tr-note">Live translation via backend. Original text is always preserved.</p>
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
  onUseInSearch, onUseInChat,
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
      {chatTab === "qa" ? (
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
      ) : (
        <TranslationPanel onUseInSearch={onUseInSearch} onUseInChat={onUseInChat} />
      )}
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
              <div className="ws-msg-text">{m.text}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
