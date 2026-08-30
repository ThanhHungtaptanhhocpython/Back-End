import { useEffect, useRef, useState } from "react";
import {
  ArrowDownOutlined,
  ArrowUpOutlined,
  CloseOutlined,
  PlusOutlined,
  SearchOutlined,
  ThunderboltOutlined,
} from "@ant-design/icons";
import { MIN_TEMPORAL_EVENTS, parseTemporalQuery, serializeTemporalQuery } from "../../shared/temporalQuery";

const MAX_TEMPORAL_EVENTS = 12;

/**
 * Editable temporal query: splits a pasted question into ordered event cards and
 * keeps a separate scene "context" that is folded into every retrieval query but
 * never counts as an event.
 *
 * The editor owns `{ context, events }` locally (so half-typed empty events do
 * not vanish on serialize) and pushes a canonical string back into `tab.query`.
 */
export default function TemporalEditor({ tab, onPatch, onRun }) {
  const [pasteOpen, setPasteOpen] = useState(false);
  const [pasteText, setPasteText] = useState("");
  const [context, setContext] = useState("");
  const [events, setEvents] = useState([""]);
  const lastSerialized = useRef(null);

  // Re-hydrate from `tab.query` only when it changed outside this editor
  // (tab switch, "Split into events", programmatic set).
  useEffect(() => {
    const incoming = tab?.query || "";
    if (incoming === lastSerialized.current) return;
    const parsed = parseTemporalQuery(incoming);
    setContext(parsed.context);
    setEvents(parsed.events.length ? parsed.events : [""]);
    lastSerialized.current = incoming;
  }, [tab?.query]);

  const push = (nextContext, nextEvents) => {
    const serialized = serializeTemporalQuery({ context: nextContext, events: nextEvents });
    lastSerialized.current = serialized;
    setContext(nextContext);
    setEvents(nextEvents.length ? nextEvents : [""]);
    onPatch({ query: serialized });
  };

  const runnableCount = events.filter((event) => String(event || "").trim()).length;
  const runnable = runnableCount >= MIN_TEMPORAL_EVENTS;

  const setEvent = (index, value) => push(context, events.map((event, i) => (i === index ? value : event)));
  const addEvent = () => {
    if (events.length >= MAX_TEMPORAL_EVENTS) return;
    push(context, [...events, ""]);
  };
  const removeEvent = (index) => push(context, events.filter((_, i) => i !== index));
  const moveEvent = (index, delta) => {
    const target = index + delta;
    if (target < 0 || target >= events.length) return;
    const next = events.slice();
    [next[index], next[target]] = [next[target], next[index]];
    push(context, next);
  };

  const applyPaste = () => {
    const parsed = parseTemporalQuery(pasteText);
    push(parsed.context, parsed.events.length ? parsed.events : [""]);
    setPasteOpen(false);
    setPasteText("");
  };

  return (
    <div className="ws-temporal-editor">
      <div className="ws-temporal-head">
        <span className="ws-panel-tag">
          <ThunderboltOutlined /> Temporal events
        </span>
        <button type="button" className="ws-btn small" onClick={() => setPasteOpen((value) => !value)}>
          {pasteOpen ? "Hide paste" : "Paste question"}
        </button>
      </div>

      {pasteOpen ? (
        <div className="ws-temporal-paste">
          <textarea
            rows={5}
            value={pasteText}
            placeholder={"query-p2-8-trake:\nĐoạn video ...\nE1: ...\nE2: ...\nE3: ...\nE4: ..."}
            onChange={(event) => setPasteText(event.target.value)}
          />
          <div className="ws-temporal-paste-actions">
            <button type="button" className="ws-btn small primary" onClick={applyPaste} disabled={!pasteText.trim()}>
              Split into events
            </button>
            <span className="ws-faint">
              Accepts E1/E2..., "Cảnh 1", numbered lists, arrow chains and a query-p&lt;phase&gt;-&lt;id&gt;-trake header.
            </span>
          </div>
        </div>
      ) : null}

      <label className="ws-field-label">Scene context (folded into every event, not an event)</label>
      <input
        className="ws-query"
        value={context}
        placeholder="e.g. Đoạn video nấu ăn một món về nấm"
        onChange={(event) => push(event.target.value, events)}
      />

      <div className="ws-temporal-events">
        {events.map((event, index) => (
          <div className="ws-temporal-event" key={index}>
            <span className="ws-temporal-badge">E{index + 1}</span>
            <textarea
              rows={2}
              value={event}
              placeholder={`Khoảnh khắc sự kiện ${index + 1}`}
              onChange={(e) => setEvent(index, e.target.value)}
            />
            <div className="ws-temporal-event-actions">
              <button type="button" className="ws-btn xsmall" title="Move up" disabled={index === 0} onClick={() => moveEvent(index, -1)}>
                <ArrowUpOutlined />
              </button>
              <button
                type="button"
                className="ws-btn xsmall"
                title="Move down"
                disabled={index === events.length - 1}
                onClick={() => moveEvent(index, 1)}
              >
                <ArrowDownOutlined />
              </button>
              <button
                type="button"
                className="ws-btn xsmall danger"
                title="Delete event"
                disabled={events.length <= 1}
                onClick={() => removeEvent(index)}
              >
                <CloseOutlined />
              </button>
            </div>
          </div>
        ))}
      </div>

      <div className="ws-temporal-foot">
        <button type="button" className="ws-btn small" onClick={addEvent} disabled={events.length >= MAX_TEMPORAL_EVENTS}>
          <PlusOutlined /> Add event
        </button>
        <div className="ws-param">
          <label className="ws-param-label">Top K</label>
          <input
            className="ws-num"
            type="number"
            min={1}
            max={100}
            value={tab?.params?.topk ?? 100}
            onChange={(event) =>
              onPatch({ params: { ...tab.params, topk: Math.min(100, Math.max(1, Number(event.target.value) || 100)) } })
            }
          />
        </div>
        <button type="button" className="ws-btn primary" onClick={onRun} disabled={!runnable || tab?.status === "running"}>
          <SearchOutlined /> {tab?.status === "running" ? "Searching..." : "Run temporal search"}
        </button>
      </div>

      {!runnable ? (
        <div className="ws-temporal-warn" role="status">
          TRAKE cần ít nhất {MIN_TEMPORAL_EVENTS} sự kiện theo thứ tự thời gian. Thêm E2 (hoặc dùng "Paste question" để tự
          tách).
        </div>
      ) : null}
    </div>
  );
}
