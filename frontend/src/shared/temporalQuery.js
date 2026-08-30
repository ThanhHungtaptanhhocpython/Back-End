/**
 * Shared TRAKE query parser.
 *
 * Used by both the interactive Temporal tab editor and the batch runner so a
 * pasted exam question is split into ordered events the same way everywhere.
 *
 * Accepted shapes (the exam set actually uses the first three):
 *   - `E1: ...`, `E 2. ...`, `E3) ...`                     (labels may repeat/skip)
 *   - `Event 1: ...`, `Cảnh 1: ...`, `Scene 1 - ...`, `Sự kiện 1: ...`, `Bước 1: ...`
 *   - numbered lists `1. ...` / `1) ...`                   (needs >= 2 lines)
 *   - bullet lists `- ...` / `* ...` / `• ...`             (needs >= 2 lines)
 *   - a single line joined by `->` or Vietnamese connectors ("sau đó", "tiếp theo", "rồi")
 *   - an optional leading `query-p<phase>-<id>-<type>` header line
 *   - an optional `context:` / `bối cảnh:` line, or any prose before the first event
 *
 * The context is NOT an event: `buildTemporalEventQueries` folds it into every
 * retrieval query so each event keeps the whole-scene framing without inflating
 * the event count.
 */

export const MIN_TEMPORAL_EVENTS = 2;
export const MAX_TEMPORAL_TOPK = 100;

const HEADER_RE = /^query-p(\d+)-(\d+)-(kis|qa|trake)\s*[:.]?\s*(.*)$/i;

const CONTEXT_RE =
  /^(?:context|bối\s*cảnh(?:\s*chung)?|boi\s*canh(?:\s*chung)?|ngữ\s*cảnh|ngu\s*canh|tổng\s*quan|tong\s*quan|overview|scene\s*context)\s*[:\-–]\s*(.+)$/i;

// Ordered so the most explicit label wins. Each entry captures the event body.
// The `E<n>` label accepts a separator (`E1:`, `E1.`, `E1)`) OR just whitespace
// (`E1 text`), which the exam questions use interchangeably.
const EVENT_RES = [
  { kind: "e", re: /^E\s*(\d+)(?:\s*[.:)\-–]\s*|\s+)(.+)$/i },
  {
    kind: "keyword",
    re: /^(?:event|cảnh|canh|scene|sự\s*kiện|su\s*kien|bước|buoc|step|shot|đoạn|doan|khoảnh\s*khắc|khoanh\s*khac)\s*(\d+)\s*[.:)\-–]\s+(.+)$/i,
  },
  { kind: "number", re: /^(\d{1,2})\s*[.):\-–]\s+(.+)$/, min: 2 },
  { kind: "bullet", re: /^[-*•·]\s+(.+)$/, min: 2 },
];

const CONNECTOR_INNER_RE =
  /\s*,?\s*(?:\b(?:rồi|roi|then)\b|(?:sau đó|sau do|tiếp theo|tiep theo|tiếp đến|tiep den|kế đến|ke den))(?:\s+(?:là|la|đến|den))?\s*[:,-]?\s*/i;

function splitConnectors(line) {
  return String(line || "")
    .split(/\s*(?:->|→)\s*/)
    .flatMap((chunk) => chunk.split(CONNECTOR_INNER_RE))
    .map((part) =>
      part
        .replace(/^\s*(?:rồi|roi|sau đó|sau do|tiếp theo|tiep theo|kế đến|ke den|then)\b[\s:,.-]*/i, "")
        .replace(/^\s*(?:cảnh|canh|khung\s*hình|khung\s*hinh|frame|clip|video|đầu\s*tiên|dau\s*tien)\s*(?:này|nay)?\s*[:,-]?\s*/i, "")
        .trim(),
    )
    .filter(Boolean);
}

const LEAD_IN_RE =
  /\s*[,–-]?\s*(?:tìm(?:\s+(?:các|cac))?|liệt\s*kê|liet\s*ke|gồm(?:\s+(?:các|cac))?|bao\s*gồm|bao\s*gom|xác\s*định|xac\s*dinh|find|với)\b[^.?!]*[:：]\s*$/i;

const TRAILING_COLON_CLAUSE_RE = /(?:^|[.;])\s*([^.;:]*[:：])\s*$/;

function cleanContext(text) {
  let out = String(text || "").replace(/\s+/g, " ").trim();
  if (!out) return "";
  if (LEAD_IN_RE.test(out)) {
    out = out.replace(LEAD_IN_RE, "").trim();
  } else if (/[:：]\s*$/.test(out)) {
    // "..., gồm các khoảnh khắc sơ chế:" -> drop the dangling clause.
    const m = out.match(TRAILING_COLON_CLAUSE_RE);
    if (m && m[1] && m[1].length < out.length) out = out.slice(0, m.index).trim();
  }
  return out.replace(/[,:–-]\s*$/, "").trim();
}

function matchEvent(line, kinds) {
  for (const entry of EVENT_RES) {
    if (kinds && !kinds.has(entry.kind)) continue;
    const m = line.match(entry.re);
    if (m) return { kind: entry.kind, body: (m[2] ?? m[1] ?? "").trim() };
  }
  return null;
}

function detectKind(lines) {
  for (const entry of EVENT_RES) {
    const hits = lines.filter((line) => entry.re.test(line)).length;
    if (hits >= (entry.min || 1)) return entry.kind;
  }
  return null;
}

/**
 * Parse raw query text into `{ header, context, events, warnings }`.
 * `events` is always an array of trimmed strings (never empty).
 */
export function parseTemporalQuery(rawText) {
  const text = String(rawText || "");
  const warnings = [];
  const allLines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  let header = null;
  const lines = [...allLines];
  if (lines.length) {
    const hm = lines[0].match(HEADER_RE);
    if (hm) {
      header = { phase: Number(hm[1]), id: Number(hm[2]), type: hm[3].toLowerCase() };
      const rest = (hm[4] || "").trim();
      lines.shift();
      if (rest) lines.unshift(rest);
    }
  }

  const contextParts = [];
  const events = [];

  // Pull explicit `context:` lines out first, wherever they sit.
  const bodyLines = [];
  for (const line of lines) {
    const cm = line.match(CONTEXT_RE);
    if (cm) contextParts.push(cm[1].trim());
    else bodyLines.push(line);
  }

  const kind = detectKind(bodyLines);

  if (kind) {
    const kinds = new Set([kind]);
    let seenEvent = false;
    for (const line of bodyLines) {
      const ev = matchEvent(line, kinds);
      if (ev) {
        seenEvent = true;
        events.push(ev.body);
      } else if (!seenEvent) {
        contextParts.push(line);
      } else if (events.length) {
        // Continuation of the previous event (wrapped line).
        events[events.length - 1] = `${events[events.length - 1]} ${line}`.trim();
      }
    }
  } else if (bodyLines.length) {
    // No labelled events: try a single line joined by connectors.
    const joined = bodyLines[bodyLines.length - 1];
    const parts = splitConnectors(joined);
    if (parts.length >= 2) {
      events.push(...parts);
      for (const line of bodyLines.slice(0, -1)) contextParts.push(line);
    } else if (joined) {
      events.push(joined);
      for (const line of bodyLines.slice(0, -1)) contextParts.push(line);
    }
  }

  const context = cleanContext(contextParts.join(" "));
  const cleanEvents = events.map((ev) => ev.replace(/\s+/g, " ").trim()).filter(Boolean);

  if (cleanEvents.length < MIN_TEMPORAL_EVENTS) {
    warnings.push(
      cleanEvents.length === 0
        ? "Không tách được sự kiện nào. Dùng E1:, E2: ... cho mỗi khoảnh khắc."
        : "Chỉ tìm thấy 1 sự kiện. TRAKE cần ít nhất 2 sự kiện theo thứ tự thời gian.",
    );
  }

  return {
    header,
    context,
    events: cleanEvents.length ? cleanEvents : [String(text).replace(/\s+/g, " ").trim()].filter(Boolean),
    warnings,
  };
}

/** Fold the shared context into every event so retrieval keeps whole-scene framing. */
export function buildTemporalEventQueries(parsed) {
  const context = String(parsed?.context || "").trim();
  const events = Array.isArray(parsed?.events) ? parsed.events : [];
  return events
    .map((ev) => String(ev || "").trim())
    .filter(Boolean)
    .map((ev) => (context ? `${context}\n${ev}` : ev));
}

/** Canonical text form so the editor can round-trip through `tab.query` (a string). */
export function serializeTemporalQuery({ context = "", events = [] } = {}) {
  const lines = [];
  const ctx = String(context || "").trim();
  if (ctx) lines.push(ctx);
  (events || []).forEach((ev, index) => {
    const body = String(ev || "").trim();
    if (body) lines.push(`E${index + 1}: ${body}`);
  });
  return lines.join("\n");
}

/** True when the parsed query has enough structure to run a temporal search. */
export function isRunnableTemporalQuery(parsed) {
  return Array.isArray(parsed?.events) && parsed.events.filter((ev) => String(ev || "").trim()).length >= MIN_TEMPORAL_EVENTS;
}
