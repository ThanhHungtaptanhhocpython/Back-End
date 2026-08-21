function firstDefined(...values) {
  return values.find((value) => value !== undefined && value !== null && value !== "");
}

function normalizeVideoName(value) {
  return String(value || "unknown-video").trim().replace(/\.(mp4|mov|avi|mkv|webm)$/i, "");
}

function videoNameForItem(item) {
  const rawVideo = item?.videoKey ?? item?.backend?.video_name ?? item?.backend?.video_id ?? item?.video_id ?? item?.videoName;
  const video = normalizeVideoName(rawVideo);
  const folder = String(item?.folderKey ?? item?.backend?.folder_key ?? item?.backend?.split ?? item?.split ?? "").trim();
  if (/^V\d+/i.test(video)) {
    if (/^L\d+/i.test(folder)) {
      return normalizeVideoName(`${folder}_${video}`);
    }
    const m = folder.match(/l(\d+)/i);
    if (m) {
      return `L${m[1]}_${video}`;
    }
  }
  return video;
}

function normalizeFrameId(item) {
  const raw = firstDefined(
    item?.submissionFrameId,
    item?.submission_frame_id,
    item?.backend?.submission_frame_id,
    item?.backend?.frame_idx,
    item?.frameKey,
    item?.frame_key,
    item?.backend?.frame_key,
    item?.frame_id,
    item?.backend?.frame_id,
    item?.globalFrameId,
    item?.global_frame_id,
    item?.backend?.global_frame_id,
    item?.id
  );
  const parsed = Number.parseInt(String(raw).replace(/[^\d-]/g, ""), 10);
  return Number.isFinite(parsed) ? String(parsed) : String(raw || "").trim();
}

function csvCell(value) {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export function sanitizeQueryFileName(name, queryType = "kis") {
  const cleaned = String(name || "").trim().replace(/[\\/:*?"<>|]+/g, "_");
  const base = cleaned || `query-1-${queryType.toLowerCase()}`;
  return base.toLowerCase().endsWith(".csv") ? base : `${base}.csv`;
}

export function queryTypeFromSearchType(searchType) {
  const type = String(searchType || "").toUpperCase();
  if (type === "QA" || type === "Q&A SEARCH") return "qa";
  if (type === "TEMPORAL" || type === "TEMPORAL SEARCH") return "trake";
  return "kis";
}

export function buildSubmissionCsv(items, queryType, answer = "") {
  const type = String(queryType || "kis").toLowerCase();
  const rows = (items || []).slice(0, 100);

  if (type === "qa") {
    return rows
      .map((item) => [
        videoNameForItem(item),
        normalizeFrameId(item),
        csvCell(String(item.answer ?? answer).slice(0, 100)),
      ].join(","))
      .join("\n");
  }

  if (type === "trake") {
    if (rows.length === 0) return "";
    const firstVideo = videoNameForItem(rows[0]);
    return [firstVideo, ...rows.map(normalizeFrameId)].join(",");
  }

  return rows
    .map((item) => [
      videoNameForItem(item),
      normalizeFrameId(item),
    ].join(","))
    .join("\n");
}

function crc32(text) {
  const bytes = new TextEncoder().encode(text);
  let crc = -1;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
    }
  }
  return (crc ^ -1) >>> 0;
}

function dosDateTime(date = new Date()) {
  const dosTime = (date.getHours() << 11) | (date.getMinutes() << 5) | Math.floor(date.getSeconds() / 2);
  const dosDate = ((date.getFullYear() - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate();
  return { dosDate, dosTime };
}

function pushU16(out, value) {
  out.push(value & 0xff, (value >>> 8) & 0xff);
}

function pushU32(out, value) {
  out.push(value & 0xff, (value >>> 8) & 0xff, (value >>> 16) & 0xff, (value >>> 24) & 0xff);
}

function pushBytes(out, bytes) {
  for (const byte of bytes) out.push(byte);
}

function normalizeZipFiles(files) {
  const normalized = (files || [])
    .map((file) => ({
      name: sanitizeQueryFileName(file?.name, file?.queryType),
      content: String(file?.content ?? ""),
    }))
    .filter((file) => file.name);

  return normalized.length > 0 ? normalized : [{ name: "query-1-kis.csv", content: "" }];
}

export function makeSubmissionZip(files) {
  const encoder = new TextEncoder();
  const { dosDate, dosTime } = dosDateTime();
  const local = [];
  const central = [];
  const zipFiles = normalizeZipFiles(files);

  zipFiles.forEach((file) => {
    // Write directly to root of zip (NO 'submission/' prefix)
    const path = file.name;
    const nameBytes = encoder.encode(path);
    const contentBytes = encoder.encode(file.content);
    const checksum = crc32(file.content);
    const localOffset = local.length;

    pushU32(local, 0x04034b50);
    pushU16(local, 20);
    pushU16(local, 0x0800);
    pushU16(local, 0);
    pushU16(local, dosTime);
    pushU16(local, dosDate);
    pushU32(local, checksum);
    pushU32(local, contentBytes.length);
    pushU32(local, contentBytes.length);
    pushU16(local, nameBytes.length);
    pushU16(local, 0);
    pushBytes(local, nameBytes);
    pushBytes(local, contentBytes);

    pushU32(central, 0x02014b50);
    pushU16(central, 20);
    pushU16(central, 20);
    pushU16(central, 0x0800);
    pushU16(central, 0);
    pushU16(central, dosTime);
    pushU16(central, dosDate);
    pushU32(central, checksum);
    pushU32(central, contentBytes.length);
    pushU32(central, contentBytes.length);
    pushU16(central, nameBytes.length);
    pushU16(central, 0);
    pushU16(central, 0);
    pushU16(central, 0);
    pushU16(central, 0);
    pushU32(central, 0);
    pushU32(central, localOffset);
    pushBytes(central, nameBytes);
  });

  const end = [];
  pushU32(end, 0x06054b50);
  pushU16(end, 0);
  pushU16(end, 0);
  pushU16(end, zipFiles.length);
  pushU16(end, zipFiles.length);
  pushU32(end, central.length);
  pushU32(end, local.length);
  pushU16(end, 0);

  return new Blob([new Uint8Array(local), new Uint8Array(central), new Uint8Array(end)], {
    type: "application/zip",
  });
}

export function makeSingleFileSubmissionZip(csvFileName, csvContent, queryType = "kis") {
  return makeSubmissionZip([{ name: csvFileName, content: csvContent, queryType }]);
}
