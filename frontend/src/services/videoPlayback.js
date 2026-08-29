import { VIDEO_LINKS } from "../config/videoLinks.js";

function firstDefined(...values) {
  return values.find((value) => value !== undefined && value !== null && value !== "");
}

function parseJsonMap(value) {
  if (!value || typeof value !== "string") return {};
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function runtimeVideoLinks() {
  const envMap = parseJsonMap(import.meta.env?.VITE_VIDEO_LINKS_JSON);
  let storageMap = {};
  try {
    storageMap = parseJsonMap(globalThis.localStorage?.getItem("aic.videoLinks"));
  } catch {
    storageMap = {};
  }
  return { ...VIDEO_LINKS, ...envMap, ...storageMap };
}

export function resolveVideoUrl(item) {
  if (!item) return "";
  const raw = item.backend && typeof item.backend === "object" ? item.backend : {};
  const mappedLinks = runtimeVideoLinks();
  const found = firstDefined(
    item.link,
    raw.youtube_url,
    raw.youtubeUrl,
    raw.media_info?.watch_url,
    raw.mediaInfo?.watchUrl,
    raw.video_metadata?.watch_url,
    raw.videoMetadata?.watchUrl,
    raw.video_url,
    raw.videoUrl,
    raw.link,
    raw.url,
    mappedLinks[item.videoKey],
    mappedLinks[String(item.videoKey || "").toUpperCase()]
  );
  const url = (found == null ? "" : String(found)).trim();
  // Upstream normalizers can stringify a missing value to "undefined"/"null".
  return url === "undefined" || url === "null" ? "" : url;
}

export function youtubeVideoId(url) {
  if (!url || typeof url !== "string") return "";
  try {
    const parsed = new URL(url);
    const host = parsed.hostname.replace(/^www\./, "").toLowerCase();
    if (host === "youtu.be") return parsed.pathname.split("/").filter(Boolean)[0] || "";
    if (host.endsWith("youtube.com")) {
      if (parsed.pathname === "/watch") return parsed.searchParams.get("v") || "";
      const parts = parsed.pathname.split("/").filter(Boolean);
      if (["embed", "shorts", "live"].includes(parts[0])) return parts[1] || "";
    }
  } catch {
    return "";
  }
  return "";
}

export function buildVideoPlayback(item, offsetSeconds = 0) {
  const url = resolveVideoUrl(item);
  if (!url) return null;

  // Playback offset comes from backend metadata (default 0). There is no
  // hardcoded per-video table here anymore — see PLAYBACK_OFFSETS_JSON.
  const offset = Number(offsetSeconds) || 0;
  const start = Math.max(0, Math.floor((Number(item?.timestamp) || 0) + offset));
  const youtubeId = youtubeVideoId(url);
  if (youtubeId) {
    const embed = new URL(`https://www.youtube.com/embed/${youtubeId}`);
    embed.searchParams.set("autoplay", "1");
    embed.searchParams.set("rel", "0");
    embed.searchParams.set("start", String(start));
    return { type: "youtube", url, embedUrl: embed.toString(), start };
  }

  return { type: "video", url, start };
}
