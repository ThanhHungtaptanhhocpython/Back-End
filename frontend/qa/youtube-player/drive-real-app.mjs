/**
 * Drive the REAL app (frontend :5173 + FastAPI :3000) through the flow the fix
 * touches: search -> open a result in Review -> Play video -> confirm real
 * playback -> Capture frame.
 *
 *   npm i -D --no-save playwright
 *   node qa/youtube-player/drive-real-app.mjs
 */
import { chromium } from "playwright";

const APP = process.env.APP_URL || "http://localhost:5173/";
const QUERIES = process.env.Q ? [process.env.Q] : ["a news anchor in a studio", "a man riding a motorbike", "people walking on a street"];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await chromium.launch(); // Playwright Chromium; autoplay is permitted
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });

const hookLog = [];
page.on("console", (m) => {
  const t = m.text();
  if (t.includes("[useVideoPlayer]")) hookLog.push(t);
});
const pageErrors = [];
page.on("pageerror", (e) => pageErrors.push(String(e)));

function stateChanges() {
  return hookLog
    .filter((l) => l.includes("onStateChange"))
    .map((l) => {
      const m = /\{.*\}/.exec(l);
      try { return JSON.parse(m[0]); } catch { return null; }
    })
    .filter(Boolean);
}

const report = { app: APP, steps: [] };
const step = (name, data) => { report.steps.push({ name, ...data }); console.log(`\n• ${name}`); console.log(JSON.stringify(data, null, 2)); };

try {
  await page.goto(APP, { waitUntil: "domcontentloaded" });
  await page.waitForSelector("input.ws-query", { timeout: 15000 });
  step("app loaded", { title: await page.title() });

  // --- search until we get result cards ---
  let cards = 0;
  for (const q of QUERIES) {
    await page.fill("input.ws-query", q);
    await page.press("input.ws-query", "Enter");
    await page.waitForFunction(() => document.querySelectorAll(".ws-card").length > 0, { timeout: 20000 }).catch(() => {});
    cards = await page.locator(".ws-card").count();
    if (cards > 0) { step("search returned results", { query: q, cards }); break; }
  }
  if (!cards) throw new Error("no result cards from any query");

  // --- open results until one exposes the "Play video" button ---
  let opened = null;
  const n = Math.min(cards, 12);
  for (let i = 0; i < n; i++) {
    await page.locator(".ws-card").nth(i).click();
    await page.waitForSelector(".ws-review", { timeout: 8000 });
    // wait for playback metadata fetch to settle
    await sleep(1500);
    const playBtn = page.locator(".ws-review button", { hasText: "Play video" });
    const videoKey = await page.locator(".ws-review-ctx").first().textContent().catch(() => "");
    if (await playBtn.count()) { opened = { index: i, videoKey: (videoKey || "").trim() }; break; }
    await page.keyboard.press("Escape");
    await sleep(300);
  }
  if (!opened) throw new Error("no reviewed result had a playable video URL");
  step("opened a playable result in Review", opened);

  // --- Play video ---
  hookLog.length = 0;
  await page.locator(".ws-review button", { hasText: "Play video" }).click();
  await page.waitForSelector(".ws-review-yt", { timeout: 8000 });
  await page.waitForFunction(() => document.querySelectorAll(".ws-review-yt iframe").length === 1, { timeout: 15000 });
  await sleep(6000); // let it actually play

  const iframeInfo = await page.evaluate(() => {
    const f = document.querySelector(".ws-review-yt iframe");
    return {
      iframesInHost: document.querySelectorAll(".ws-review-yt iframe").length,
      iframesInDoc: document.querySelectorAll("iframe").length,
      hostChildTag: document.querySelector(".ws-review-yt")?.firstElementChild?.tagName || null,
      src: f?.src || null,
      connected: !!f?.isConnected,
    };
  });
  const sc = stateChanges();
  const times = sc.map((s) => s.currentTime).filter((t) => typeof t === "number");
  const playing = sc.some((s) => s.state === 1);
  const timeAdvanced = times.length > 1 && Math.max(...times) - Math.min(...times) > 0.75;
  const hintText = await page.locator(".ws-review-capture-hint").textContent().catch(() => "");
  step("Play video → real playback check", {
    ...iframeInfo,
    reachedPlayingState: playing,
    currentTimeSamples: times,
    timeAdvanced,
    autoplayShadeShown: await page.locator("#btn-clicktoplay, .ws-review-play-shade").count() > 0,
    captureHint: (hintText || "").trim(),
    hookEvents: hookLog.slice(0, 12),
  });

  // --- Capture frame ---
  const capBtn = page.locator(".ws-review button", { hasText: "Capture frame" });
  await capBtn.click().catch(() => {});
  await page.waitForSelector(".ws-review-capture-note", { timeout: 10000 }).catch(() => {});
  const note = await page.locator(".ws-review-capture-note").textContent().catch(() => "(none)");
  const trayCount = await page.locator(".ws-tray-item, .ws-selection-item, [data-tray-item]").count().catch(() => -1);
  step("Capture frame", { note: (note || "").trim(), disabledAtClick: await capBtn.isDisabled().catch(() => null), trayItems: trayCount });

  // --- hide / re-show (lifecycle) ---
  await page.locator(".ws-review button", { hasText: "Show frame" }).click().catch(() => {});
  await sleep(800);
  const afterHide = await page.evaluate(() => document.querySelectorAll(".ws-review-yt iframe").length);
  await page.locator(".ws-review button", { hasText: "Play video" }).click().catch(() => {});
  await page.waitForFunction(() => document.querySelectorAll(".ws-review-yt iframe").length === 1, { timeout: 15000 }).catch(() => {});
  await sleep(3000);
  const afterReshow = await page.evaluate(() => document.querySelectorAll(".ws-review-yt iframe").length);
  step("hide → re-show", { iframesWhileHidden: afterHide, iframesAfterReshow: afterReshow });

  await page.screenshot({ path: "qa/youtube-player/real-app.png", fullPage: false });
  report.screenshot = "qa/youtube-player/real-app.png";
  report.pageErrors = pageErrors;
} catch (err) {
  report.error = String(err && err.stack ? err.stack : err);
  await page.screenshot({ path: "qa/youtube-player/real-app-error.png" }).catch(() => {});
} finally {
  await browser.close();
}

console.log("\n================ SUMMARY ================");
console.log(JSON.stringify(report, null, 2));
process.exit(report.error ? 1 : 0);
