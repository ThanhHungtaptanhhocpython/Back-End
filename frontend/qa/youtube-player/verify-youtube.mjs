/**
 * Real-Chromium verification for the useVideoPlayer YouTube integration.
 *
 * Prereq:
 *   npm run dev                 # Vite dev server on http://localhost:5173
 *   npm i -D --no-save playwright   # not a project dependency
 *
 * Run:
 *   node qa/youtube-player/verify-youtube.mjs
 *
 * It launches Chromium twice - once with autoplay allowed (the permissive
 * profile a headless CI browser uses) and once with autoplay gated on a user
 * gesture (what a normal desktop Chromium does) - and checks that playback
 * actually starts (state PLAYING + advancing currentTime), that a blocked
 * autoplay produces a working click-to-play, that StrictMode leaves exactly
 * one iframe, and that hide/show and id changes stay clean.
 */
import { chromium } from "playwright";

const BASE = process.env.HARNESS_URL || "http://localhost:5173/qa/youtube-player/harness.html";
const GOOD = "M7lc1UVf-VE";
const HTV = "mjy8h-iT-ms";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function state(page) {
  return page.evaluate(() => window.__harnessState || {});
}
async function waitReady(page, ms = 20000) {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    const s = await state(page);
    if (s.ready || s.error) return s;
    await sleep(250);
  }
  return state(page);
}
function advanced(times) {
  const nums = (times || []).filter((t) => typeof t === "number");
  if (nums.length < 2) return false;
  return Math.max(...nums) - Math.min(...nums) > 0.75;
}

async function scenario(label, launchArgs, body) {
  const browser = await chromium.launch({
    args: launchArgs,
    // Playwright pins a permissive autoplay policy by default; drop it so the
    // "desktop" scenario can actually gate autoplay on a user gesture.
    ignoreDefaultArgs: ["--autoplay-policy=no-user-gesture-required"],
  });
  const page = await browser.newPage();
  const logs = [];
  page.on("console", (m) => logs.push(`${m.type()}: ${m.text()}`));
  const result = { label, steps: {} };
  try {
    await body(page, result);
  } catch (err) {
    result.error = String(err && err.stack ? err.stack : err);
  } finally {
    result.harnessLog = await page.evaluate(() => window.__harnessLog || []).catch(() => []);
    await browser.close();
  }
  return result;
}

const AUTOPLAY_OK = ["--autoplay-policy=no-user-gesture-required"];
const AUTOPLAY_BLOCKED = ["--autoplay-policy=document-user-activation-required", "--mute-audio"];

const results = [];

// 1. Permissive profile: playback must really start.
results.push(
  await scenario("autoplay-allowed / plays for real", AUTOPLAY_OK, async (page, r) => {
    await page.goto(`${BASE}?v=${GOOD}&show=1`);
    const ready = await waitReady(page);
    r.steps.reachedReady = Boolean(ready.ready);
    await sleep(4500);
    const s = await state(page);
    r.steps.playerState = s.playerState;
    r.steps.playing = s.playing;
    r.steps.timeAdvanced = advanced(s.times);
    r.steps.iframeCount = s.iframeCount;
    r.pass =
      r.steps.reachedReady && r.steps.playing && r.steps.timeAdvanced && s.iframeCount === 1;
  }),
);

// 2. Desktop profile: autoplay blocked, then click-to-play must work.
results.push(
  await scenario("autoplay-blocked / click-to-play recovers", AUTOPLAY_BLOCKED, async (page, r) => {
    await page.goto(`${BASE}?v=${GOOD}&show=1`);
    const ready = await waitReady(page);
    r.steps.reachedReady = Boolean(ready.ready);
    await sleep(4500);
    let s = await state(page);
    r.steps.beforeClick = {
      playing: s.playing,
      playerState: s.playerState,
      autoplayBlocked: s.autoplayBlocked,
      timeAdvanced: advanced(s.times),
    };
    r.steps.autoplayWasBlocked = !s.playing && !advanced(s.times);

    // A real user gesture: prefer the rendered shade button, fall back to the hook.
    const btn = page.locator("#btn-clicktoplay");
    if (await btn.count()) await btn.click();
    else await page.evaluate(() => window.__play && window.__play());

    await sleep(4500);
    s = await state(page);
    r.steps.afterClick = {
      playing: s.playing,
      playerState: s.playerState,
      timeAdvanced: advanced(s.times),
      iframeCount: s.iframeCount,
    };
    r.pass =
      r.steps.reachedReady &&
      r.steps.autoplayWasBlocked &&
      s.playing &&
      advanced(s.times) &&
      s.iframeCount === 1;
  }),
);

// 3. StrictMode + hide/show + id change, on the permissive profile.
results.push(
  await scenario("strictmode / hide-show / id-change stay single", AUTOPLAY_OK, async (page, r) => {
    await page.goto(`${BASE}?v=${GOOD}&show=1`);
    await waitReady(page);
    await sleep(2500);
    let s = await state(page);
    r.steps.afterMount_iframes = s.iframeCount;
    const log = await page.evaluate(() => window.__harnessLog || []);
    r.steps.onReadyCount = log.filter((l) => l.includes("onReady")).length;
    r.steps.cleanupCount = log.filter((l) => l.includes("cleanup")).length;

    // hide then show
    await page.evaluate(() => window.__setShow(false));
    await sleep(800);
    const hidden = await state(page);
    r.steps.iframesWhileHidden = hidden.iframeCount;
    await page.evaluate(() => window.__setShow(true));
    await waitReady(page);
    await sleep(3000);
    s = await state(page);
    r.steps.afterReshow_iframes = s.iframeCount;
    r.steps.afterReshow_timeAdvanced = advanced(s.times);

    // change id
    await page.evaluate(() => window.__setVideo("dQw4w9WgXcQ"));
    await sleep(3500);
    s = await state(page);
    r.steps.afterIdChange_iframes = s.iframeCount;
    r.steps.afterIdChange_error = s.error;

    // invalid id -> numeric error surfaced
    await page.evaluate(() => window.__setVideo("00000000000"));
    await sleep(3500);
    s = await state(page);
    r.steps.invalidIdError = s.error;

    r.pass =
      r.steps.afterMount_iframes === 1 &&
      (r.steps.iframesWhileHidden ?? 0) === 0 &&
      r.steps.afterReshow_iframes === 1 &&
      r.steps.afterReshow_timeAdvanced &&
      r.steps.afterIdChange_iframes === 1 &&
      !r.steps.afterIdChange_error &&
      /YouTube error \d+/.test(String(r.steps.invalidIdError || ""));
  }),
);

// 4. The HTV video from the brief (report-only; availability varies).
results.push(
  await scenario("htv video mjy8h-iT-ms (report only)", AUTOPLAY_OK, async (page, r) => {
    await page.goto(`${BASE}?v=${HTV}&show=1`);
    const ready = await waitReady(page, 15000);
    await sleep(4000);
    const s = await state(page);
    r.steps = {
      reachedReady: Boolean(ready.ready),
      error: s.error,
      playerState: s.playerState,
      playing: s.playing,
      timeAdvanced: advanced(s.times),
    };
    r.pass = null; // informational
  }),
);

let failed = 0;
for (const r of results) {
  const tag = r.pass === null ? "INFO" : r.pass ? "PASS" : "FAIL";
  if (r.pass === false) failed++;
  console.log(`\n=== [${tag}] ${r.label} ===`);
  console.log(JSON.stringify(r.steps, null, 2));
  if (r.error) console.log("ERROR:", r.error);
  if (r.harnessLog?.length) {
    console.log("--- hook instrumentation ---");
    console.log(r.harnessLog.slice(0, 40).join("\n"));
  }
}
console.log(`\n${failed === 0 ? "ALL CHECKS PASSED" : failed + " CHECK(S) FAILED"}`);
process.exit(failed === 0 ? 0 : 1);
