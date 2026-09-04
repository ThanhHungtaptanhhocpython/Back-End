/**
 * Shared <img> load / error handling for keyframe thumbnails.
 *
 * The backend now serves a missing keyframe as `404 + Cache-Control: no-store`
 * (main.serve_keyframe / _keyframe_missing_response) instead of a permanently
 * cached 200 placeholder. That means:
 *   - a broken thumbnail is never cached, so a later re-render can succeed;
 *   - the <img> `error` event actually fires.
 *
 * These handlers stamp `data-img-state` on the <img> (and its parent, so CSS
 * can style a wrapper overlay) and clear it when a retry loads, instead of
 * leaving the browser's broken-image glyph.
 */

function _mark(event, state) {
  const img = event && event.currentTarget;
  if (!img) return;
  img.dataset.imgState = state;
  const host = img.parentElement;
  if (host) host.dataset.imgState = state;
}

export function onKeyframeImgError(event) {
  _mark(event, "error");
}

export function onKeyframeImgLoad(event) {
  _mark(event, "ok");
}

/** Spread onto a keyframe <img>: `<img src={...} {...keyframeImgProps} />`. */
export const keyframeImgProps = {
  loading: "lazy",
  onError: onKeyframeImgError,
  onLoad: onKeyframeImgLoad,
};
