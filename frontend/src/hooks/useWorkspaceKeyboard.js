import { useEffect } from "react";

const NAV_CODES = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"]);
const GRID_NAV_CODES = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"]);

/**
 * Contextual keyboard shortcuts for the workstation.
 *
 * Mandatory keys (always on — cannot be disabled):
 *   ?            open help (outside fields and outside IME composition)
 *   Tab / Shift+Tab   native navigation, never intercepted
 *   ↑ ↓ ← → Home End  roving focus across the results grid
 *   Enter        open the focused frame reviewer
 *   Space        keep / unkeep — only when a result card or reviewer owns focus
 *   Delete       remove frame with undo — never Backspace
 *   ← / →        reviewer previous / next
 *   Esc          layered: first Esc leaves a field, next closes the top overlay;
 *                closing restores focus to the exact invoker card / button
 *
 * Optional power-user mode (off by default — separate setting):
 *   /  focus query   S  similar pivot   X  remove   E  export
 *
 * Safety rules enforced here:
 *  - IME composition (`isComposing` / keyCode 229) is never intercepted.
 *  - Ctrl/Cmd/Alt are never intercepted — the app uses plain contextual keys.
 *  - Accelerators never fire inside input/textarea/select/contenteditable.
 *  - Auto-repeat is allowed ONLY for navigation arrows.
 *  - No Vim HJKL, no brackets, no G-chords, no Cmd+K, no Alt combos, no F2,
 *    no Backspace. Focused native buttons always win their own Space/Enter
 *    activation, so shortcuts never double-activate a control.
 *
 * `handlersRef.current` is refreshed every render with the latest state and
 * callbacks so the single listener never sees stale closures.
 */
export default function useWorkspaceKeyboard(handlersRef) {
  useEffect(() => {
    const keydown = (e) => {
      const H = handlersRef.current;

      /* IME composition — never intercept */
      if (e.isComposing || e.keyCode === 229) return;

      const tgt = e.target;
      const inField = tgt && (tgt.tagName === "INPUT" || tgt.tagName === "TEXTAREA" || tgt.tagName === "SELECT" || tgt.isContentEditable);
      const onCard = tgt && tgt.dataset && tgt.dataset.wscard;
      const onInteractive = tgt && (tgt.tagName === "BUTTON" || tgt.tagName === "A");
      const code = e.code;

      /* never intercept modifier chords — plain contextual keys only */
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      /* "?" (Shift+Slash) is a reserved keycap, not a chord */
      if (e.shiftKey && code !== "Slash") return;

      /* auto-repeat is allowed only for navigation arrows */
      if (e.repeat && !NAV_CODES.has(code)) return;

      /* typing surfaces never receive single-key accelerators;
         Esc is the one key that still leaves the field (layered close) */
      if (inField) {
        if (code === "Escape") {
          e.preventDefault();
          tgt.blur();
        }
        return;
      }

      /* help is available outside fields/IME in every context */
      if (code === "Slash" && e.shiftKey) {
        e.preventDefault();
        H.toggleHelp();
        return;
      }

      /* ---------- top-most modal owns focus ---------- */
      if (H.showShortcuts) {
        if (code === "Escape") { e.preventDefault(); H.closeHelp(); }
        return;
      }
      if (H.exportOpen) {
        if (code === "Escape") { e.preventDefault(); H.closeExport(); }
        return;
      }

      /* ---------- reviewer overlay owns focus ---------- */
      if (H.reviewItem) {
        if (code === "Escape") { e.preventDefault(); H.closeReview(); return; }
        if (code === "ArrowRight") { e.preventDefault(); H.reviewNav(1); return; }
        if (code === "ArrowLeft") { e.preventDefault(); H.reviewNav(-1); return; }
        /* let a focused native button keep its own Space/Enter activation */
        if (!onInteractive) {
          if (code === "Space") { e.preventDefault(); H.keepCurrent(); return; }
          if (code === "Delete") { e.preventDefault(); H.removeCurrent(); return; }
          if (H.powerUser) {
            if (code === "KeyS") { e.preventDefault(); H.pivotCurrent(); return; }
            if (code === "KeyX") { e.preventDefault(); H.removeCurrent(); return; }
          }
        }
        return;
      }

      /* ---------- focused-reading overlay owns focus ---------- */
      if (H.chatFocus) {
        if (code === "Escape") { e.preventDefault(); H.closeChatFocus(); return; }
        return;
      }

      /* ---------- workspace context ---------- */
      if (code === "Slash") {
        if (H.powerUser) {
          e.preventDefault();
          H.focusQuery();
        }
        return;
      }

      if (code === "Escape") {
        e.preventDefault();
        H.escWorkspace();
        return;
      }

      if (H.powerUser && code === "KeyE") {
        e.preventDefault();
        H.openExport();
        return;
      }

      /* ---------- grid navigation only when a card owns focus ---------- */
      if (onCard) {
        if (GRID_NAV_CODES.has(code)) { e.preventDefault(); H.moveGridFocus(code); return; }
        if (code === "Enter") { e.preventDefault(); if (H.focusedItem) H.openReview(H.focusedItem); return; }
        if (code === "Space") { e.preventDefault(); if (H.focusedItem) H.toggleKeep(H.focusedItem); return; }
        if (code === "Delete") { e.preventDefault(); if (H.focusedItem) H.removeWithUndo(H.focusedItem); return; }
        if (H.powerUser) {
          if (code === "KeyS") { e.preventDefault(); if (H.focusedItem) H.pivot(H.focusedItem); return; }
          if (code === "KeyX") { e.preventDefault(); if (H.focusedItem) H.removeWithUndo(H.focusedItem); return; }
        }
        return;
      }
    };
    window.addEventListener("keydown", keydown);
    return () => window.removeEventListener("keydown", keydown);
  }, [handlersRef]);
}
