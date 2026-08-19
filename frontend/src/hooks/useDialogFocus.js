import { useEffect, useRef } from "react";

const FOCUSABLE = [
  "a[href]",
  "button:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  "[tabindex]:not([tabindex='-1'])",
].join(",");

export default function useDialogFocus(initialFocusRef, active = true) {
  const dialogRef = useRef(null);

  useEffect(() => {
    if (!active) return undefined;
    const dialog = dialogRef.current;
    if (!dialog) return undefined;

    const focusInitial = () => {
      const target = initialFocusRef?.current || dialog.querySelector(FOCUSABLE) || dialog;
      target.focus({ preventScroll: true });
    };

    const frame = requestAnimationFrame(focusInitial);
    const trapTab = (event) => {
      if (event.key !== "Tab") return;
      const focusable = Array.from(dialog.querySelectorAll(FOCUSABLE)).filter(
        (element) => !element.hasAttribute("hidden") && element.getAttribute("aria-hidden") !== "true",
      );
      if (!focusable.length) {
        event.preventDefault();
        dialog.focus({ preventScroll: true });
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    dialog.addEventListener("keydown", trapTab);
    return () => {
      cancelAnimationFrame(frame);
      dialog.removeEventListener("keydown", trapTab);
    };
  }, [active, initialFocusRef]);

  return dialogRef;
}
