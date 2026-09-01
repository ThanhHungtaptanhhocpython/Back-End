/** Shared UI constants. */

export const SEARCH_TYPES = [
  { value: "TEXT", label: "Text" },
  { value: "QA", label: "Q&A" },
  // Backend runs OCR only (no object detection); "/ocrandodsearch" is kept as a
  // compatibility alias for the old "OCR+OD" type value.
  { value: "OCR", label: "OCR Text" },
  { value: "IMAGE", label: "Image" },
  { value: "TEMPORAL", label: "Temporal" },
];

export const CARD_W = 188;
export const GAP = 12;
