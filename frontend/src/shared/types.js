/**
 * Shared data shapes used across features.
 *
 * The project is plain JS today; these typedefs document the implicit
 * contracts so real providers/backends can be wired in later.
 *
 * @typedef {Object} FrameItem
 * @property {string} id            - Stable frame id, e.g. "FR-CAM01-001"
 * @property {number} gid           - Synthetic global ordinal
 * @property {number} globalFrameId - Global frame id used in submissions
 * @property {string} folderKey     - Archive folder, e.g. "ARC_DOCK"
 * @property {string} videoKey      - Video/camera feed key, e.g. "cam01_boarding"
 * @property {string} camera        - Camera label, e.g. "CAM 01"
 * @property {string} frameKey      - Archive frame key (zero-padded)
 * @property {string} frameName     - `${videoKey}_${frameKey}`
 * @property {number} timestamp     - Elapsed seconds in the video
 * @property {string} timecode      - HH:MM:SS:FF
 * @property {number} fps           - Nominal fps
 * @property {number} width         - Native width
 * @property {number} height        - Native height
 * @property {string} image         - URL / data-URI thumbnail
 * @property {string} link          - Archive video link (stub)
 * @property {boolean} real         - True for real assets (live-feed frames)
 * @property {number} [faissIndex]  - Mock vector id (backend placeholder)
 * @property {number} [seed]        - Deterministic seed
 * @property {number} [motion]      - Mock motion score
 * @property {string} [ocrText]     - OCR tokens (mock)
 * @property {string[]} [odClasses] - Detected object classes (mock)
 * @property {number} [score]       - Search score 0..1 (set by search)
 * @property {number} [rank]        - Rank within results (set by search)
 * @property {string} [answer]      - QA answer (set by QA search)
 */

/**
 * @typedef {Object} QueryTab
 * @property {string} key           - Tab key, e.g. "q1"
 * @property {string} label         - Tab label
 * @property {string} searchType    - One of SEARCH_TYPES values
 * @property {string} query         - Query text
 * @property {{topk:number, clip:boolean, clipv2:boolean, imageFile:File|string|null}} params
 * @property {"running"|"done"|"err"} status
 * @property {number} latency       - Mock latency ms
 * @property {FrameItem[]} results  - Ranked result frames
 * @property {number} total         - Result count
 * @property {Object|null} meta     - Backend QA summary/diagnostics
 * @property {"live"|"demo"|"fallback"|null} resultSource
 * @property {string|null} resultMode
 */

/**
 * @typedef {Object} ChatMessage
 * @property {string} id
 * @property {"user"|"assistant"} role
 * @property {string} text
 * @property {FrameItem[]} frames   - Frames the message is grounded on
 * @property {boolean} demo         - True when the text is a mock/placeholder answer
 */

export {};
