/** Small shared UI primitive: keyboard keycap chip. */
export default function Keycap({ children, tone }) {
  return <span className={`ws-keycap ${tone ? "amber" : ""}`}>{children}</span>;
}
