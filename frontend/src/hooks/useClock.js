import { useEffect, useState } from "react";

/** Live clock used by the workstation status bar. */
export default function useClock() {
  const [clock, setClock] = useState(() => ({ time: "00:00:00", date: "SUN 02 AUG 2026" }));
  useEffect(() => {
    const id = setInterval(() => {
      const d = new Date();
      setClock({
        time: d.toLocaleTimeString("en-GB", { hour12: false }),
        date: d.toLocaleDateString("en-GB", { weekday: "short", day: "2-digit", month: "short", year: "numeric" }).toUpperCase(),
      });
    }, 1000);
    return () => clearInterval(id);
  }, []);
  return clock;
}
