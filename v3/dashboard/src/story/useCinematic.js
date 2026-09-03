// "May this screen pin and play as you scroll?" — no under reduced motion,
// and no on a narrow viewport, where a pinned figure would fight the reader
// for a screen that has no room to spare. Both fall back to the settled
// figure, so the argument is identical; only the choreography is dropped.
import { useReducedMotion } from "framer-motion";
import { useEffect, useState } from "react";

const QUERY = "(min-width: 900px)";

export function useCinematic() {
  const reduce = useReducedMotion();
  const [wide, setWide] = useState(() => Boolean(globalThis.matchMedia?.(QUERY)?.matches));

  useEffect(() => {
    const mq = globalThis.matchMedia?.(QUERY);
    if (!mq?.addEventListener) return undefined;
    const onChange = (e) => setWide(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  return !reduce && wide;
}
