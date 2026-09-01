// Viewport hooks, ported from ver1/dashboard/src/useInView.js (T-0127, #61).
// `useScrolledSection` no longer takes an `enabled` flag — v1 needed it to
// stand down while the playhead was jacking the scroll; this shell never
// jacks, so it just runs.
import { useEffect, useState } from "react";
import { SECTIONS } from "./sections.js";

/**
 * "Has the reader reached this yet?" — the trigger for loading an artifact on
 * entry to the section that needs it, rather than eagerly on first paint.
 * Latches: once reached, it stays reached, so leaving and returning does not
 * re-trigger the fetch (the loader's own cache then makes it free anyway).
 */
export function useInView(ref, { rootMargin = "200px" } = {}) {
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || seen) return undefined;
    const observer = new IntersectionObserver(
      (entries) => entries.some((e) => e.isIntersecting) && setSeen(true),
      { rootMargin }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [ref, seen, rootMargin]);
  return seen;
}

/** Which section is actually on screen, for the rail's active state. */
export function useScrolledSection() {
  const [index, setIndex] = useState(0);
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting);
        if (!visible.length) return;
        const top = visible.reduce((a, b) => (a.intersectionRatio > b.intersectionRatio ? a : b));
        setIndex(SECTIONS.findIndex((s) => s.id === top.target.id));
      },
      { threshold: [0.25, 0.5, 0.75] }
    );
    SECTIONS.forEach((s) => {
      const el = document.getElementById(s.id);
      if (el) observer.observe(el);
    });
    return () => observer.disconnect();
  }, []);
  return index;
}
