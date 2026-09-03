// One deliberate push, one page.
//
// CSS snapping alone cannot express "a scroll moves a whole page": it decides
// where a scroll ENDS, so a small gesture settles back and a large one carries
// you a screen, and everything between is a judgement call the browser makes,
// not us. This hook makes the rule explicit — accumulate the wheel, and once
// the reader has pushed past a threshold, move exactly one page and refuse to
// move again until that move has landed.
//
// WHAT IT DOES NOT TAKE OVER. It catches wheel and touch and nothing else. The
// document still scrolls, and the scroll position is still real, so Page Down,
// Home, End, arrow keys, find-in-page, keyboard focus, deep links and screen
// readers all keep working exactly as they did — none of them go through the
// wheel. That is the whole reason it is written this way rather than as a
// transform stack: this is the version of paging that survives the objection
// the project already sustained once against v1's playhead.
//
// The escape hatches, deliberately:
//   - prefers-reduced-motion turns the whole thing off, along with the CSS
//     snapping, leaving an ordinary document.
//   - a scrollable element under the pointer gets the gesture first, and only
//     once it is exhausted does the page take over. A table that scrolls
//     inside a screen is still readable.
import { animate, useReducedMotion } from "framer-motion";
import { useEffect } from "react";

// How much wheel a reader must push before the page moves. One notch of a
// mouse wheel is ~100-120px in Chrome, so this deliberately sits above a
// single notch: a page turn takes a real push, not a twitch. One trackpad
// swipe clears it easily.
const THRESHOLD = 160;
// The accumulator forgets an unfinished push, so a slow drift never adds up
// into a page turn the reader did not ask for.
const FORGET_MS = 220;
// One page change, in seconds.
const GLIDE = 0.6;
const EASE = [0.22, 1, 0.36, 1];
// A touch has to travel this far before it counts as a swipe.
const SWIPE_PX = 48;

/** Can this element, or one of its ancestors, still scroll `delta` itself? */
function scrollableUnder(node, delta) {
  for (let el = node; el && el !== document.body; el = el.parentElement) {
    const style = getComputedStyle(el);
    if (!/(auto|scroll)/.test(style.overflowY)) continue;
    const room = el.scrollHeight - el.clientHeight;
    if (room <= 1) continue;
    if (delta > 0 && el.scrollTop < room - 1) return true;
    if (delta < 0 && el.scrollTop > 1) return true;
  }
  return false;
}

/**
 * @param ids     section ids in order; the hook pages between them
 * @param enabled false to stand down entirely
 */
export function usePagedScroll(ids, { enabled = true } = {}) {
  const reduce = useReducedMotion();

  useEffect(() => {
    if (!enabled || reduce || !ids.length) return undefined;

    let accumulated = 0;
    let forgetTimer = null;
    let moving = false;
    let controls = null;

    const pageHeight = () => window.innerHeight;
    const currentIndex = () => Math.round(window.scrollY / pageHeight());

    function goTo(index) {
      const clamped = Math.max(0, Math.min(ids.length - 1, index));
      const target = clamped * pageHeight();
      if (Math.abs(target - window.scrollY) < 2) return;

      moving = true;
      accumulated = 0;
      // Snapping is suspended for the duration: the snap controller and this
      // animation both want the scroll position, and the animation lands on a
      // snap point anyway. It is restored the moment the move finishes, so a
      // keyboard or find-in-page scroll is still caught by CSS.
      const root = document.documentElement;
      root.style.scrollSnapType = "none";

      const done = () => {
        root.style.scrollSnapType = "";
        moving = false;
        controls = null;
      };

      controls = animate(window.scrollY, target, {
        duration: GLIDE,
        ease: EASE,
        onUpdate: (y) => window.scrollTo(0, y),
        onComplete: done,
      });
    }

    function onWheel(event) {
      if (event.ctrlKey) return; // pinch-zoom is not a page turn
      if (scrollableUnder(event.target, event.deltaY)) return;

      // The page owns this gesture from here, so the document must not also
      // scroll freely underneath it.
      event.preventDefault();
      if (moving) return;

      accumulated += event.deltaY;
      clearTimeout(forgetTimer);
      forgetTimer = setTimeout(() => {
        accumulated = 0;
      }, FORGET_MS);

      if (Math.abs(accumulated) < THRESHOLD) return;
      goTo(currentIndex() + Math.sign(accumulated));
    }

    let touchY = null;
    function onTouchStart(event) {
      touchY = event.touches[0]?.clientY ?? null;
    }
    function onTouchMove(event) {
      if (touchY == null || moving) return;
      const y = event.touches[0]?.clientY ?? touchY;
      const travelled = touchY - y;
      if (scrollableUnder(event.target, travelled)) return;
      event.preventDefault();
      if (Math.abs(travelled) < SWIPE_PX) return;
      touchY = null;
      goTo(currentIndex() + Math.sign(travelled));
    }
    function onTouchEnd() {
      touchY = null;
    }

    window.addEventListener("wheel", onWheel, { passive: false });
    window.addEventListener("touchstart", onTouchStart, { passive: true });
    window.addEventListener("touchmove", onTouchMove, { passive: false });
    window.addEventListener("touchend", onTouchEnd, { passive: true });

    return () => {
      window.removeEventListener("wheel", onWheel);
      window.removeEventListener("touchstart", onTouchStart);
      window.removeEventListener("touchmove", onTouchMove);
      window.removeEventListener("touchend", onTouchEnd);
      clearTimeout(forgetTimer);
      controls?.stop();
      document.documentElement.style.scrollSnapType = "";
    };
  }, [ids, enabled, reduce]);
}
