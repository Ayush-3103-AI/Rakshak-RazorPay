// The site's own chrome: the brand, a line of context, and the one action that
// takes you to the other half.
//
// This exists so the page list on the right can be nothing but page titles.
// Anything that belongs to the SITE rather than to the sequence belongs here —
// which is where it was before the list briefly grew into a full sidebar and
// swallowed it.
import { cn } from "../lib/cn.js";
import Brand from "./Brand.jsx";

export const TOPBAR_HEIGHT = 52;

export default function TopBar({ meta, action, className }) {
  return (
    <header
      style={{ height: TOPBAR_HEIGHT }}
      className={cn(
        "glass fixed top-0 right-0 left-0 z-30 flex items-center justify-between gap-[var(--spacing-5)] !rounded-none !border-x-0 !border-t-0 px-[clamp(16px,3.4vw,56px)] max-lg:hidden",
        className
      )}
    >
      <Brand href="#/" size="xs" />
      <div className="flex shrink-0 items-center gap-[var(--spacing-5)]">
        {meta && (
          <span className="font-mono text-[10px] tracking-[0.16em] text-faint uppercase max-xl:hidden">{meta}</span>
        )}
        {action}
      </div>
    </header>
  );
}
