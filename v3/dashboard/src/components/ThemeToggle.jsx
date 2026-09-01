// Blade supports both modes; so does this panel. Defaults to the system
// preference (tokens.css's prefers-color-scheme block) and lets a judge
// override it explicitly — the override is what stamps data-theme on <html>,
// which is what tokens.css's :root[data-theme] rules key off.
import { Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";

const STORAGE_KEY = "rakshak-v3-theme";

function apply(theme) {
  const root = document.documentElement;
  if (theme) root.setAttribute("data-theme", theme);
  else root.removeAttribute("data-theme");
}

export default function ThemeToggle() {
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem(STORAGE_KEY) || null;
    } catch {
      return null;
    }
  });

  useEffect(() => {
    apply(theme);
    try {
      if (theme) localStorage.setItem(STORAGE_KEY, theme);
      else localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* private-mode storage can throw; the toggle still works for this load */
    }
  }, [theme]);

  const systemDark = window.matchMedia?.("(prefers-color-scheme: dark)").matches;
  const isDark = theme ? theme === "dark" : systemDark;

  return (
    <button
      type="button"
      onClick={() => setTheme(isDark ? "light" : "dark")}
      className="inline-flex cursor-pointer items-center gap-[var(--spacing-2)] rounded-full border border-border bg-canvas-well px-[var(--spacing-3)] py-[var(--spacing-2)] text-xs font-medium text-muted-foreground transition-colors duration-[var(--duration-quick)] hover:border-border-strong hover:text-foreground"
      aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
    >
      {isDark ? <Sun className="h-3.5 w-3.5" aria-hidden="true" /> : <Moon className="h-3.5 w-3.5" aria-hidden="true" />}
      {isDark ? "Light" : "Dark"}
    </button>
  );
}
