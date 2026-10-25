// Theme: two equivalent first-class themes, resolved once and written
// to the root node.
//
// Panels are plain loopback pages: they style themselves, the desktop
// only reads and writes data-theme.

export type Theme = "light" | "dark";

const STORAGE_KEY = "nan.theme";

export function loadTheme(): Theme {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored === "light" || stored === "dark") return stored;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function saveTheme(theme: Theme): void {
  localStorage.setItem(STORAGE_KEY, theme);
}

export function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
}

export function toggleTheme(theme: Theme): Theme {
  return theme === "dark" ? "light" : "dark";
}

/** The minimal token set for panels: color name -> resolved value. */
export function themeTokens(theme: Theme): Record<string, string> {
  const read = (name: string) =>
    getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  void theme;
  return {
    canvas: read("--canvas"),
    surface: read("--surface"),
    raised: read("--raised"),
    border: read("--border"),
    ink: read("--ink"),
    muted: read("--muted"),
    live: read("--live"),
    warn: read("--warn"),
    error: read("--error"),
  };
}
