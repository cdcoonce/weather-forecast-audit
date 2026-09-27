import { useEffect, useState } from "react";

const STORAGE_KEY = "wfa-theme";
type Theme = "light" | "dark";

function readStoredTheme(): Theme | null {
  try {
    const value = window.localStorage.getItem(STORAGE_KEY);
    return value === "light" || value === "dark" ? value : null;
  } catch {
    return null;
  }
}

function writeStoredTheme(theme: Theme): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // Storage may throw (private browsing, quota, disabled). The toggle
    // still works for this page load; it just won't be remembered.
  }
}

function prefersDark(): boolean {
  try {
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  } catch {
    return false;
  }
}

/**
 * Sets `data-theme` on `<html>`: "light"/"dark" once the visitor picks one
 * explicitly (remembered in localStorage), or no attribute at all so
 * `theme.css`'s `prefers-color-scheme` rule follows the system setting.
 */
export function ThemeToggle(): JSX.Element {
  const [explicit, setExplicit] = useState<Theme | null>(() => readStoredTheme());
  const [systemDark, setSystemDark] = useState<boolean>(() => prefersDark());

  useEffect(() => {
    if (explicit) {
      document.documentElement.setAttribute("data-theme", explicit);
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
  }, [explicit]);

  useEffect(() => {
    let mql: MediaQueryList | null = null;
    try {
      mql = window.matchMedia("(prefers-color-scheme: dark)");
    } catch {
      return;
    }
    const listener = (event: MediaQueryListEvent) => setSystemDark(event.matches);
    mql.addEventListener("change", listener);
    return () => mql?.removeEventListener("change", listener);
  }, []);

  const isDark = explicit ? explicit === "dark" : systemDark;

  function toggle(): void {
    const next: Theme = isDark ? "light" : "dark";
    setExplicit(next);
    writeStoredTheme(next);
  }

  return (
    <button
      type="button"
      className="theme-toggle"
      aria-pressed={isDark}
      onClick={toggle}
    >
      {isDark ? "Light" : "Dark"}
    </button>
  );
}
