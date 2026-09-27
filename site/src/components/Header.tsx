import type { Manifest } from "../types/export";
import { LastUpdated } from "./LastUpdated";
import { ThemeToggle } from "./ThemeToggle";

export function Header({ manifest }: { manifest: Manifest | null }): JSX.Element {
  return (
    <header className="site-header">
      <a href="#/" className="site-header__brand">
        Forecast Audit
      </a>
      {manifest ? <LastUpdated manifest={manifest} /> : null}
      <ThemeToggle />
    </header>
  );
}
