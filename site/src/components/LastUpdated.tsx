import type { Manifest } from "../types/export";
import { formatGeneratedAt } from "../lib/format";

export function LastUpdated({ manifest }: { manifest: Manifest }): JSX.Element {
  const updated = formatGeneratedAt(manifest.generated_at);
  return (
    <p className="last-updated">
      {manifest.data_through ? `Data through ${manifest.data_through}` : "No scored data yet"}
      {" · "}
      updated {updated}
    </p>
  );
}
