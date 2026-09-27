import { CityCard } from "../components/CityCard";
import { NotFound } from "../components/NotFound";
import { useFetchJson } from "../lib/useFetchJson";
import type { City, CityIndex } from "../types/export";

export interface CityPageProps {
  icao: string;
  cityIndex: CityIndex;
}

export function CityPage({ icao, cityIndex }: CityPageProps): JSX.Element {
  const entry = cityIndex.cities.find((city) => city.icao === icao);
  // Called unconditionally (rules of hooks); its result is only used once
  // `entry` confirms this is a known ICAO.
  const cityFetch = useFetchJson<City>(`cities/${icao}.json`);

  if (!entry) {
    return <NotFound icao={icao} />;
  }

  if (cityFetch.status === "loading") {
    return <p role="status">Loading{"…"}</p>;
  }

  if (cityFetch.status === "error") {
    return (
      <div role="alert" className="fetch-error">
        <p>Couldn&apos;t load the data.</p>
        <button type="button" onClick={cityFetch.retry}>
          Retry
        </button>
      </div>
    );
  }

  return <CityCard city={cityFetch.data} summary={entry.summary} />;
}
