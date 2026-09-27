import { CitySearch } from "../components/CitySearch";
import type { CityIndex } from "../types/export";

export function Home({ cityIndex }: { cityIndex: CityIndex }): JSX.Element {
  return (
    <div className="home">
      <p className="home__intro">
        The National Blend of Models (NBM) is the machine forecast National
        Weather Service forecasters start from for daily high and low
        temperatures. This audit grades that guidance against what actually
        happened at climate-report stations across the continental US, by
        season and lead time.
      </p>
      <CitySearch cities={cityIndex.cities} />
    </div>
  );
}
