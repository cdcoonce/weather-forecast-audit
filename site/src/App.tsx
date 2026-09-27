import { Header } from "./components/Header";
import { NotFound } from "./components/NotFound";
import { useFetchJson } from "./lib/useFetchJson";
import { CityPage } from "./pages/CityPage";
import { Home } from "./pages/Home";
import { useHashRoute } from "./router/useHashRoute";
import type { CityIndex, Manifest } from "./types/export";

export function App(): JSX.Element {
  const manifestFetch = useFetchJson<Manifest>("manifest.json");
  const cityIndexFetch = useFetchJson<CityIndex>("city_index.json");
  const route = useHashRoute();

  const loading =
    manifestFetch.status === "loading" || cityIndexFetch.status === "loading";
  const errored = manifestFetch.status === "error" || cityIndexFetch.status === "error";

  function retryAll(): void {
    manifestFetch.retry();
    cityIndexFetch.retry();
  }

  return (
    <>
      <a href="#main" className="skip-link">
        Skip to content
      </a>
      <Header manifest={manifestFetch.status === "success" ? manifestFetch.data : null} />
      <main id="main">
        {loading && <p role="status">Loading{"…"}</p>}
        {!loading && errored && (
          <div role="alert" className="fetch-error">
            <p>Couldn&apos;t load the data.</p>
            <button type="button" onClick={retryAll}>
              Retry
            </button>
          </div>
        )}
        {!loading && !errored && cityIndexFetch.status === "success" && (
          <>
            {route.name === "home" && <Home cityIndex={cityIndexFetch.data} />}
            {route.name === "city" && (
              <CityPage icao={route.icao} cityIndex={cityIndexFetch.data} />
            )}
            {route.name === "not-found" && <NotFound />}
          </>
        )}
      </main>
    </>
  );
}
