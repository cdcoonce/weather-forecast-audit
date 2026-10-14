import { useId, useState } from "react";
import { CitySearch } from "../components/CitySearch";
import { ScorecardTable } from "../components/ScorecardTable";
import { useFetchJson } from "../lib/useFetchJson";
import type {
  CityIndex,
  Scorecard as ScorecardData,
  ScorecardCity,
  ScorecardLog,
  ScorecardWindow,
} from "../types/export";

function FetchError({ onRetry }: { onRetry: () => void }): JSX.Element {
  return (
    <div role="alert" className="fetch-error">
      <p>Couldn&apos;t load the data.</p>
      <button type="button" onClick={onRetry}>
        Retry
      </button>
    </div>
  );
}

function WindowLine({ window }: { window: ScorecardWindow }): JSX.Element {
  return (
    <p className="scorecard__window">
      Target dates {window.start} to {window.end} ({window.days} days).
    </p>
  );
}

function LogBlock({ log }: { log: ScorecardLog }): JSX.Element {
  return (
    <section aria-label="Prediction log" className="scorecard__log">
      <h2 className="scorecard__log-title">Prediction log</h2>
      <ul className="scorecard__log-list">
        <li>{log.partitions} partitions</li>
        <li>{log.rows} rows</li>
        <li>
          head <code>{log.head_chain_sha256.slice(0, 12)}</code>
        </li>
        <li>verified: {log.verified ? "yes" : "no"}</li>
      </ul>
    </section>
  );
}

function CityScorecard({
  icao,
  label,
}: {
  icao: string;
  label: string;
}): JSX.Element {
  const headingId = useId();
  const cityFetch = useFetchJson<ScorecardCity>(`scorecard_cities/${icao}.json`);

  let body: JSX.Element;
  if (cityFetch.status === "loading") {
    body = <p role="status">Loading{"…"}</p>;
  } else if (cityFetch.status === "error") {
    body = <FetchError onRetry={cityFetch.retry} />;
  } else if (cityFetch.data.stats.length === 0) {
    body = (
      <p className="scorecard__empty">
        No graded predictions for {label} in this window.
      </p>
    );
  } else {
    body = (
      <ScorecardTable
        stats={cityFetch.data.stats}
        windowDays={cityFetch.data.window?.days ?? 30}
      />
    );
  }

  return (
    <section aria-labelledby={headingId} className="scorecard__city">
      <h2 id={headingId} className="scorecard__city-title">
        {label} <span className="scorecard__icao">{icao}</span>
      </h2>
      {body}
    </section>
  );
}

export interface ScorecardProps {
  cityIndex: CityIndex;
}

/**
 * The `#/scorecard` view: rolling-window MAE, bias and skill for raw NBM,
 * baseline and challenger, graded from the append-only prediction log.
 * National numbers come with the page; a city's file is fetched only when
 * that city is chosen.
 */
export function Scorecard({ cityIndex }: ScorecardProps): JSX.Element {
  const scorecardFetch = useFetchJson<ScorecardData>("scorecard.json");
  const [selected, setSelected] = useState<string | null>(null);

  if (scorecardFetch.status === "loading") {
    return <p role="status">Loading{"…"}</p>;
  }
  if (scorecardFetch.status === "error") {
    return <FetchError onRetry={scorecardFetch.retry} />;
  }

  const card = scorecardFetch.data;
  const selectedEntry = selected
    ? cityIndex.cities.find((city) => city.icao === selected)
    : undefined;

  return (
    <div className="scorecard">
      <h1 className="scorecard__title">Live scorecard</h1>

      {card.window === null ? (
        <p className="scorecard__empty">
          {card.log.rows === 0
            ? "No live predictions have been logged yet."
            : "Predictions are logged, but none has been compared with an observation yet."}
        </p>
      ) : (
        <>
          <WindowLine window={card.window} />
          <p className="scorecard__explainer">
            Bias = forecast {"−"} observed. Skill = 1 {"−"} MAE {"÷"} raw NBM&apos;s
            MAE on the same cases, so positive means better than raw. Intervals are{" "}
            {Math.round(card.intervals.level * 100)}% {card.intervals.method}{" "}
            intervals.
          </p>
          <p className="scorecard__note-standing">{card.intervals.note}</p>
          <h2 className="scorecard__section-title">National</h2>
          <ScorecardTable stats={card.national} windowDays={card.window.days} />
        </>
      )}

      <LogBlock log={card.log} />

      {card.window !== null && (
        <div className="scorecard__city-picker">
          <h2 className="scorecard__section-title">By city</h2>
          <CitySearch cities={cityIndex.cities} onNavigate={setSelected} />
          {selected && (
            <CityScorecard
              key={selected}
              icao={selected}
              label={selectedEntry?.label ?? selected}
            />
          )}
        </div>
      )}
    </div>
  );
}
