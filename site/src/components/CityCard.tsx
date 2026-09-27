import type { City, CitySummary, CityStat, Season, Variable } from "../types/export";
import {
  biasCellContent,
  formatTypicalMissLine,
  medianCellN,
  seasonWord,
} from "../lib/format";

const SEASON_ORDER: Season[] = ["DJF", "MAM", "JJA", "SON"];
const VARIABLE_LABELS: Record<Variable, string> = { max: "Highs", min: "Lows" };
// Lead-3 max never occurs in NBS guidance (docs/methodology.md, "Scope: lead
// 1-3, no lead-0 max, no lead-3 max"): the audit only ever grades minimums
// at lead 3, so that column is omitted rather than rendered empty.
const LEADS_BY_VARIABLE: Record<Variable, number[]> = { max: [1, 2], min: [1, 2, 3] };

function statKey(variable: Variable, leadDay: number, season: Season): string {
  return `${variable}-${leadDay}-${season}`;
}

interface BiasCellProps {
  stat: CityStat;
}

/**
 * One table cell. A significant cell is a bold primary line (e.g. "−1.1°F")
 * with the interval endpoints beneath it, smaller and muted. A
 * non-significant or too-few-days cell shows a muted glyph/word to sighted
 * users (`aria-hidden`) plus a visually-hidden span carrying the full
 * accessible name, so screen readers hear "no detectable bias" or "too few
 * days" rather than just "dash".
 */
function BiasCell({ stat }: BiasCellProps): JSX.Element {
  const content = biasCellContent(stat);
  if (content.kind === "significant") {
    return (
      <td className="city-card__cell city-card__cell--significant">
        <span className="city-card__cell-primary">{content.primary}</span>
        {content.secondary && (
          <span className="city-card__cell-secondary">{content.secondary}</span>
        )}
      </td>
    );
  }
  return (
    <td className="city-card__cell city-card__cell--muted">
      <span aria-hidden="true">{content.visible}</span>
      <span className="visually-hidden">{content.accessibleName}</span>
    </td>
  );
}

interface VariableTableProps {
  variable: Variable;
  statsByKey: Map<string, CityStat>;
}

function VariableTable({ variable, statsByKey }: VariableTableProps): JSX.Element {
  const leads = LEADS_BY_VARIABLE[variable];
  const cellsForCaption = Array.from(statsByKey.values()).filter(
    (s) => s.variable === variable
  );
  const captionDays = medianCellN(cellsForCaption.map((s) => s.n_dates));

  return (
    <>
      <div className="city-card__table-scroll">
        <table className="city-card__table">
          <caption>{VARIABLE_LABELS[variable]}</caption>
          <thead>
            <tr>
              <th scope="col">Season</th>
              {leads.map((lead) => (
                <th scope="col" key={lead}>
                  Day {lead}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {SEASON_ORDER.map((season) => (
              <tr key={season}>
                <th scope="row">{seasonWord(season)}</th>
                {leads.map((lead) => {
                  const s = statsByKey.get(statKey(variable, lead, season));
                  return s ? (
                    <BiasCell key={lead} stat={s} />
                  ) : (
                    <td key={lead}>no data</td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="city-card__table-caption">
        {"–"} no detectable bias · each cell {"≈"} {captionDays} days
      </p>
    </>
  );
}

export interface CityCardProps {
  city: City;
  summary: CitySummary;
}

/** The `#/city/:icao` view: label, summary sentence, and a Highs/Lows table. */
export function CityCard({ city, summary }: CityCardProps): JSX.Element {
  const statsByKey = new Map<string, CityStat>();
  for (const stat of city.stats) {
    statsByKey.set(statKey(stat.variable, stat.lead_day, stat.season), stat);
  }
  const typicalMissLine = city.typical_miss
    ? formatTypicalMissLine(city.typical_miss)
    : null;

  return (
    <article className="city-card">
      <h1 className="city-card__title">
        {city.label} <span className="city-card__icao">{city.icao}</span>
      </h1>

      <p
        className={
          summary.significant
            ? "city-card__summary city-card__summary--significant"
            : "city-card__summary"
        }
      >
        {summary.text}
      </p>

      {typicalMissLine && (
        <p className="city-card__typical-miss">{typicalMissLine}</p>
      )}

      <p className="city-card__explainer">
        Bias = forecast {"−"} observed. Warm means the forecast ran
        higher than what happened.
      </p>

      <VariableTable variable="max" statsByKey={statsByKey} />
      <VariableTable variable="min" statsByKey={statsByKey} />

      <p className="city-card__footnote">
        Lead-3 max is never shown: NBS guidance never carries a lead-3 max
        forecast, so at lead 3 the audit only grades minimums.
      </p>
    </article>
  );
}
