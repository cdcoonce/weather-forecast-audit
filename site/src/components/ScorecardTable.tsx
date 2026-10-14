import type { ScorecardSource, ScorecardStat, Variable } from "../types/export";
import {
  formatInterval,
  formatMae,
  formatSignedFixed,
  formatSkillPercent,
} from "../lib/format";

const VARIABLE_LABELS: Record<Variable, string> = { max: "Highs", min: "Lows" };
const VARIABLES: Variable[] = ["max", "min"];
const SOURCE_ORDER: ScorecardSource[] = ["raw_nbm", "baseline", "challenger"];
const SOURCE_LABELS: Record<ScorecardSource, string> = {
  raw_nbm: "Raw NBM",
  baseline: "Baseline",
  challenger: "Challenger",
};

const WHISKER_WIDTH = 120;
const WHISKER_HEIGHT = 16;

/** The wording a cell group carries when `min_sample_flag` is set. */
export function approximateNote(windowDays: number): string {
  return (
    `The interval is approximate: the ${windowDays}-day window is too ` +
    "short for a significance call."
  );
}

interface WhiskerProps {
  stat: ScorecardStat;
  domain: [number, number];
}

/**
 * A bias interval as a small inline SVG: a line from lo to hi, a dot at the
 * point estimate, and a tick marking zero. The x scale is shared across one
 * variable's table so rows are comparable. It is a picture of the numbers
 * beside it, never the only place they appear.
 */
function BiasWhisker({ stat, domain }: WhiskerProps): JSX.Element | null {
  if (stat.bias_lo_f == null || stat.bias_hi_f == null) return null;
  const [min, max] = domain;
  const span = max - min || 1;
  const x = (value: number) => ((value - min) / span) * (WHISKER_WIDTH - 8) + 4;
  const mid = WHISKER_HEIGHT / 2;
  const label =
    `bias interval ${formatSignedFixed(stat.bias_lo_f)} to ` +
    `${formatSignedFixed(stat.bias_hi_f)}, zero marked`;
  return (
    <svg
      className="scorecard__whisker"
      role="img"
      aria-label={label}
      width={WHISKER_WIDTH}
      height={WHISKER_HEIGHT}
      viewBox={`0 0 ${WHISKER_WIDTH} ${WHISKER_HEIGHT}`}
    >
      <line
        className="scorecard__whisker-zero"
        x1={x(0)}
        x2={x(0)}
        y1={1}
        y2={WHISKER_HEIGHT - 1}
      />
      <line
        className="scorecard__whisker-span"
        x1={x(stat.bias_lo_f)}
        x2={x(stat.bias_hi_f)}
        y1={mid}
        y2={mid}
      />
      <circle className="scorecard__whisker-point" cx={x(stat.bias_f)} cy={mid} r={3} />
    </svg>
  );
}

function biasDomain(stats: ScorecardStat[]): [number, number] {
  const values = [0];
  for (const s of stats) {
    values.push(s.bias_f);
    if (s.bias_lo_f != null) values.push(s.bias_lo_f);
    if (s.bias_hi_f != null) values.push(s.bias_hi_f);
  }
  return [Math.min(...values), Math.max(...values)];
}

function NoInterval(): JSX.Element {
  return (
    <span className="scorecard__cell-secondary">
      no interval (too few dates to estimate one)
    </span>
  );
}

function StatRow({
  stat,
  domain,
}: {
  stat: ScorecardStat;
  domain: [number, number];
}): JSX.Element {
  const maeInterval = formatInterval(stat.mae_lo_f, stat.mae_hi_f, formatMae);
  const biasInterval = formatInterval(
    stat.bias_lo_f,
    stat.bias_hi_f,
    formatSignedFixed
  );
  const skillText = formatSkillPercent(stat.skill);
  const skillInterval = formatInterval(stat.skill_lo, stat.skill_hi, (v) => {
    return formatSkillPercent(v) ?? "";
  });
  // A claim that the interval excludes zero is only ever made when the
  // window is long enough to support one; a flagged group never makes it.
  const excludesZero = !stat.min_sample_flag && !stat.no_detectable_bias;
  const rowClass = stat.min_sample_flag
    ? "scorecard__row scorecard__row--approximate"
    : "scorecard__row";

  return (
    <tr className={rowClass}>
      <th scope="row" className="scorecard__source">
        {SOURCE_LABELS[stat.source]}
      </th>
      <td className="scorecard__cell">
        <span className="scorecard__cell-primary">{formatMae(stat.mae_f)}</span>
        {maeInterval ? (
          <span className="scorecard__cell-secondary">{maeInterval}</span>
        ) : (
          <NoInterval />
        )}
      </td>
      <td
        className={
          excludesZero
            ? "scorecard__cell scorecard__cell--significant"
            : "scorecard__cell"
        }
      >
        <span className="scorecard__cell-primary">
          {formatSignedFixed(stat.bias_f)}
        </span>
        {biasInterval ? (
          <span className="scorecard__cell-secondary">{biasInterval}</span>
        ) : (
          <NoInterval />
        )}
        {excludesZero && (
          <span className="scorecard__cell-tag">interval excludes zero</span>
        )}
        <BiasWhisker stat={stat} domain={domain} />
      </td>
      <td className="scorecard__cell">
        {stat.source === "raw_nbm" ? (
          <span className="scorecard__cell-secondary">reference</span>
        ) : skillText ? (
          <>
            <span className="scorecard__cell-primary">{skillText}</span>
            {skillInterval ? (
              <span className="scorecard__cell-secondary">{skillInterval}</span>
            ) : (
              <NoInterval />
            )}
          </>
        ) : (
          <span className="scorecard__cell-secondary">not available</span>
        )}
      </td>
      <td className="scorecard__cell scorecard__cell--count">{stat.n}</td>
    </tr>
  );
}

export interface ScorecardTableProps {
  stats: ScorecardStat[];
  windowDays: number;
}

/**
 * The live scorecard as one table per variable. Each lead is a row group of
 * three rows (raw NBM, baseline, challenger) with MAE, bias and skill versus
 * raw, each with its interval. A group with `min_sample_flag` carries a
 * standing note, and is never worded or coloured as a significant result.
 */
export function ScorecardTable({
  stats,
  windowDays,
}: ScorecardTableProps): JSX.Element {
  return (
    <>
      {VARIABLES.map((variable) => {
        const forVariable = stats.filter((s) => s.variable === variable);
        if (forVariable.length === 0) return null;
        const domain = biasDomain(forVariable);
        const leads = Array.from(new Set(forVariable.map((s) => s.lead_day))).sort(
          (a, b) => a - b
        );
        return (
          <div className="scorecard__table-scroll" key={variable}>
            <table className="scorecard__table">
              <caption>{VARIABLE_LABELS[variable]}</caption>
              <thead>
                <tr>
                  <th scope="col">Source</th>
                  <th scope="col">MAE (°F)</th>
                  <th scope="col">Bias (°F)</th>
                  <th scope="col">Skill vs raw</th>
                  <th scope="col">Scored</th>
                </tr>
              </thead>
              {leads.map((lead) => {
                const group = SOURCE_ORDER.map((source) =>
                  forVariable.find((s) => s.lead_day === lead && s.source === source)
                ).filter((s): s is ScorecardStat => s !== undefined);
                const flagged = group.some((s) => s.min_sample_flag);
                return (
                  <tbody key={lead}>
                    <tr>
                      <th scope="rowgroup" colSpan={5} className="scorecard__lead">
                        Day {lead}
                      </th>
                    </tr>
                    {group.map((s) => (
                      <StatRow key={s.source} stat={s} domain={domain} />
                    ))}
                    {flagged && (
                      <tr>
                        <td colSpan={5} className="scorecard__note">
                          {approximateNote(windowDays)}
                        </td>
                      </tr>
                    )}
                  </tbody>
                );
              })}
            </table>
          </div>
        );
      })}
    </>
  );
}
