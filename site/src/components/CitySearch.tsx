import { useId, useMemo, useState } from "react";
import type { KeyboardEvent } from "react";
import type { CityIndexEntry } from "../types/export";

const MAX_RESULTS = 8;

function defaultNavigate(icao: string): void {
  window.location.hash = `#/city/${icao}`;
}

export interface CitySearchProps {
  cities: CityIndexEntry[];
  onNavigate?: (icao: string) => void;
}

/**
 * An ARIA APG "combobox with listbox popup" (editable, list autocomplete):
 * https://www.w3.org/WAI/ARIA/apg/patterns/combobox/
 *
 * Filters `city_index.json` by ICAO or label, case-insensitive, capped at
 * `MAX_RESULTS`. Arrow keys move the active option (`aria-activedescendant`,
 * never real focus movement); Enter navigates to the active option or the
 * only match; Escape clears the query and closes the listbox.
 */
export function CitySearch({ cities, onNavigate }: CitySearchProps): JSX.Element {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const baseId = useId();
  const listboxId = `${baseId}-listbox`;
  const navigate = onNavigate ?? defaultNavigate;

  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    return cities
      .filter(
        (city) =>
          city.icao.toLowerCase().includes(q) || city.label.toLowerCase().includes(q)
      )
      .slice(0, MAX_RESULTS);
  }, [cities, query]);

  const showListbox = open && results.length > 0;
  const optionId = (icao: string) => `${listboxId}-option-${icao}`;

  function reset(): void {
    setQuery("");
    setOpen(false);
    setActiveIndex(-1);
  }

  function select(icao: string): void {
    navigate(icao);
    reset();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLInputElement>): void {
    if (event.key === "ArrowDown") {
      if (results.length === 0) return;
      event.preventDefault();
      setOpen(true);
      setActiveIndex((i) => (i + 1) % results.length);
    } else if (event.key === "ArrowUp") {
      if (results.length === 0) return;
      event.preventDefault();
      setOpen(true);
      setActiveIndex((i) => (i - 1 + results.length) % results.length);
    } else if (event.key === "Enter") {
      const target = activeIndex >= 0 ? results[activeIndex] : results[0];
      if (open && target) {
        event.preventDefault();
        select(target.icao);
      }
    } else if (event.key === "Escape") {
      event.preventDefault();
      reset();
    }
  }

  const activeOption = activeIndex >= 0 ? results[activeIndex] : undefined;

  return (
    <div className="city-search">
      <label htmlFor={baseId} className="city-search__label">
        Search cities
      </label>
      <input
        id={baseId}
        role="combobox"
        type="text"
        autoComplete="off"
        aria-expanded={showListbox}
        aria-controls={listboxId}
        aria-autocomplete="list"
        aria-activedescendant={activeOption ? optionId(activeOption.icao) : undefined}
        className="city-search__input"
        placeholder="Search by city or ICAO (e.g. Phoenix or KPHX)"
        value={query}
        onChange={(event) => {
          setQuery(event.target.value);
          setOpen(true);
          setActiveIndex(-1);
        }}
        onFocus={() => query && setOpen(true)}
        onKeyDown={handleKeyDown}
      />
      <ul
        id={listboxId}
        role="listbox"
        aria-label="City results"
        className="city-search__listbox"
        hidden={!showListbox}
      >
        {results.map((city, index) => (
          <li
            key={city.icao}
            id={optionId(city.icao)}
            role="option"
            aria-selected={index === activeIndex}
            className={
              index === activeIndex
                ? "city-search__option city-search__option--active"
                : "city-search__option"
            }
            onMouseDown={(event) => {
              event.preventDefault();
              select(city.icao);
            }}
          >
            <span className="city-search__option-label">{city.label}</span>
            <span className="city-search__option-icao">{city.icao}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
