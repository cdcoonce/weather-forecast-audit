import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { CityIndexEntry } from "../types/export";
import { CitySearch } from "./CitySearch";

function entry(overrides: Partial<CityIndexEntry>): CityIndexEntry {
  return {
    icao: "KPHX",
    label: "PHOENIX/SKY HARBOR, AZ",
    lat: 33.4,
    lon: -112.0,
    climate_region: "Southwest",
    summary: { text: "Not enough data yet.", significant: false },
    ...overrides,
  };
}

const CITIES: CityIndexEntry[] = [
  entry({ icao: "KPHX", label: "PHOENIX/SKY HARBOR, AZ" }),
  entry({ icao: "KPDX", label: "PORTLAND INTL, OR" }),
  entry({ icao: "KBOS", label: "BOSTON LOGAN INTL, MA" }),
];

describe("CitySearch", () => {
  it("filters results by typing (ICAO or label, case-insensitive)", async () => {
    const user = userEvent.setup();
    render(<CitySearch cities={CITIES} />);

    const input = screen.getByRole("combobox");
    await user.type(input, "phoenix");

    expect(screen.getByRole("option", { name: /PHOENIX/i })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: /PORTLAND/i })).not.toBeInTheDocument();
  });

  it("moves the active option with ArrowDown and navigates on Enter", async () => {
    const onNavigate = vi.fn();
    const user = userEvent.setup();
    render(<CitySearch cities={CITIES} onNavigate={onNavigate} />);

    const input = screen.getByRole("combobox");
    await user.type(input, "k");
    await user.keyboard("{ArrowDown}");

    const activeId = input.getAttribute("aria-activedescendant");
    expect(activeId).toBeTruthy();
    expect(document.getElementById(activeId!)).toHaveAttribute("aria-selected", "true");

    await user.keyboard("{Enter}");
    expect(onNavigate).toHaveBeenCalledTimes(1);
    expect(onNavigate).toHaveBeenCalledWith(CITIES[0].icao);
  });

  it("Escape clears the query and closes the listbox", async () => {
    const user = userEvent.setup();
    render(<CitySearch cities={CITIES} />);

    const input = screen.getByRole("combobox");
    await user.type(input, "port");
    expect(screen.getByRole("listbox")).toBeVisible();

    await user.keyboard("{Escape}");
    expect(input).toHaveValue("");
    expect(screen.getByRole("listbox", { hidden: true })).not.toBeVisible();
  });

  it("caps results at 8", async () => {
    const many: CityIndexEntry[] = Array.from({ length: 20 }, (_, i) =>
      entry({ icao: `K${String(i).padStart(3, "0")}`, label: `STATION ${i}` })
    );
    const user = userEvent.setup();
    render(<CitySearch cities={many} />);

    await user.type(screen.getByRole("combobox"), "station");
    expect(screen.getAllByRole("option")).toHaveLength(8);
  });

  it("sets aria-expanded and aria-controls wired to the listbox", async () => {
    const user = userEvent.setup();
    render(<CitySearch cities={CITIES} />);
    const input = screen.getByRole("combobox");
    expect(input).toHaveAttribute("aria-expanded", "false");

    await user.type(input, "bos");
    expect(input).toHaveAttribute("aria-expanded", "true");
    const listboxId = input.getAttribute("aria-controls");
    expect(listboxId).toBeTruthy();
    expect(document.getElementById(listboxId!)).toHaveAttribute("role", "listbox");
  });
});
