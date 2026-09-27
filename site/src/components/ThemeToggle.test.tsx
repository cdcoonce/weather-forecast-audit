import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ThemeToggle } from "./ThemeToggle";

const LABEL = "Dark mode";

describe("ThemeToggle", () => {
  beforeEach(() => {
    document.documentElement.removeAttribute("data-theme");
  });

  afterEach(() => {
    document.documentElement.removeAttribute("data-theme");
    vi.restoreAllMocks();
  });

  it('has a fixed "Dark mode" label regardless of state, and aria-pressed flips on click', async () => {
    const user = userEvent.setup();
    render(<ThemeToggle />);

    const button = screen.getByRole("button", { name: LABEL });
    const initiallyDark = button.getAttribute("aria-pressed") === "true";
    expect(document.documentElement.getAttribute("data-theme")).toBe(
      initiallyDark ? "dark" : null
    );

    await user.click(button);

    const nowDark = !initiallyDark;
    expect(button).toHaveAttribute("aria-pressed", String(nowDark));
    expect(button).toHaveTextContent(LABEL); // label never changes
    expect(document.documentElement.getAttribute("data-theme")).toBe(
      nowDark ? "dark" : "light"
    );
  });

  it("sets data-theme on <html> and keeps working when localStorage throws on every read and write", async () => {
    const getItem = vi.spyOn(window.localStorage.__proto__, "getItem").mockImplementation(
      () => {
        throw new Error("storage disabled");
      }
    );
    const setItem = vi.spyOn(window.localStorage.__proto__, "setItem").mockImplementation(
      () => {
        throw new Error("storage disabled");
      }
    );

    const user = userEvent.setup();
    expect(() => render(<ThemeToggle />)).not.toThrow();

    const button = screen.getByRole("button", { name: LABEL });
    await expect(user.click(button)).resolves.not.toThrow();
    expect(document.documentElement.getAttribute("data-theme")).toMatch(/^(light|dark)$/);

    getItem.mockRestore();
    setItem.mockRestore();
  });

  it("has no icon, only the fixed text label", () => {
    render(<ThemeToggle />);
    const button = screen.getByRole("button");
    expect(button.textContent).toBe(LABEL);
  });

  it("reflects the pressed (dark) state via aria-pressed, independent of the label", async () => {
    const user = userEvent.setup();
    render(<ThemeToggle />);
    const button = screen.getByRole("button", { name: LABEL });

    if (button.getAttribute("aria-pressed") !== "true") {
      await user.click(button);
    }
    expect(button).toHaveAttribute("aria-pressed", "true");
    expect(button).toHaveTextContent(LABEL);
  });
});
