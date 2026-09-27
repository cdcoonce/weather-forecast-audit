import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ThemeToggle } from "./ThemeToggle";

describe("ThemeToggle", () => {
  beforeEach(() => {
    document.documentElement.removeAttribute("data-theme");
  });

  afterEach(() => {
    document.documentElement.removeAttribute("data-theme");
    vi.restoreAllMocks();
  });

  it("sets data-theme on <html> and flips aria-pressed/label when clicked", async () => {
    const user = userEvent.setup();
    render(<ThemeToggle />);

    const button = screen.getByRole("button");
    const initiallyDark = button.getAttribute("aria-pressed") === "true";
    expect(document.documentElement.getAttribute("data-theme")).toBe(
      initiallyDark ? "dark" : null
    );

    await user.click(button);

    const nowDark = !initiallyDark;
    expect(button).toHaveAttribute("aria-pressed", String(nowDark));
    expect(button).toHaveTextContent(nowDark ? "Light" : "Dark");
    expect(document.documentElement.getAttribute("data-theme")).toBe(
      nowDark ? "dark" : "light"
    );
  });

  it("keeps working when localStorage throws on every read and write", async () => {
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

    const button = screen.getByRole("button");
    await expect(user.click(button)).resolves.not.toThrow();
    expect(document.documentElement.getAttribute("data-theme")).toMatch(/^(light|dark)$/);

    getItem.mockRestore();
    setItem.mockRestore();
  });

  it("has no icon, only a text label", () => {
    render(<ThemeToggle />);
    const button = screen.getByRole("button");
    expect(button.textContent).toMatch(/^(Light|Dark)$/);
  });
});
