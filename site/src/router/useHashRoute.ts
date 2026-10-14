import { useEffect, useState } from "react";

export type Route =
  | { name: "home" }
  | { name: "city"; icao: string }
  | { name: "scorecard" }
  | { name: "not-found" };

export function parseHash(hash: string): Route {
  const path = hash.replace(/^#/, "") || "/";
  if (path === "/" || path === "") return { name: "home" };
  if (path === "/scorecard") return { name: "scorecard" };
  const match = path.match(/^\/city\/([A-Za-z0-9]+)$/);
  if (match) return { name: "city", icao: match[1].toUpperCase() };
  return { name: "not-found" };
}

/** `#/`, `#/city/KPHX` and `#/scorecard` hash routing -- no router library. */
export function useHashRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));

  useEffect(() => {
    const onHashChange = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  return route;
}
