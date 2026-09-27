export function NotFound({ icao }: { icao?: string }): JSX.Element {
  return (
    <div className="not-found">
      <h1>City not found</h1>
      <p>{icao ? `We don't have data for "${icao}".` : "That page doesn't exist."}</p>
      <a href="#/">Back to search</a>
    </div>
  );
}
