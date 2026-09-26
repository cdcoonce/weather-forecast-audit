"""Calendar-month chunking shared by the IEM clients (one request per month)."""

from datetime import date, timedelta


def month_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Split [start, end] into calendar-month chunks, clipped to the range."""
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        if cursor.month == 12:
            next_month_start = date(cursor.year + 1, 1, 1)
        else:
            next_month_start = date(cursor.year, cursor.month + 1, 1)
        chunk_end = min(end, next_month_start - timedelta(days=1))
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def date_range(start: date, end: date) -> list[date]:
    """Every calendar date in [start, end], inclusive."""
    days = (end - start).days
    return [start + timedelta(days=offset) for offset in range(days + 1)]
