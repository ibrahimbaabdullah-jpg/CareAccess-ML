"""Bounded CSV decoding before any appointment table is allocated.

The four supported fields cannot contain line breaks. Checking each short
physical record before csv.reader also prevents a small, extremely wide file
from allocating millions of columns in a general-purpose table parser.
"""

import csv
import io

import pandas as pd

from careaccess.validation import INPUT_COLUMNS, MAX_ROWS

MAX_BYTES = 5 * 1024 * 1024
MAX_CELL_CHARS = 128
# Allow every character to be quoted/escaped, delimiters and a CRLF terminator.
MAX_RECORD_CHARS = len(INPUT_COLUMNS) * (2 * MAX_CELL_CHARS + 3) + 2


def read_appointments_csv(payload: bytes) -> pd.DataFrame:
    """Read UTF-8 bytes with bounded records and an exact, unique column set.

    Values remain strings so temporary references and invalid tokens are not
    silently converted. Semantic validation happens before model inference.
    """
    if not isinstance(payload, bytes):
        raise ValueError("A UTF-8 CSV file is required.")
    if len(payload) > MAX_BYTES:
        raise ValueError("The file is larger than 5 MB. Please use a smaller schedule.")
    try:
        text = payload.decode("utf-8-sig", errors="strict")
    except UnicodeError as error:
        raise ValueError("This file could not be read as a UTF-8 CSV. Use the downloadable empty template.") from error
    if "\x00" in text:
        raise ValueError("The CSV contains unsupported NUL characters.")
    stream = io.StringIO(text, newline="")

    def read_record():
        line = stream.readline(MAX_RECORD_CHARS + 1)
        if not line:
            return None
        if len(line) > MAX_RECORD_CHARS:
            raise ValueError("A CSV record is too long. Use only the four template columns and short values.")
        try:
            values = next(csv.reader([line], strict=True))
        except csv.Error as error:
            raise ValueError(
                "Malformed CSV. Use one appointment per line, without line breaks inside fields."
            ) from error
        if len(values) > len(INPUT_COLUMNS):
            raise ValueError("The CSV contains too many columns. Use only the four template columns.")
        if any(len(value) > MAX_CELL_CHARS for value in values):
            raise ValueError(f"CSV fields must be no longer than {MAX_CELL_CHARS} characters.")
        return values

    header = read_record()
    if header is None or len(header) != len(INPUT_COLUMNS) or set(header) != set(INPUT_COLUMNS):
        raise ValueError("Required unique CSV columns: " + ", ".join(INPUT_COLUMNS) + ".")
    rows = []
    while (values := read_record()) is not None:
        if not values:  # As in pandas, ignore empty physical lines.
            continue
        if len(values) != len(header):
            raise ValueError("Each appointment row must contain exactly four CSV fields.")
        if len(rows) >= MAX_ROWS:
            raise ValueError("The schedule exceeds 5,000 rows. Please split it into smaller files.")
        rows.append(values)
    return pd.DataFrame(rows, columns=header, dtype=str)
