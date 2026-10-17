export function escapeCsvCell(value) {
  const text = value == null ? "" : String(value);
  const safe = /^[\s]*[=+\-@]/.test(text) ? `'${text}` : text;
  return `"${safe.replaceAll('"', '""')}"`;
}

export function buildCsv(rows) {
  return rows
    .map((row) => row.map(escapeCsvCell).join(","))
    .join("\r\n");
}
