export type CsvValue = string | number | null | undefined;

export function escapeCsvCell(value: CsvValue): string;
export function buildCsv(rows: CsvValue[][]): string;
