import assert from "node:assert/strict";
import test from "node:test";

import { buildCsv, escapeCsvCell } from "./csv-export.mjs";

test("CSV cells quote delimiters and embedded quotes", () => {
  assert.equal(escapeCsvCell('FastAPI, "REST"'), '"FastAPI, ""REST"""');
});

test("CSV cells neutralize formula prefixes after leading whitespace", () => {
  for (const value of ["=1+1", "  +cmd", "\t-HYPERLINK()", "@SUM(A1:A2)"]) {
    assert.equal(escapeCsvCell(value).startsWith("\"'"), true);
  }
});

test("CSV keeps only explicitly supplied export columns and rows", () => {
  assert.equal(
    buildCsv([
      ["candidate_identifier", "score"],
      ["candidate-01", 72.5],
    ]),
    '"candidate_identifier","score"\r\n"candidate-01","72.5"',
  );
});
