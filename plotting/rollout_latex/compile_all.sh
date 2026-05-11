#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEX_DIR="$ROOT/tex"
PDF_DIR="$ROOT/pdf"

mkdir -p "$PDF_DIR"
cd "$TEX_DIR"

if command -v latexmk >/dev/null 2>&1; then
  for tex in *.tex; do
    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$PDF_DIR" "$tex"
  done
elif command -v pdflatex >/dev/null 2>&1; then
  for tex in *.tex; do
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$PDF_DIR" "$tex"
  done
else
  echo "Neither latexmk nor pdflatex was found in PATH." >&2
  exit 1
fi
