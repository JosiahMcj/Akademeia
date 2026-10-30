#!/bin/bash
# Render paper/PAPER.md to paper/PAPER.pdf with the figures inline.
# Needs pandoc and a Chrome or Chromium binary (CHROME overrides the path).
set -euo pipefail
cd "$(dirname "$0")"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
pandoc PAPER.md --from gfm --to html5 --standalone --embed-resources \
  --resource-path=.:.. --css paper.css --metadata pagetitle="$(head -1 PAPER.md | sed "s/^# //")" -o PAPER.html
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf=PAPER.pdf "file://$PWD/PAPER.html" 2>/dev/null
rm -f PAPER.html
echo "wrote $PWD/PAPER.pdf"
