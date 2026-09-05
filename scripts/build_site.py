"""Wrap submission.html into a standalone page for static hosting.

    python scripts/build_site.py

``submission.html`` is authored as a fragment: it opens with <title>, <link> and
<style> and has no <!doctype>, <html>, <head> or <body>, because the artifact
runtime supplies those at publish time. A static host does not, so a browser
would fall into quirks mode and the viewport meta would be missing entirely --
which on a phone means the page renders at desktop width and everything is tiny.

This wraps the same file, unchanged, into a complete document at
``site/index.html``. One source, two outputs, no copy that can drift.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission.html"
OUT = ROOT / "site" / "index.html"

DESCRIPTION = (
    "Four approaches to extracting 15 fields from semi-structured healthcare "
    "documents -- rules, a fine-tuned DistilBERT, a local 3B LLM and a frontier "
    "model -- compared on one 90-document human-verified gold set."
)

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="{description}">
<meta name="color-scheme" content="light">

<meta property="og:type" content="article">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{description}">
<meta name="twitter:card" content="summary">

<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><text y='13' font-size='13'>&#128224;</text></svg>">

<style>
  html{{-webkit-text-size-adjust:100%}}
  img{{max-width:100%;height:auto}}
  [hidden]{{display:none!important}}
</style>
{fragment}
</body>
</html>
"""


def main() -> int:
    if not SOURCE.exists():
        print(f"missing {SOURCE}", file=sys.stderr)
        return 1

    fragment = SOURCE.read_text(encoding="utf-8")
    match = re.search(r"<title>(.*?)</title>", fragment, re.S)
    title = match.group(1).strip() if match else "Document Intelligence"

    # The fragment carries its own <title>, which belongs in <head>. Everything
    # after it is body content; the <link> and <style> that follow are valid
    # there too, so the split is safe to make at the title.
    if match:
        head_bits = fragment[: match.end()]
        rest = fragment[match.end():]
    else:
        head_bits, rest = "", fragment

    page = TEMPLATE.format(
        title=title,
        description=DESCRIPTION,
        fragment=head_bits + "\n</head>\n<body>" + rest,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(ROOT)}  ({kb:.0f} KB)  title: {title!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
