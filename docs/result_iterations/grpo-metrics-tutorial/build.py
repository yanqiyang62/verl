"""Build a portable, offline HTML tutorial; only Python standard library needed."""
import base64
import hashlib
import html as html_module
import json
from pathlib import Path
from catalog import CATALOG, GROUPS, SOURCES

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent


def main():
    snapshot = json.loads((HERE / "snapshot.json").read_text())
    expected = {m["key"] for m in snapshot["inventory"]}
    assert expected == CATALOG.keys(), (expected - CATALOG.keys(), CATALOG.keys() - expected)
    observed = {s["key"]: s for s in snapshot["series"]}
    assert expected == observed.keys()
    sources = {}
    for key, (path, start, end) in SOURCES.items():
        file = Path(path) if path.startswith("/") else ROOT / path
        raw = file.read_text()
        sources[key] = dict(path=path, start=start, end=end,
                            sha256=hashlib.sha256(raw.encode()).hexdigest(),
                            code="\n".join(f"{i}: {line}" for i, line in enumerate(raw.splitlines(), 1) if start <= i <= end))
    metrics = []
    for key, description in CATALOG.items():
        assert all(s in sources for s in description["source"])
        assert all(k in CATALOG for k in description["related"])
        points = sorted(observed[key]["metrics"], key=lambda x: x["index"])
        assert points and len({p["index"] for p in points}) == len(points), key
        metrics.append({**description, "points": points})
    data = dict(metrics=metrics, sources=sources, groups=GROUPS,
                meta={k: snapshot[k] for k in ["fetchedAt", "url", "name", "state", "run", "visibility"]})
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    html = (HERE / "template.html").read_text().replace("__TUTORIAL_DATA__", payload)
    html = html.replace("__FONT_LICENSE__", html_module.escape((HERE / "FONT-LICENSE.txt").read_text()))
    html = html.replace("__FONT_DATA__", base64.b64encode((HERE / "noto-sans-sc-subset.woff2").read_bytes()).decode())
    (HERE / "index.html").write_text(html, encoding="utf-8")
    print(f"Built {HERE / 'index.html'}: {len(metrics)} metrics, {len(sources)} source excerpts, {len(html.encode()):,} bytes")


if __name__ == "__main__":
    main()
