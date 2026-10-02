"""Step 2: build (once) and then reproduce a frozen arXiv corpus.

Modes
  build   no data/manifest.jsonl yet: query arXiv, filter, download, write manifest
  frozen  manifest exists: download exactly those papers/versions, verify checksums

Usage
  python scripts/download_corpus.py --dry-run   # preview the query, download nothing
  python scripts/download_corpus.py             # build (first time) or frozen (after)
  python scripts/download_corpus.py --rebuild   # build a NEW corpus (invalidates golden set!)

Interrupted? Just rerun: PDFs already on disk are reused, and the arXiv query
results are cached in data/raw/candidates.json so the API isn't queried again.
Rate-limited (HTTP 429/503)? Requests back off automatically (15 s -> 4 min).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import median

import arxiv
import httpx
import pymupdf
import requests
from tqdm import tqdm

from rag_eval.config import REPO_ROOT, load_config, resolve_path

MANIFEST = REPO_ROOT / "data" / "manifest.jsonl"
USER_AGENT = "rag-eval-arxiv/0.1 (personal research project; one-time corpus download)"


RETRYABLE_STATUS = {429, 500, 502, 503, 504}
BACKOFF_S = [15, 30, 60, 120, 240]  # exponential backoff between retries


class TooLarge(Exception):
    pass


def _status_of(err: Exception) -> int | None:
    if isinstance(err, arxiv.HTTPError):
        return err.status
    if isinstance(err, httpx.HTTPStatusError):
        return err.response.status_code
    return None


def with_backoff(fn, what: str):
    """Call fn(); on rate-limit / server / network errors, wait and retry.

    Retrying immediately after a 429 ("too many requests") makes things worse,
    so the wait doubles each time: 15 s, 30 s, 60 s, 120 s, 240 s.
    """
    for attempt in range(len(BACKOFF_S) + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            status = _status_of(e)
            transient = (
                status in RETRYABLE_STATUS
                or isinstance(e, (httpx.TransportError, requests.exceptions.ConnectionError,
                                  arxiv.UnexpectedEmptyPageError))
            )
            if not transient or attempt == len(BACKOFF_S):
                raise
            wait = BACKOFF_S[attempt]
            if isinstance(e, httpx.HTTPStatusError):
                retry_after = e.response.headers.get("retry-after", "")
                if retry_after.isdigit():
                    wait = max(wait, int(retry_after))
            reason = f"HTTP {status}" if status else type(e).__name__
            tqdm.write(f"  {what}: {reason}; waiting {wait}s (retry {attempt + 1}/{len(BACKOFF_S)})")
            time.sleep(wait)


# ---------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pdf_path(raw_dir: Path, arxiv_id: str) -> Path:
    return raw_dir / f"{arxiv_id.replace('/', '_')}.pdf"


def base_id(arxiv_id: str) -> str:
    """'2005.11401v4' -> '2005.11401' (version stripped, for de-duplication)."""
    return re.sub(r"v\d+$", "", arxiv_id)


def to_record(result: arxiv.Result, is_seed: bool) -> dict:
    sid = result.get_short_id()  # includes version, e.g. 2005.11401v4
    return {
        "arxiv_id": sid,
        "base_id": base_id(sid),
        "title": " ".join(result.title.split()),
        "authors": [a.name for a in result.authors],
        "published": result.published.date().isoformat(),
        "primary_category": result.primary_category,
        "categories": list(result.categories),
        "abstract": " ".join(result.summary.split()),
        "pdf_url": f"https://arxiv.org/pdf/{sid}",
        "seed": is_seed,
    }


def download_pdf(http: httpx.Client, url: str, dest: Path, max_mb: float) -> None:
    """Stream to a .part file, enforce a size cap, then atomically rename."""
    tmp = dest.with_suffix(".part")
    limit = max_mb * 1024 * 1024
    try:
        with http.stream("GET", url) as r:
            r.raise_for_status()
            ctype = r.headers.get("content-type", "")
            if "pdf" not in ctype:
                raise ValueError(f"not a PDF (content-type: {ctype})")
            size = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(1 << 16):
                    size += len(chunk)
                    if size > limit:
                        raise TooLarge(f"> {max_mb} MB")
                    f.write(chunk)
        tmp.rename(dest)
    finally:
        tmp.unlink(missing_ok=True)


def inspect_pdf(path: Path) -> tuple[int, int]:
    """Return (page_count, characters of text on the first two pages)."""
    with pymupdf.open(path) as doc:
        pages = doc.page_count
        text = "".join(doc[i].get_text() for i in range(min(2, pages)))
    return pages, len(text.strip())


def read_manifest() -> list[dict]:
    with open(MANIFEST, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_manifest(records: list[dict]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(MANIFEST)


# ---------------------------------------------------------------- arXiv query
def fetch_candidates(c: dict, cache_path: Path) -> list[dict]:
    """Seeds + relevance-ranked query results, de-duplicated. Cached after first success."""
    if cache_path.exists():
        records = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"Using cached candidates: {cache_path.name} ({len(records)} papers). "
              f"Delete it to re-query arXiv.")
        return records

    # num_retries=0: retries are handled by with_backoff, which waits much longer.
    api = arxiv.Client(page_size=c["api_page_size"], delay_seconds=c["api_delay_s"], num_retries=0)

    print(f"Fetching {len(c['seed_ids'])} seed papers...")
    seeds = with_backoff(lambda: list(api.results(arxiv.Search(id_list=c["seed_ids"]))), "seed request")

    query = f"({c['arxiv_query']}) AND submittedDate:[{c['date_from']} TO {c['date_to']}]"
    print(f"Querying arXiv (up to {c['candidate_pool']} results, {c['api_page_size']} per request):\n  {query}")
    pool: list[arxiv.Result] = []
    for offset in range(0, c["candidate_pool"], c["api_page_size"]):
        n = min(c["api_page_size"], c["candidate_pool"] - offset)
        page_search = arxiv.Search(
            query=query,
            max_results=offset + n,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        page = with_backoff(lambda: list(api.results(page_search, offset=offset)),
                            f"results {offset}-{offset + n}")
        pool.extend(page)
        print(f"  got results {offset}-{offset + len(page)}")
        if len(page) < n:  # no more matches
            break

    seen: set[str] = set()
    records: list[dict] = []
    for result, is_seed in [(r, True) for r in seeds] + [(r, False) for r in pool]:
        rec = to_record(result, is_seed)
        if rec["base_id"] in seen:
            continue
        seen.add(rec["base_id"])
        records.append(rec)
    print(f"  {len(seeds)} seeds + {len(pool)} query results -> {len(records)} unique candidates")

    cache_path.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    return records


# ---------------------------------------------------------------- modes
def build(cfg: dict, raw_dir: Path) -> list[dict]:
    c = cfg["corpus"]
    candidates = fetch_candidates(c, raw_dir / "candidates.json")
    accepted: list[dict] = []
    rejected: Counter = Counter()

    with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=90) as http:
        bar = tqdm(total=c["max_papers"], desc="accepted", unit="paper")
        for rec in candidates:
            if len(accepted) >= c["max_papers"]:
                break
            dest = pdf_path(raw_dir, rec["arxiv_id"])
            try:
                if not dest.exists():
                    with_backoff(lambda: download_pdf(http, rec["pdf_url"], dest, c["max_pdf_mb"]),
                                 f"download {rec['arxiv_id']}")
                    time.sleep(c["download_delay_s"])
                pages, n_chars = inspect_pdf(dest)
            except TooLarge:
                rejected["too large"] += 1
                continue
            except Exception as e:  # noqa: BLE001
                rejected["download/parse error"] += 1
                tqdm.write(f"  skip {rec['arxiv_id']}: {type(e).__name__}: {str(e)[:120]}")
                dest.unlink(missing_ok=True)
                time.sleep(c["download_delay_s"])
                continue

            if pages > c["max_pages"] and not rec["seed"]:
                rejected[f"> {c['max_pages']} pages"] += 1
                dest.unlink()
                continue
            if n_chars < c["min_text_chars"]:
                rejected["no extractable text"] += 1
                dest.unlink()
                continue

            rec.update(pages=pages, size_bytes=dest.stat().st_size, sha256=sha256_file(dest))
            accepted.append(rec)
            bar.update(1)
        bar.close()

    if len(accepted) < c["max_papers"]:
        print(f"WARNING: only {len(accepted)} papers accepted; consider raising candidate_pool.")

    # Remove PDFs left over from earlier interrupted runs that didn't make the cut.
    keep = {pdf_path(raw_dir, r["arxiv_id"]).name for r in accepted}
    stray = [p for p in raw_dir.glob("*.pdf") if p.name not in keep]
    for p in stray:
        p.unlink()

    write_manifest(accepted)
    print(f"\nWrote {MANIFEST.relative_to(REPO_ROOT)}  |  removed {len(stray)} stray PDFs")
    if rejected:
        print("Rejected: " + ", ".join(f"{k}={v}" for k, v in rejected.most_common()))
    return accepted


def reproduce(cfg: dict, raw_dir: Path) -> list[dict]:
    c = cfg["corpus"]
    records = read_manifest()
    print(f"Frozen mode: {len(records)} papers listed in {MANIFEST.relative_to(REPO_ROOT)}")
    mismatches, failures = [], []

    with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=90) as http:
        for rec in tqdm(records, desc="verifying", unit="paper"):
            dest = pdf_path(raw_dir, rec["arxiv_id"])
            if not dest.exists():
                try:
                    with_backoff(lambda: download_pdf(http, rec["pdf_url"], dest, max_mb=max(c["max_pdf_mb"], 50)),
                                 f"download {rec['arxiv_id']}")
                    time.sleep(c["download_delay_s"])
                except Exception as e:  # noqa: BLE001
                    failures.append(f"{rec['arxiv_id']}: {type(e).__name__}: {e}")
                    continue
            if sha256_file(dest) != rec["sha256"]:
                mismatches.append(rec["arxiv_id"])

    if mismatches:
        # arXiv occasionally re-renders PDFs from source; same version, same content,
        # different bytes. Text-level differences are what would actually matter.
        print(f"NOTE: {len(mismatches)} checksum mismatches (same arXiv version, re-rendered PDF): "
              f"{', '.join(mismatches[:5])}{' ...' if len(mismatches) > 5 else ''}")
    if failures:
        print(f"FAILED downloads ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
    return records


def summarize(records: list[dict], raw_dir: Path) -> None:
    if not records:
        print("No papers.")
        return
    pages = [r["pages"] for r in records]
    total_mb = sum(r["size_bytes"] for r in records) / 1e6
    years = Counter(r["published"][:4] for r in records)
    cats = Counter(r["primary_category"] for r in records)
    on_disk = sum(1 for r in records if pdf_path(raw_dir, r["arxiv_id"]).exists())

    print("\n=========== CORPUS SUMMARY ===========")
    print(f"papers:        {len(records)} (on disk: {on_disk}, seeds: {sum(r['seed'] for r in records)})")
    print(f"total size:    {total_mb:.1f} MB")
    print(f"pages:         total={sum(pages)}  median={median(pages):.0f}  min={min(pages)}  max={max(pages)}")
    print("years:         " + ", ".join(f"{y}={n}" for y, n in sorted(years.items())))
    print("primary cat:   " + ", ".join(f"{k}={v}" for k, v in cats.most_common(6)))
    print("======================================")


def dry_run(cfg: dict, raw_dir: Path) -> None:
    candidates = fetch_candidates(cfg["corpus"], raw_dir / "candidates.json")
    print("\nFirst 25 candidates (seeds first):")
    for r in candidates[:25]:
        tag = "[seed]" if r["seed"] else "      "
        print(f"  {tag} {r['arxiv_id']:<14} {r['published'][:4]}  {r['title'][:80]}")
    print(f"\nTotal unique candidates: {len(candidates)}. Nothing was downloaded.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="preview candidates, download nothing")
    parser.add_argument("--rebuild", action="store_true", help="ignore existing manifest and build a new corpus")
    args = parser.parse_args()

    cfg = load_config()
    raw_dir = resolve_path(cfg, "raw_dir")

    if args.dry_run:
        dry_run(cfg, raw_dir)
        return 0

    if MANIFEST.exists() and not args.rebuild:
        records = reproduce(cfg, raw_dir)
    else:
        if MANIFEST.exists():
            backup = MANIFEST.with_suffix(".jsonl.bak")
            shutil.copy(MANIFEST, backup)
            print(f"--rebuild: previous manifest backed up to {backup.relative_to(REPO_ROOT)}")
        (raw_dir / "candidates.json").unlink(missing_ok=True)  # re-query arXiv for a new corpus
        records = build(cfg, raw_dir)

    summarize(records, raw_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
