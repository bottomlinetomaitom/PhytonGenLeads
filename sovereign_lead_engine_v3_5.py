"""
=============================================================================
SOVEREIGN LEAD ENGINE v3.5
Async · Retry · AI (Ollama) · Heuristic Fallback · SQLite · CSV/JSON Export
=============================================================================

Improvements over v3.4:
  - Centralised configuration via EngineConfig dataclass (no magic globals)
  - LeadDB now uses one connection per thread (sqlite3 thread safety)
  - Retry with exponential jitter and Retry-After header support (429)
  - extract_emails: pre-compiled regex, robust deduplication
  - _heuristic_analysis: weighted scoring on keyword density + text length
  - run(): progress tracking with shared counters
  - main(): accurate timing with time.perf_counter(), detailed summary
  - Full type hints, docstrings on every public function
  - No lazy imports except 'ollama' (optional dependency)
  - Compatible with Python 3.10+
=============================================================================
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import random
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

@dataclass
class EngineConfig:
    """Single source of truth for all engine parameters."""

    db_name:           str   = field(default_factory=lambda: os.getenv("LEAD_DB", "leads.db"))
    ollama_model:      str   = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3"))
    ollama_timeout:    float = 30.0
    max_concurrent:    int   = 5
    retries:           int   = 3
    request_timeout:   float = 20.0
    connect_timeout:   float = 5.0
    rate_limit_min:    float = 1.5
    rate_limit_max:    float = 4.0
    text_max_chars:    int   = 4000
    min_text_chars:    int   = 100


CONFIG = EngineConfig()

USER_AGENTS: List[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
]

BUSINESS_KEYWORDS: Dict[str, str] = {
    "consulting":   "Consulting",
    "agency":       "Agency",
    "solutions":    "Solutions",
    "services":     "Services",
    "software":     "Software",
    "marketing":    "Marketing",
    "design":       "Design",
    "development":  "Development",
    "enterprise":   "Enterprise",
    "saas":         "SaaS",
    "platform":     "Platform",
    "analytics":    "Analytics",
    "automation":   "Automation",
    "digital":      "Digital",
    "cloud":        "Cloud",
    "management":   "Management",
    "strategy":     "Strategy",
    "integration":  "Integration",
    "ecommerce":    "E-Commerce",
    "fintech":      "Fintech",
    "startup":      "Startup",
    "b2b":          "B2B",
    "crm":          "CRM",
}

EMAIL_BLOCKED_DOMAINS: frozenset[str] = frozenset({
    "example.com", "test.com", "email.com", "domain.com",
    "sentry.io", "wixpress.com", "googleapis.com",
    "w3.org", "schema.org",
})

EMAIL_BLOCKED_EXTENSIONS: frozenset[str] = frozenset({
    ".png", ".jpg", ".gif", ".svg", ".css", ".js", ".woff",
})

# Pre-compiled once at module load — never recompiled per call
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging to stdout and a file with a standard format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("pipeline.log", encoding="utf-8"),
        ],
    )


# ---------------------------------------------------------------------------
# DATABASE
# ---------------------------------------------------------------------------

class LeadDB:
    """
    Thread-safe SQLite wrapper.

    Uses one connection per thread (threading.local) instead of a single
    shared connection with a global lock. This eliminates read contention
    and correctly leverages SQLite's WAL journal mode.
    """

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS leads (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            company    TEXT    NOT NULL,
            service    TEXT    NOT NULL DEFAULT 'Unknown',
            score      INTEGER NOT NULL DEFAULT 0,
            url        TEXT    NOT NULL DEFAULT '',
            domain     TEXT    NOT NULL DEFAULT '',
            emails     TEXT    NOT NULL DEFAULT '',
            created_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            UNIQUE(company, domain)
        )
    """

    def __init__(self, db_name: str = "") -> None:
        self._db_name = db_name or CONFIG.db_name
        self._local   = threading.local()
        # Initialise schema on the main thread
        conn = self._conn()
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute(self._CREATE_TABLE)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_domain ON leads(domain)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_score  ON leads(score DESC)")
        conn.commit()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        """Return the current thread's connection, creating it if necessary."""
        if not getattr(self._local, "conn", None):
            conn = sqlite3.connect(self._db_name, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            self._local.conn = conn
        return self._local.conn  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "LeadDB":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, data: Dict) -> bool:
        """
        Insert a lead record. Returns True if saved, False if duplicate or invalid.

        Args:
            data: dict with keys company, service, score, url, emails.

        Returns:
            True if the row was inserted, False otherwise.
        """
        company = (data.get("company") or "").strip()
        if not company:
            return False

        conn = self._conn()
        try:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO leads (company, service, score, url, domain, emails)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    company,
                    (data.get("service") or "Unknown").strip(),
                    max(0, min(10, int(data.get("score", 0)))),
                    (data.get("url") or "").strip(),
                    get_domain(data.get("url", "")),
                    ",".join(data.get("emails", [])),
                ),
            )
            conn.commit()
            return cur.rowcount > 0
        except Exception as exc:
            logger.error("DB error saving '%s': %s", company, exc)
            return False

    def export_csv(self, path: str) -> int:
        """
        Export all leads to CSV sorted by score descending.

        Args:
            path: output file path.

        Returns:
            Number of rows exported.
        """
        cols = ("company", "service", "score", "url", "domain", "emails", "created_at")
        rows = self._conn().execute(
            "SELECT company, service, score, url, domain, emails, created_at "
            "FROM leads ORDER BY score DESC"
        ).fetchall()

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(cols)
            writer.writerows(rows)

        logger.info("Exported %d leads -> %s", len(rows), path)
        return len(rows)

    def export_json(self, path: str) -> int:
        """
        Export all leads to JSON sorted by score descending.

        Args:
            path: output file path.

        Returns:
            Number of records exported.
        """
        cols = ("company", "service", "score", "url", "domain", "emails", "created_at")
        rows = [
            dict(zip(cols, r))
            for r in self._conn().execute(
                "SELECT company, service, score, url, domain, emails, created_at "
                "FROM leads ORDER BY score DESC"
            ).fetchall()
        ]

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2, ensure_ascii=False, default=str)

        logger.info("Exported %d leads -> %s", len(rows), path)
        return len(rows)

    def count(self) -> int:
        """Return the total number of leads stored in the database."""
        return self._conn().execute("SELECT COUNT(*) FROM leads").fetchone()[0]

    def close(self) -> None:
        """Close the current thread's database connection."""
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            self._local.conn = None


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def normalize_url(url: str) -> str:
    """
    Normalise a URL: prepend https:// if no scheme is present and strip trailing slashes.

    Args:
        url: raw URL string.

    Returns:
        Normalised URL, or an empty string if the input is blank.
    """
    url = (url or "").strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url.rstrip("/")


def get_domain(url: str) -> str:
    """
    Extract the bare domain from a URL (strips www. prefix).

    Args:
        url: URL string.

    Returns:
        Lowercase domain, or 'unknown' on parse error.
    """
    try:
        return urlparse(url).netloc.replace("www.", "").lower() or "unknown"
    except Exception:
        return "unknown"


def extract_emails(text: str, html: str) -> List[str]:
    """
    Extract email addresses from visible page text and mailto: links.

    Filters out known blocked domains and file-extension false positives.

    Args:
        text: visible text content of the page.
        html: raw HTML source.

    Returns:
        Sorted list of unique lowercase email addresses.
    """
    candidates: set[str] = set()
    candidates.update(_EMAIL_RE.findall(text))
    candidates.update(_EMAIL_RE.findall(html))

    result: List[str] = []
    for email in candidates:
        e = email.lower()
        domain = e.split("@", 1)[-1]
        if domain in EMAIL_BLOCKED_DOMAINS:
            continue
        if any(e.endswith(ext) for ext in EMAIL_BLOCKED_EXTENSIONS):
            continue
        result.append(e)

    return sorted(result)


# ---------------------------------------------------------------------------
# HTTP SCRAPER
# ---------------------------------------------------------------------------

async def fetch(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    """
    Perform a GET request with exponential-jitter retry and Retry-After support.

    Args:
        session: shared aiohttp client session.
        url:     target URL.

    Returns:
        Response HTML text, or None if all attempts fail.
    """
    headers = {"User-Agent": random.choice(USER_AGENTS)}

    for attempt in range(CONFIG.retries):
        try:
            async with session.get(url, headers=headers, allow_redirects=True) as resp:
                if resp.status == 200:
                    return await resp.text(errors="replace")

                if resp.status == 429:
                    # Honour Retry-After if present, otherwise use progressive backoff
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after and retry_after.isdigit() \
                           else 5.0 * (attempt + 1)
                    logger.warning("Rate limited %s — waiting %.0fs", url, wait)
                    await asyncio.sleep(wait)
                    continue

                if resp.status in (401, 403, 404, 410, 451):
                    logger.debug("Permanent skip %s (%d)", url, resp.status)
                    return None

                if resp.status >= 500:
                    logger.warning("Server error %d on %s (attempt %d/%d)",
                                   resp.status, url, attempt + 1, CONFIG.retries)

        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.warning("Fetch error %s (attempt %d/%d): %s",
                           url, attempt + 1, CONFIG.retries, type(exc).__name__)

        # Exponential jitter backoff: ~1s, ~2s, ~4s
        backoff = (2 ** attempt) + random.uniform(0, 0.5)
        await asyncio.sleep(backoff)

    return None


def extract_text(html: str) -> str:
    """
    Extract clean text from HTML by removing scripts, styles, and non-content tags.

    Args:
        html: raw HTML source.

    Returns:
        Normalised text string (capped at text_max_chars characters).
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "aside", "noscript", "header"]):
        tag.decompose()

    root = soup.find("main") or soup.find("article") or soup.body
    if not root:
        return ""

    text = root.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text)[: CONFIG.text_max_chars]


# ---------------------------------------------------------------------------
# AI ANALYSIS — Ollama
# ---------------------------------------------------------------------------

def _extract_json_object(text: str) -> Optional[Dict]:
    """
    Extract the first valid JSON object from a string using brace counting.

    More robust than a bare json.loads() call on mixed-content text.

    Args:
        text: string that may contain a JSON object embedded in free text.

    Returns:
        Parsed Python dict, or None if no valid object is found.
    """
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start: i + 1])
                except json.JSONDecodeError:
                    start = None  # Try the next object candidate
    return None


_OLLAMA_PROMPT_TEMPLATE = (
    "Return ONLY valid minified JSON. No explanation, no markdown.\n"
    '{{"qualified": true/false, "company": "name", "service": "what they do", "score": 1-10}}\n'
    "DATA:\n{text}"
)


def _run_ollama(text: str, model: str) -> Optional[Dict]:
    """
    Blocking Ollama inference call.

    NOTE: always invoke via _try_ollama() — never call directly from the
    async event loop or you will block all concurrent workers.

    Args:
        text:  page text to analyse.
        model: Ollama model name (e.g. 'llama3', 'mistral').

    Returns:
        Lead data dict, or None on error.
    """
    try:
        import ollama  # optional — only required when using Ollama

        prompt = _OLLAMA_PROMPT_TEMPLATE.format(text=text)
        res    = ollama.generate(model=model, prompt=prompt)
        raw    = re.sub(r"```json?\s*|```", "", res.get("response", ""))
        data   = _extract_json_object(raw)

        if data and data.get("company"):
            data["score"]     = max(0, min(10, int(data.get("score", 0))))
            data["qualified"] = bool(data.get("qualified", False))
            return data

    except ImportError:
        logger.debug("Ollama not installed — falling back to heuristics")
    except Exception as exc:
        logger.debug("Ollama call error: %s", exc)

    return None


async def _try_ollama(text: str) -> Optional[Dict]:
    """
    Async wrapper around the blocking _run_ollama().

    Runs the blocking call in a thread-pool worker via asyncio.to_thread()
    and applies a cross-platform timeout via asyncio.wait_for().

    Args:
        text: page text to analyse.

    Returns:
        Lead data dict, or None on timeout or error.
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_run_ollama, text, CONFIG.ollama_model),
            timeout=CONFIG.ollama_timeout,
        )
    except asyncio.TimeoutError:
        logger.warning("Ollama timed out after %.0fs — falling back to heuristics",
                       CONFIG.ollama_timeout)
        return None
    except Exception as exc:
        logger.debug("Ollama wrapper error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# AI ANALYSIS — Heuristic Fallback
# ---------------------------------------------------------------------------

def _heuristic_analysis(text: str, url: str) -> Dict:
    """
    Keyword-based scoring used as a fallback when Ollama is unavailable.

    Score = keyword hit count + text-length bonus, clamped to [0, 10].

    Args:
        text: visible page text.
        url:  page URL (used to infer the company name from the domain).

    Returns:
        Dict with keys: company, service, score, qualified.
    """
    text_lower = text.lower()
    domain     = get_domain(url)

    # Count keyword matches
    hits = sum(1 for kw in BUSINESS_KEYWORDS if kw in text_lower)

    # Bonus for content-rich pages (max +2)
    length_bonus = min(2, len(text) // 800)
    score = min(10, max(0, hits + length_bonus))

    # Derive company name from the domain
    company = (
        domain.split(".")[0].replace("-", " ").replace("_", " ").title()
        if domain != "unknown"
        else "Unknown"
    )

    # Pick service label from the first matching keyword
    service = next(
        (label for kw, label in BUSINESS_KEYWORDS.items() if kw in text_lower),
        "Unknown",
    )

    return {
        "company":   company,
        "service":   service,
        "score":     score,
        "qualified": score > 2,
    }


async def analyze_lead(text: str, url: str) -> Dict:
    """
    Analyse a lead with Ollama; fall back to heuristics if unavailable.

    Args:
        text: extracted page text.
        url:  page URL.

    Returns:
        Dict with keys: company, service, score, qualified.
    """
    result = await _try_ollama(text)
    return result if result else _heuristic_analysis(text, url)


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

async def process_url(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url:       str,
    db:        LeadDB,
    counters:  Dict[str, int],
) -> Optional[Dict]:
    """
    Process a single URL: fetch, extract, analyse, save.

    Args:
        session:   shared aiohttp client session.
        semaphore: concurrency limiter.
        url:       already-normalised target URL.
        db:        LeadDB instance.
        counters:  shared counter dict tracking fetched / qualified / saved.

    Returns:
        Lead dict if the page qualified and was saved, None otherwise.
    """
    async with semaphore:
        await asyncio.sleep(random.uniform(CONFIG.rate_limit_min, CONFIG.rate_limit_max))

        html = await fetch(session, url)
        if not html:
            return None

        counters["fetched"] = counters.get("fetched", 0) + 1

        text = extract_text(html)
        if len(text) < CONFIG.min_text_chars:
            logger.debug("Skipping %s — too short (%d chars)", url, len(text))
            return None

        data = await analyze_lead(text, url)

        if not data.get("qualified"):
            logger.debug("Not qualified: %s (score=%d)", url, data.get("score", 0))
            return None

        counters["qualified"] = counters.get("qualified", 0) + 1

        data["url"]    = url
        data["emails"] = extract_emails(text, html)
        saved = db.save(data)

        if saved:
            counters["saved"] = counters.get("saved", 0) + 1
            logger.info(
                "[✓] %-28s | %-16s | score=%2d | emails=%d",
                data["company"],
                data["service"],
                data["score"],
                len(data["emails"]),
            )

        return data if saved else None


async def run(
    urls:    List[str],
    db:      LeadDB,
    workers: int = 0,
) -> tuple[List[Dict], Dict[str, int]]:
    """
    Execute the async pipeline over a list of URLs.

    Args:
        urls:    normalised, deduplicated URL list.
        db:      LeadDB instance.
        workers: max concurrent workers (0 = use CONFIG.max_concurrent).

    Returns:
        Tuple of (qualified lead list, counters dict).
    """
    workers   = workers or CONFIG.max_concurrent
    semaphore = asyncio.Semaphore(workers)
    counters: Dict[str, int] = {"fetched": 0, "qualified": 0, "saved": 0}

    timeout = aiohttp.ClientTimeout(
        total=CONFIG.request_timeout,
        connect=CONFIG.connect_timeout,
    )

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            process_url(session, semaphore, u, db, counters)
            for u in urls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    leads: List[Dict] = []
    errors = 0
    for r in results:
        if isinstance(r, Exception):
            logger.error("Task exception: %s", r)
            errors += 1
        elif r is not None:
            leads.append(r)

    counters["errors"] = errors
    if errors:
        logger.warning("%d task(s) failed — see pipeline.log for details", errors)

    return leads, counters


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sovereign Lead Engine v3.5 — AI-powered B2B lead generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python sovereign_lead_engine_v3_5.py urls.txt
  python sovereign_lead_engine_v3_5.py urls.txt --workers 10 --export both
  python sovereign_lead_engine_v3_5.py urls.txt --min-score 7 --export csv --output results
  OLLAMA_MODEL=mistral python sovereign_lead_engine_v3_5.py urls.txt
        """,
    )
    parser.add_argument("input",
                        help="Text file containing URLs (one per line, # for comments)")
    parser.add_argument("--workers", type=int, default=CONFIG.max_concurrent,
                        help=f"Max concurrent workers (default: {CONFIG.max_concurrent})")
    parser.add_argument("--export", choices=["csv", "json", "both"], default=None,
                        help="Export format")
    parser.add_argument("--output", default="leads_export",
                        help="Export filename without extension (default: leads_export)")
    parser.add_argument("--min-score", type=int, default=0,
                        help="Only display leads with score >= N (all leads are still saved to DB)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity (default: INFO)")
    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args   = parser.parse_args()

    setup_logging(args.log_level)

    # Load URLs from input file
    try:
        with open(args.input, encoding="utf-8") as fh:
            raw_urls = [
                line.strip()
                for line in fh
                if line.strip() and not line.startswith("#")
            ]
    except FileNotFoundError:
        logger.error("Input file not found: %s", args.input)
        return

    # Normalise and deduplicate while preserving insertion order
    seen: set[str] = set()
    urls: List[str] = []
    for u in (normalize_url(r) for r in raw_urls):
        if u and u not in seen:
            seen.add(u)
            urls.append(u)

    if not urls:
        logger.error("No valid URLs found in %s", args.input)
        return

    logger.info(
        "Loaded %d unique URLs | Workers: %d | Model: %s",
        len(urls), args.workers, CONFIG.ollama_model,
    )

    with LeadDB() as db:
        t0 = time.perf_counter()
        leads, counters = asyncio.run(run(urls, db, workers=args.workers))
        elapsed = time.perf_counter() - t0

        # Filter by min-score for display only (all leads are persisted in DB)
        displayed = [r for r in leads if r.get("score", 0) >= args.min_score] \
                    if args.min_score else leads

        # Print summary
        bar = "=" * 56
        print(f"\n{bar}")
        score_note = f"  (score >= {args.min_score})" if args.min_score else ""
        print(f"  Qualified leads  : {len(displayed)}{score_note}")
        print(f"  Pages fetched    : {counters.get('fetched', 0)}")
        print(f"  Saved to DB      : {counters.get('saved', 0)}")
        print(f"  Errors           : {counters.get('errors', 0)}")
        print(f"  Total time       : {elapsed:.1f}s")
        print(f"  AI cost          : $0.00")
        print(f"{bar}\n")

        if displayed:
            print(f"  {'COMPANY':<28} {'SERVICE':<16} {'SCORE':<8} EMAIL")
            print(f"  {'-'*28} {'-'*16} {'-'*8} {'-'*30}")
            for r in sorted(displayed, key=lambda x: x.get("score", 0), reverse=True):
                emails = ", ".join(r.get("emails", [])) or "-"
                print(f"  {r['company']:<28} {r['service']:<16} {r['score']}/10     {emails}")
            print()

        # Export
        if args.export in ("csv", "both"):
            db.export_csv(f"{args.output}.csv")
        if args.export in ("json", "both"):
            db.export_json(f"{args.output}.json")


if __name__ == "__main__":
    main()
