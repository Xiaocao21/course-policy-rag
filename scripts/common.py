"""Shared helpers for the course-policy-rag pipeline.

Design notes that matter for this project (do not "simplify" these away):

* Two text layers exist per document:
    - evidence  : character-faithful to the source. ALL gold spans are anchored here.
    - retrieval : normalised, used for embedding and retrieval.
  A charmap connects retrieval positions back to evidence positions so that
  "full-span containment" and "partial_span_coverage" can be computed exactly.
  Normalisation that changes character counts MUST be registered in the charmap.

* Clause spans are stable: chunking may never redefine a clause boundary.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
import urllib.error
import urllib.request
import urllib.robotparser as robotparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a stage-A dependency
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _prefer_local_hf_cache() -> None:
    """Default to the local Hugging Face cache when the model is already present.

    Rationale: with transformers 5.x and sentence-transformers 6.x, fetching
    bge-small-en-v1.5 fresh can fail while instantiating AutoProcessor, whereas the
    cached snapshot loads fine. Setting this *before* huggingface_hub is imported is
    what makes the switch take effect, so it runs at module import time.
    """
    import os
    cache = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub"
    try:
        if any(cache.glob("models--BAAI--bge-small-en-v1.5*")):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    except OSError:
        pass


_prefer_local_hf_cache()


# --------------------------------------------------------------------- config
def load_settings(root: Path | None = None) -> dict[str, Any]:
    root = Path(root) if root else PROJECT_ROOT
    path = root / "config" / "settings.yaml"
    if yaml is None:
        raise RuntimeError("PyYAML is required: .venv\\Scripts\\python.exe -m pip install -r requirements.txt")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_taxonomy(root: Path | None = None) -> dict[str, Any]:
    root = Path(root) if root else PROJECT_ROOT
    path = root / "config" / "domain_taxonomy.yaml"
    if yaml is None:
        raise RuntimeError("PyYAML is required")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def compile_detectors(taxonomy: dict[str, Any]) -> dict[str, re.Pattern]:
    out: dict[str, re.Pattern] = {}
    for name, pattern in taxonomy.get("detectors", {}).items():
        out[name] = re.compile(pattern)
    return out


def project_paths(root: Path | None = None) -> dict[str, Path]:
    root = Path(root) if root else PROJECT_ROOT
    return {
        "root": root,
        "config": root / "config",
        "raw": root / "raw",
        "snapshots": root / "snapshots",
        "extracted": root / "extracted",
        "cleaned": root / "cleaned",
        "corpus": root / "corpus_safe_512",
        "annotations": root / "annotations",
        "reports": root / "reports",
        "tests": root / "tests",
        "fixtures": root / "tests" / "fixtures",
        "logs": root / "logs",
    }


# --------------------------------------------------------------------- hashing
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------- robots
@dataclass
class RobotsCache:
    """Per-host robots.txt cache that records the raw policy text as audit evidence."""

    user_agent: str
    cache: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, str] = field(default_factory=dict)
    fetched_at: dict[str, str] = field(default_factory=dict)

    def _load(self, host_root: str) -> Any:
        if host_root in self.cache:
            return self.cache[host_root]
        url = f"{host_root}/robots.txt"
        parser = robotparser.RobotFileParser()
        parser.set_url(url)
        status = "ok"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
            with urllib.request.urlopen(req, timeout=20) as r:
                body = r.read().decode("utf-8", "ignore")
            self.raw[host_root] = body
            parser.parse(body.splitlines())
        except urllib.error.HTTPError as e:
            # 404 robots.txt means "no restrictions"
            status = f"http_{e.code}"
            self.raw[host_root] = ""
            parser.parse([])
        except Exception as e:  # noqa: BLE001
            status = f"error_{type(e).__name__}"
            self.raw[host_root] = ""
            parser.parse([])
        self.cache[host_root] = (parser, status)
        self.fetched_at[host_root] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        return self.cache[host_root]

    def decision(self, url: str) -> dict[str, Any]:
        parts = urlsplit(url)
        host_root = f"{parts.scheme}://{parts.netloc}"
        parser, status = self._load(host_root)
        allowed = parser.can_fetch(self.user_agent, url)
        delay = None
        try:
            delay = parser.crawl_delay(self.user_agent)
        except Exception:  # noqa: BLE001
            delay = None
        return {
            "url": url,
            "robots_url": f"{host_root}/robots.txt",
            "robots_status": status,
            "allowed": bool(allowed),
            "crawl_delay": delay,
            "checked_at": self.fetched_at.get(host_root, ""),
        }

    def audit(self) -> dict[str, Any]:
        return {
            "user_agent": self.user_agent,
            "hosts": {
                host: {
                    "robots_url": f"{host}/robots.txt",
                    "status": status,
                    "fetched_at": self.fetched_at.get(host, ""),
                    "raw_sha256": sha256_text(self.raw.get(host, "")),
                    "raw": self.raw.get(host, ""),
                }
                for host, (_parser, status) in self.cache.items()
            },
        }


# --------------------------------------------------------------------- fetcher
@dataclass
class FetchResult:
    url: str
    status: int | None
    content: bytes
    content_type: str
    final_url: str
    error: str | None = None
    attempts: int = 1
    last_modified: str | None = None
    etag: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.status is not None and 200 <= self.status < 300


class RateLimitedFetcher:
    """Polite fetcher: per-host delay, retry with exponential backoff, honest UA."""

    def __init__(self, user_agent: str, delay: float = 1.5, timeout: int = 30,
                 max_retries: int = 4, backoff_base: float = 2.0, backoff_max: float = 60.0,
                 respect_crawl_delay: bool = True):
        self.user_agent = user_agent
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.respect_crawl_delay = respect_crawl_delay
        self._last_hit: dict[str, float] = {}

    def _host(self, url: str) -> str:
        p = urlsplit(url)
        return f"{p.scheme}://{p.netloc}"

    def _wait(self, url: str, crawl_delay: float | None) -> None:
        host = self._host(url)
        want = self.delay
        if self.respect_crawl_delay and crawl_delay:
            want = max(want, float(crawl_delay))
        elapsed = time.time() - self._last_hit.get(host, 0.0)
        if elapsed < want:
            time.sleep(want - elapsed)

    def fetch(self, url: str, crawl_delay: float | None = None, accept: str | None = None) -> FetchResult:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": accept or "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            self._wait(url, crawl_delay)
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    body = r.read()
                    self._last_hit[self._host(url)] = time.time()
                    return FetchResult(
                        url=url, status=r.status, content=body,
                        content_type=r.headers.get("Content-Type", ""),
                        final_url=r.geturl(),
                        attempts=attempt,
                        last_modified=r.headers.get("Last-Modified"),
                        etag=r.headers.get("ETag"),
                    )
            except urllib.error.HTTPError as e:
                self._last_hit[self._host(url)] = time.time()
                last_error = f"HTTP {e.code}"
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(min(self.backoff_base ** attempt, self.backoff_max))
                    continue
                return FetchResult(url=url, status=e.code, content=b"", content_type="",
                                   final_url=url, error=last_error, attempts=attempt)
            except Exception as e:  # noqa: BLE001
                self._last_hit[self._host(url)] = time.time()
                last_error = f"{type(e).__name__}: {e}"
                if attempt < self.max_retries:
                    time.sleep(min(self.backoff_base ** attempt, self.backoff_max))
                    continue
        return FetchResult(url=url, status=None, content=b"", content_type="",
                           final_url=url, error=last_error or "unknown", attempts=self.max_retries)


# --------------------------------------------------------------------- text layers
_META_CHARSET_RE = re.compile(
    rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_\-]+)""", re.I)


def decode_html(raw: bytes, content_type: str = "") -> str:
    """Decode an HTML byte string using its declared charset.

    Decoding everything as UTF-8 with errors="ignore" silently deleted bytes and left
    U+FFFD behind, so a curly quote became a replacement character. That is worse than
    a visible failure here: gold spans are anchored to this text and its character
    count feeds the charmap, so lost characters corrupt offsets.

    Order: charset from the Content-Type header, then a <meta charset> declaration,
    then strict UTF-8, then windows-1252 (which maps every byte and never fails).
    """
    candidates: list[str] = []
    if content_type:
        m = re.search(r"charset\s*=\s*\"?([A-Za-z0-9_\-]+)", content_type, re.I)
        if m:
            candidates.append(m.group(1).lower())
    m = _META_CHARSET_RE.search(raw[:4096])
    if m:
        candidates.append(m.group(1).decode("ascii", "ignore").lower())
    candidates += ["utf-8", "windows-1252"]
    for cand in candidates:
        if not cand:
            continue
        try:
            return raw.decode(cand)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", "replace")


_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)
_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
              "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st"}
_QUOTES = {"\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
           "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2032": "'", "\u2033": '"'}
_DASHES = {"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
           "\u2015": "-", "\u2212": "-"}
_BULLETS = {"\u2022": "-", "\u2023": "-", "\u25cf": "-", "\u25aa": "-", "\u2043": "-",
            "\u00b7": "-", "\u25e6": "-"}


@dataclass
class TextLayers:
    """Evidence text plus normalised retrieval text plus a retrieval->evidence charmap."""

    evidence: str
    retrieval: str
    charmap: list[int]          # charmap[i] = index in evidence for retrieval[i]
    ops: list[dict] = field(default_factory=list)

    def evidence_span(self, retrieval_start: int, retrieval_end: int) -> tuple[int, int]:
        """Map a [start, end) span in retrieval space to evidence space."""
        if not self.charmap:
            return 0, 0
        rs = max(0, min(retrieval_start, len(self.charmap) - 1))
        re_ = max(rs, min(retrieval_end, len(self.charmap)))
        start = self.charmap[rs]
        end = self.charmap[re_ - 1] + 1 if re_ > rs else start
        return start, end


def build_text_layers(raw_text: str, cfg: dict[str, Any] | None = None) -> TextLayers:
    """Build evidence + retrieval layers with an exact charmap.

    Evidence layer keeps the original characters (including the original
    whitespace runs) so gold spans stay byte-faithful. Retrieval layer applies
    only semantics-preserving normalisation. Every output character records the
    evidence index it came from, which makes the map exact even for expansions
    (e.g. a ligature becoming two characters).
    """
    cfg = (cfg or {}).get("retrieval", {}) if cfg else {}
    evidence = raw_text
    out_chars: list[str] = []
    charmap: list[int] = []
    ops: list[dict] = []

    i = 0
    n = len(evidence)
    prev_space = True  # collapse leading whitespace
    while i < n:
        ch = evidence[i]

        if cfg.get("strip_zero_width", True) and ord(ch) in _ZERO_WIDTH:
            ops.append({"op": "strip_zero_width", "ev": i})
            i += 1
            continue

        if ch in _LIGATURES and cfg.get("expand_ligatures", True):
            rep = _LIGATURES[ch]
            for r in rep:
                out_chars.append(r)
                charmap.append(i)
            ops.append({"op": "ligature", "ev": i, "to": rep})
            i += 1
            prev_space = False
            continue

        if ch in _QUOTES and cfg.get("unify_quotes", True):
            out_chars.append(_QUOTES[ch])
            charmap.append(i)
            ops.append({"op": "quote", "ev": i})
            i += 1
            prev_space = False
            continue

        if ch in _DASHES and cfg.get("unify_dashes", True):
            out_chars.append("-")
            charmap.append(i)
            ops.append({"op": "dash", "ev": i})
            i += 1
            prev_space = False
            continue

        if cfg.get("unify_bullets", True) and ch == "\u2022":
            out_chars.append("-")
            charmap.append(i)
            ops.append({"op": "bullet", "ev": i})
            i += 1
            prev_space = False
            continue

        if ch.isspace():
            if cfg.get("collapse_whitespace", True):
                if prev_space:
                    i += 1
                    continue
                out_chars.append(" ")
                charmap.append(i)
                prev_space = True
                i += 1
                continue

        if cfg.get("nfkc", True) or cfg.get("fullwidth_to_halfwidth", True):
            norm = unicodedata.normalize("NFKC", ch)
            if norm != ch:
                for r in norm:
                    out_chars.append(r)
                    charmap.append(i)
                ops.append({"op": "nfkc", "ev": i, "from": ch, "to": norm})
                i += 1
                prev_space = False
                continue

        out_chars.append(ch)
        charmap.append(i)
        prev_space = False
        i += 1

    retrieval = "".join(out_chars)
    # trailing space produced by collapse is harmless but trim for stability
    while retrieval.endswith(" ") and charmap:
        retrieval = retrieval[:-1]
        charmap.pop()
    return TextLayers(evidence=evidence, retrieval=retrieval, charmap=charmap, ops=ops)


def contains_span(evidence_text: str, covered: list[tuple[int, int]], span: tuple[int, int]) -> bool:
    """True when the union of covered evidence ranges fully contains `span` (half-open)."""
    s, e = span
    if e <= s:
        return False
    ranges = sorted((a, b) for a, b in covered if b > a)
    cursor = s
    for a, b in ranges:
        if b <= cursor:
            continue
        if a > cursor:
            return False
        cursor = max(cursor, b)
        if cursor >= e:
            return True
    return cursor >= e


def covered_token_fraction(evidence_text: str, covered: list[tuple[int, int]],
                           span: tuple[int, int]) -> float:
    """Fraction of annotated source characters in `span` present in the union of `covered`."""
    s, e = span
    if e <= s:
        return 0.0
    marks = bytearray(e - s)
    for a, b in covered:
        lo = max(a, s)
        hi = min(b, e)
        if hi > lo:
            marks[lo - s:hi - s] = b"\x01" * (hi - lo)
    return sum(marks) / (e - s)


# --------------------------------------------------------------------- minhash
def shingles(text: str, k: int = 5) -> set[str]:
    t = re.sub(r"\s+", " ", text.lower()).strip()
    if len(t) <= k:
        return {t} if t else set()
    return {t[i:i + k] for i in range(len(t) - k + 1)}


def minhash_signature(text: str, k: int = 5, num_hashes: int = 128) -> tuple[int, ...]:
    """Deterministic MinHash signature using blake2b-derived permutations."""
    sh = shingles(text, k)
    if not sh:
        return tuple([0] * num_hashes)
    sig = []
    for seed in range(num_hashes):
        m = None
        for s in sh:
            h = int.from_bytes(
                hashlib.blake2b(f"{seed}:{s}".encode("utf-8"), digest_size=8).digest(), "big")
            if m is None or h < m:
                m = h
        sig.append(m if m is not None else 0)
    return tuple(sig)


def minhash_similarity(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# --------------------------------------------------------------------- misc
def slugify(text: str, max_len: int = 60) -> str:
    t = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return t[:max_len] or "untitled"


def domain_hits(text: str, taxonomy: dict[str, Any]) -> dict[str, int]:
    low = text.lower()
    return {
        name: sum(low.count(kw.lower()) for kw in spec.get("keywords", []))
        for name, spec in taxonomy.get("domains", {}).items()
    }


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def normalise_doc_id(institution: str, doc_type: str, term: str = "", version: str = "") -> str:
    parts = [slugify(institution, 24).upper(), slugify(doc_type, 28).upper()]
    if term:
        parts.append(slugify(term, 12).upper())
    doc_id = "-".join(p for p in parts if p)
    if version:
        doc_id = f"{doc_id}-{slugify(version, 8)}"
    return doc_id


# --------------------------------------------------------------------- models
def load_sentence_transformer(model_name: str, device: str = "cpu"):
    """Load a SentenceTransformer, falling back to the local cache when offline.

    The hub can serve a model config that the installed transformers/PyTorch
    combination rejects while instantiating AutoProcessor (observed with
    bge-small-en-v1.5 and transformers 5.x). The weights are already cached, so a
    second attempt with hub access disabled is both faster and more reproducible.
    """
    import os
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_name, device=device)
    except Exception as first_error:  # noqa: BLE001
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            return SentenceTransformer(model_name, device=device)
        except Exception as second_error:  # noqa: BLE001
            raise RuntimeError(
                f"could not load {model_name} online ({type(first_error).__name__}) "
                f"or offline ({type(second_error).__name__}). If it has never been "
                f"downloaded, run once with network access to populate the HF cache."
            ) from second_error
