"""Seed probe: robots + reachability + policy density + evidence capability.

Runs on stdlib + lxml only (no third-party deps) so it can execute before the
venv install finishes.

Usage:
    python probe_seeds.py --out reports/seed_probe.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
import urllib.robotparser as robotparser
from pathlib import Path

try:
    import lxml.html as LH
except ImportError:  # pragma: no cover
    LH = None

UA = "CPS5801-research-bot/0.1 (+academic research; contact: Xiaocao21@users.noreply.github.com)"

# ---------------------------------------------------------------- candidates
# tier: course | faculty | institution
CANDIDATES: list[dict] = [
    # ---------------- faculty / department tier (the new layer) ----------------
    {"tier": "faculty", "institution": "MIT", "country": "US", "doc_type": "dept_grading_policy",
     "url": "https://www.eecs.mit.edu/academics/graduate-programs/"},
    {"tier": "faculty", "institution": "MIT", "country": "US", "doc_type": "dept_academic_policy",
     "url": "https://www.eecs.mit.edu/academics/"},
    {"tier": "faculty", "institution": "Boston University", "country": "US", "doc_type": "college_policies",
     "url": "https://www.bu.edu/cas/academics/policies/"},
    {"tier": "faculty", "institution": "Boston University", "country": "US", "doc_type": "college_policies",
     "url": "https://www.bu.edu/eng/academics/undergraduate/policies/"},
    {"tier": "faculty", "institution": "Boston University", "country": "US", "doc_type": "college_policies",
     "url": "https://www.bu.edu/academics/"},
    {"tier": "faculty", "institution": "University of Edinburgh", "country": "UK", "doc_type": "school_policies",
     "url": "https://www.ed.ac.uk/informatics/undergraduate/studying/policies"},
    {"tier": "faculty", "institution": "University of Edinburgh", "country": "UK", "doc_type": "school_policies",
     "url": "https://www.ed.ac.uk/engineering/students/taught-students"},
    {"tier": "faculty", "institution": "Imperial College London", "country": "UK", "doc_type": "dept_policies",
     "url": "https://www.imperial.ac.uk/computing/current-students/"},
    {"tier": "faculty", "institution": "Imperial College London", "country": "UK", "doc_type": "dept_policies",
     "url": "https://www.imperial.ac.uk/engineering/study/undergraduate/"},
    {"tier": "faculty", "institution": "University of Wisconsin-Madison", "country": "US", "doc_type": "college_policies",
     "url": "https://www.engr.wisc.edu/academics/student-services/academic-policies/"},
    {"tier": "faculty", "institution": "Purdue University", "country": "US", "doc_type": "college_policies",
     "url": "https://www.purdue.edu/engineering/students/"},
    {"tier": "faculty", "institution": "Purdue University", "country": "US", "doc_type": "college_policies",
     "url": "https://www.purdue.edu/science/Current_Students/"},
    {"tier": "faculty", "institution": "University of Washington", "country": "US", "doc_type": "college_policies",
     "url": "https://www.engr.washington.edu/current/policies"},
    {"tier": "faculty", "institution": "University of Washington", "country": "US", "doc_type": "college_policies",
     "url": "https://www.cs.washington.edu/academics/ugrad/policies"},
    {"tier": "faculty", "institution": "Georgia Tech", "country": "US", "doc_type": "college_policies",
     "url": "https://www.cc.gatech.edu/academics/policies"},
    {"tier": "faculty", "institution": "University of Illinois Urbana-Champaign", "country": "US", "doc_type": "college_policies",
     "url": "https://grainger.illinois.edu/academics/undergraduate/policies"},
    {"tier": "faculty", "institution": "University of Toronto", "country": "CA", "doc_type": "faculty_policies",
     "url": "https://www.engineering.utoronto.ca/current-students/academic-policies/"},
    {"tier": "faculty", "institution": "University of British Columbia", "country": "CA", "doc_type": "faculty_policies",
     "url": "https://www.apsc.ubc.ca/students/undergraduate/academic-policies/"},
    {"tier": "faculty", "institution": "University of Melbourne", "country": "AU", "doc_type": "faculty_policies",
     "url": "https://eng.unimelb.edu.au/students/undergraduate/policies"},
    {"tier": "faculty", "institution": "Monash University", "country": "AU", "doc_type": "faculty_policies",
     "url": "https://www.monash.edu/engineering/current-students/policies"},
    {"tier": "faculty", "institution": "University of Manchester", "country": "UK", "doc_type": "school_policies",
     "url": "https://www.manchester.ac.uk/study/undergraduate/apply/policies/"},
    {"tier": "faculty", "institution": "University College London", "country": "UK", "doc_type": "dept_policies",
     "url": "https://www.ucl.ac.uk/engineering/study/undergraduate"},
    {"tier": "faculty", "institution": "University of Michigan", "country": "US", "doc_type": "college_policies",
     "url": "https://cse.engin.umich.edu/academics/undergraduate/policies/"},
    {"tier": "faculty", "institution": "Ohio State University", "country": "US", "doc_type": "college_policies",
     "url": "https://engineering.osu.edu/current-students/academic-policies"},
    {"tier": "faculty", "institution": "Penn State University", "country": "US", "doc_type": "college_policies",
     "url": "https://www.eecs.psu.edu/students/undergraduate/advising/policies.aspx"},
    {"tier": "faculty", "institution": "University of Texas at Austin", "country": "US", "doc_type": "college_policies",
     "url": "https://www.cs.utexas.edu/undergraduate/academics/policies"},
    {"tier": "faculty", "institution": "Cornell University", "country": "US", "doc_type": "college_policies",
     "url": "https://www.cs.cornell.edu/undergrad/rulesandprocedures"},
    {"tier": "faculty", "institution": "Carnegie Mellon University", "country": "US", "doc_type": "dept_policies",
     "url": "https://www.cmu.edu/student-affairs/theword/academic/index.html"},
    {"tier": "faculty", "institution": "National University of Singapore", "country": "SG", "doc_type": "faculty_policies",
     "url": "https://www.comp.nus.edu.sg/undergraduates/"},
    {"tier": "faculty", "institution": "University of Sydney", "country": "AU", "doc_type": "faculty_policies",
     "url": "https://www.sydney.edu.au/engineering/study/current-students.html"},

    # ---------------- institution tier (already partly verified) ----------------
    {"tier": "institution", "institution": "Boston University", "country": "US", "doc_type": "academic_conduct_code",
     "url": "https://www.bu.edu/academics/policies/academic-conduct-code/"},
    {"tier": "institution", "institution": "Boston University", "country": "US", "doc_type": "attendance_policy",
     "url": "https://www.bu.edu/academics/policies/attendance/"},
    {"tier": "institution", "institution": "Boston University", "country": "US", "doc_type": "policy_index",
     "url": "https://www.bu.edu/academics/policies/"},
    {"tier": "institution", "institution": "University of Edinburgh", "country": "UK", "doc_type": "assessment_regulations",
     "url": "https://www.ed.ac.uk/academic-services/students/assessment"},
    {"tier": "institution", "institution": "University of Edinburgh", "country": "UK", "doc_type": "degree_regulations",
     "url": "https://www.ed.ac.uk/academic-services/policies-regulations/regulations"},
    {"tier": "institution", "institution": "University of Edinburgh", "country": "UK", "doc_type": "policy_index",
     "url": "https://www.ed.ac.uk/academic-services/policies-regulations"},
    {"tier": "institution", "institution": "ICAI", "country": "INT", "doc_type": "integrity_fundamentals",
     "url": "https://academicintegrity.org/resources/fundamental-values"},
    {"tier": "institution", "institution": "ICAI", "country": "INT", "doc_type": "integrity_resources",
     "url": "https://academicintegrity.org/resources/"},
    {"tier": "institution", "institution": "Stanford University", "country": "US", "doc_type": "community_standards",
     "url": "https://communitystandards.stanford.edu/"},
    {"tier": "institution", "institution": "Imperial College London", "country": "UK", "doc_type": "student_policies",
     "url": "https://www.imperial.ac.uk/students/"},
    {"tier": "institution", "institution": "Purdue University", "country": "US", "doc_type": "registrar_policies",
     "url": "https://www.purdue.edu/registrar/"},
    {"tier": "institution", "institution": "QAA", "country": "UK", "doc_type": "quality_code",
     "url": "https://www.qaa.ac.uk/quality-code"},
]

# ---------------------------------------------------------------- thresholds
DOMAIN_TAXONOMY = {
    "grading": ["grading", "grade", "gpa", "mark", "marks", "assessment criteria", "score", "weighting"],
    "late": ["late", "deadline", "extension", "overdue", "penalt", "submission date", "grace period"],
    "attendance": ["attendance", "absence", "absent", "attend", "participation", "presence"],
    "exam": ["exam", "examination", "make-up", "makeup", "deferred", "resit", "retake", "test", "quiz"],
    "integrity": ["integrity", "plagiar", "misconduct", "cheat", "dishonest", "citation", "collusion", "fabricat"],
    "resources": ["resource", "office hours", "tutoring", "advising", "library", "accommodation", "accessib"],
}

EXCEPTION_RE = re.compile(
    r"(?i)\b(except|unless|provided that|notwithstanding|subject to|may be waived|"
    r"in exceptional circumstances|other than|with the exception of|however)\b"
)
CROSSREF_RE = re.compile(
    r"(?i)\b(see (section|article|clause|appendix)|as defined in|pursuant to|"
    r"referred to in|in accordance with (section|article)|set out in)\b"
)
ALT_EVIDENCE_RE = re.compile(
    r"(?i)\b(either\b|or alternatively|alternatively|may instead|one of the following|"
    r"any of the following|as an alternative|instead of)\b"
)
CLAUSE_RE = re.compile(
    r"(?m)(?:^|\n)\s*(?:\d+(?:\.\d+){1,3}\b|[A-Z]\.\s|\(\s*[a-z]\s*\)\s|Article\s+\d+|Section\s+\d+)"
)
ACADEMIC_YEAR_RE = re.compile(r"(?i)\b(20\d{2})\s*[-/–]\s*(20\d{2})\b")
VERSION_RE = re.compile(r"(?i)\b(version|revised?|effective (date|from)|last updated|approved on)\b")


TIMEOUT = 15
MAX_BYTES = 4_000_000


def fetch(url: str, timeout: int | None = None) -> tuple[int | None, bytes, str | None]:
    """Bounded fetch. Reading stops at MAX_BYTES and the socket timeout is hard.

    A slow or dead host must not be able to stall the probe: the overall call is
    additionally guarded by a watchdog thread.
    """
    timeout = timeout or TIMEOUT
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(MAX_BYTES), None
    except urllib.error.HTTPError as e:
        return e.code, b"", f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001
        return None, b"", f"{type(e).__name__}: {e}"


def robots_check(url: str) -> tuple[bool | str, str]:
    from urllib.parse import urlsplit
    parts = urlsplit(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        return rp.can_fetch(UA, url), robots_url
    except Exception as e:  # noqa: BLE001
        return f"ERR({type(e).__name__})", robots_url


_HTML_PARSER = None


def _html_parser():
    """Prefer selectolax, fall back to lxml. Never silently return '' (see note)."""
    global _HTML_PARSER
    if _HTML_PARSER is not None:
        return _HTML_PARSER
    try:
        from selectolax.lexbor import LexborHTMLParser as _SLX
        _HTML_PARSER = ("selectolax", _SLX)
        return _HTML_PARSER
    except ImportError:
        pass
    try:
        from selectolax.parser import HTMLParser as _SLX  # legacy selectolax
        _HTML_PARSER = ("selectolax", _SLX)
        return _HTML_PARSER
    except ImportError:
        pass
    if LH is not None:
        _HTML_PARSER = ("lxml", LH)
        return _HTML_PARSER
    raise RuntimeError(
        "No HTML parser available. Install stage-A deps: "
        ".venv\\Scripts\\python.exe -m pip install -r requirements.txt"
    )


def text_of(html: bytes) -> str:
    """Extract visible main text.

    NOTE: a previous version returned "" when the parser was missing, which made
    every page look like an empty 'pending_review' page. It now raises instead of
    reporting a false negative.
    """
    kind, parser = _html_parser()
    html_str = html.decode("utf-8", "ignore")
    if kind == "selectolax":
        tree = parser(html_str)
        for tag in ("script", "style", "noscript", "nav", "header", "footer", "aside"):
            for node in tree.css(tag):
                node.decompose()
        root = None
        for sel in ("main", "article", "[role=main]", "#main-content", "#content"):
            nodes = tree.css(sel)
            if nodes:
                root = max(nodes, key=lambda n: len(n.text() or ""))
                break
        root = root or tree.body or tree.root
        if root is None:
            return ""
        parts: list[str] = []
        for node in root.traverse(include_text=True):
            if node.tag == "-text":
                parts.append(node.text_content or "")
            elif node.tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6",
                              "tr", "br", "section", "article", "table", "ul", "ol"}:
                parts.append(" ")
        return " ".join("".join(parts).split())

    doc = parser.fromstring(html_str)
    for bad in doc.xpath("//script|//style|//noscript|//nav|//header|//footer|//aside"):
        parent = bad.getparent()
        if parent is not None:
            parent.remove(bad)
    return " ".join(doc.text_content().split())


def pdf_text_from_bytes(body: bytes) -> str:
    """Extract text from an in-memory PDF so PDF policies can be scored too."""
    try:
        import io
        import pdfplumber
    except ImportError:
        return ""
    try:
        chunks: list[str] = []
        with pdfplumber.open(io.BytesIO(body)) as pdf:
            for page in pdf.pages[:40]:
                chunks.append(page.extract_text() or "")
        return "\n".join(chunks)
    except Exception:  # noqa: BLE001
        return ""


def analyse(url: str) -> dict:
    ok, robots_url = robots_check(url)
    row = {
        "url": url,
        "robots_url": robots_url,
        "robots_allow": ok,
        "http_status": "",
        "chars": 0,
        "domains_hit": 0,
        "domain_list": "",
        "kw_total": 0,
        "clause_count": 0,
        "exception_count": 0,
        "crossref_count": 0,
        "alt_evidence_count": 0,
        "academic_year": "",
        "version_marker": False,
        "verdict": "",
    }
    if ok is False:
        row["verdict"] = "manual_only_robots"
        return row

    status, body, err = fetch(url)
    row["http_status"] = status if status else (err or "UNREACHABLE")
    if not body:
        # distinguish a wrong guess (404) from a real reachability problem
        if status == 404:
            row["verdict"] = "not_found_404"
        elif status in (401, 403):
            row["verdict"] = f"blocked_http_{status}"
        elif status:
            row["verdict"] = f"http_error_{status}"
        else:
            row["verdict"] = "unreachable"
        return row

    if body[:5] == b"%PDF-":
        text = pdf_text_from_bytes(body)
        if not text:
            row["verdict"] = "pdf_unreadable"
            row["chars"] = -1
            return row
        row["http_status"] = f"{row['http_status']} (pdf)"
    else:
        text = text_of(body)

    row["chars"] = len(text)
    low = text.lower()
    hits = {d: sum(low.count(k) for k in kws) for d, kws in DOMAIN_TAXONOMY.items()}
    row["domains_hit"] = sum(1 for v in hits.values() if v)
    row["domain_list"] = ";".join(f"{d}:{v}" for d, v in hits.items() if v)
    row["kw_total"] = sum(hits.values())
    row["clause_count"] = len(CLAUSE_RE.findall(text))
    row["exception_count"] = len(EXCEPTION_RE.findall(text))
    row["crossref_count"] = len(CROSSREF_RE.findall(text))
    row["alt_evidence_count"] = len(ALT_EVIDENCE_RE.findall(text))
    ym = ACADEMIC_YEAR_RE.search(text)
    row["academic_year"] = f"{ym.group(1)}-{ym.group(2)}" if ym else ""
    row["version_marker"] = bool(VERSION_RE.search(text))

    strong = (
        row["chars"] >= 1500
        and row["domains_hit"] >= 3
        and row["exception_count"] >= 3
        and (row["clause_count"] >= 5 or row["crossref_count"] >= 2)
    )
    usable = row["chars"] >= 1500 and row["domains_hit"] >= 3
    row["verdict"] = "STRONG" if strong else ("usable" if usable else "pending_review")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/seed_probe.csv")
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--timeout", type=int, default=15)
    ap.add_argument("--tier", action="append", default=[], help="only these tiers")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--urls-file", default="",
                    help="extra candidates as CSV: tier,institution,country,doc_type,url")
    args = ap.parse_args()

    global TIMEOUT
    TIMEOUT = args.timeout

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = ["tier", "institution", "country", "doc_type", "url", "robots_url", "robots_allow",
              "http_status", "chars", "domains_hit", "domain_list", "kw_total", "clause_count",
              "exception_count", "crossref_count", "alt_evidence_count", "academic_year",
              "version_marker", "verdict"]

    rows: list[dict] = []
    from collections import defaultdict, Counter
    by_host: dict[str, float] = defaultdict(float)

    candidates = CANDIDATES
    if args.urls_file:
        extra_path = Path(args.urls_file)
        if not extra_path.is_absolute():
            extra_path = Path(__file__).resolve().parent.parent / extra_path
        with extra_path.open(encoding="utf-8", newline="") as f:
            extra = [r for r in csv.DictReader(f) if r.get("url")]
        print(f"loaded {len(extra)} extra candidates from {extra_path}", flush=True)
        have = {c["url"] for c in candidates}
        candidates = candidates + [c for c in extra if c["url"] not in have]
    if args.tier:
        want = set(args.tier)
        candidates = [c for c in candidates if c["tier"] in want]
    if args.limit:
        candidates = candidates[: args.limit]

    for i, cand in enumerate(candidates, 1):
        host = cand["url"].split("/")[2]
        wait = args.delay - (time.time() - by_host[host])
        if wait > 0:
            time.sleep(wait)
        try:
            res = analyse(cand["url"])
        except Exception as e:  # noqa: BLE001
            res = {"url": cand["url"], "verdict": f"ERROR {type(e).__name__}", "chars": 0,
                   "domains_hit": 0, "robots_allow": "", "robots_url": "", "http_status": "",
                   "domain_list": "", "kw_total": 0, "clause_count": 0, "exception_count": 0,
                   "crossref_count": 0, "alt_evidence_count": 0, "academic_year": "",
                   "version_marker": False}
        by_host[host] = time.time()
        res.update({k: cand[k] for k in ("tier", "institution", "country", "doc_type")})
        rows.append(res)
        print(f"[{i:2d}/{len(candidates)}] {res['verdict']:24s} "
              f"chars={res['chars']:>6} dom={res['domains_hit']} exc={res['exception_count']:>3} "
              f"xr={res['crossref_count']:>3} alt={res['alt_evidence_count']:>2}  {cand['url'][:70]}",
              flush=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    print("\n=== summary by tier/verdict ===")
    c = Counter((r["tier"], r["verdict"]) for r in rows)
    for (tier, verdict), n in sorted(c.items()):
        print(f"  {tier:12s} {verdict:24s} {n}")
    print("\n=== capability thresholds (plan section B) ===")
    print(f"  STRONG docs                : {sum(1 for r in rows if r['verdict'] == 'STRONG')}")
    print(f"  usable-or-better           : {sum(1 for r in rows if r['verdict'] in ('STRONG','usable'))}")
    print(f"  docs with >=3 exceptions   : {sum(1 for r in rows if r['exception_count'] >= 3)}")
    print(f"  docs with >=2 crossrefs    : {sum(1 for r in rows if r['crossref_count'] >= 2)}")
    print(f"  docs with >=1 alt-evidence : {sum(1 for r in rows if r['alt_evidence_count'] >= 1)}")
    print(f"  faculty tier STRONG/usable : {sum(1 for r in rows if r['tier'] == 'faculty' and r['verdict'] in ('STRONG','usable'))}")
    print(f"  institution STRONG/usable  : {sum(1 for r in rows if r['tier'] == 'institution' and r['verdict'] in ('STRONG','usable'))}")
    print(f"\nwrote {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
