"""09_annotate_server.py - local annotation workbench for the gold question set.

Runs a small HTTP server on 127.0.0.1 only (never exposed to the network) and serves a
single-page editor: the document's evidence text on the left with its clauses, the
question form on the right, and live validation on every save.

Why a workbench instead of a spreadsheet: the protocol anchors every gold span to exact
character offsets in the *evidence* layer, and a clause counts as covered only when the
retrieved union contains the whole annotated span. Hand-counting offsets is error-prone;
here the span is produced by selecting text, and the server re-checks it against the
corpus before accepting the question.

The tool does not write questions for the researcher. It pre-fills candidate clauses and
skeletons (see 08_prefill_candidates.py) and refuses to let a prefilled question enter the
held-out test split until `manual_reviewed` is set by hand.

Usage:
    .venv\\Scripts\\python.exe scripts/09_annotate_server.py            # http://127.0.0.1:8765
    .venv\\Scripts\\python.exe scripts/09_annotate_server.py --port 8770
    .venv\\Scripts\\python.exe scripts/09_annotate_server.py --no-browser
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    PROJECT_ROOT, load_settings, load_taxonomy, project_paths, read_jsonl, write_jsonl,
)
import goldset  # noqa: E402

STATE: dict = {}


# --------------------------------------------------------------------- data access
def questions_path() -> Path:
    return STATE["paths"]["annotations"] / "questions.jsonl"


def load_questions() -> list[dict]:
    return read_jsonl(questions_path())


def save_questions(questions: list[dict]) -> None:
    """Atomic replace: never leave a half-written questions.jsonl behind."""
    target = questions_path()
    tmp = target.with_suffix(".jsonl.tmp")
    write_jsonl(tmp, questions)
    tmp.replace(target)


def corpus_payload() -> dict:
    corpus: goldset.Corpus = STATE["corpus"]
    docs = []
    for doc_id in sorted(corpus.by_doc):
        clauses = corpus.by_doc[doc_id]
        docs.append({
            "doc_id": doc_id,
            "tier": corpus.doc_tier.get(doc_id, ""),
            "institution": corpus.doc_institution.get(doc_id, ""),
            "clauses": len(clauses),
            "chars": len(goldset.evidence_text(corpus, doc_id)),
            "usable": sum(1 for c in clauses if c.char_len >= 80 and not c.boilerplate),
        })
    return {"corpus_version": corpus.corpus_version, "documents": docs}


def doc_payload(doc_id: str) -> dict:
    corpus: goldset.Corpus = STATE["corpus"]
    if doc_id not in corpus.by_doc:
        return {"error": f"unknown document {doc_id}"}
    text = goldset.evidence_text(corpus, doc_id)
    clauses = [{
        "clause_id": c.clause_id,
        "char_start": c.char_start,
        "char_end": c.char_end,
        "char_len": c.char_len,
        "section_path": c.section_path,
        "has_exception": c.has_exception,
        "has_crossref": c.has_crossref,
        "has_alt_evidence": c.has_alt_evidence,
        "boilerplate": c.boilerplate,
        "usable": c.char_len >= 80 and not c.boilerplate,
        "text": c.text,
    } for c in corpus.by_doc[doc_id]]
    return {"doc_id": doc_id, "text": text, "clauses": clauses,
            "tier": corpus.doc_tier.get(doc_id, ""),
            "institution": corpus.doc_institution.get(doc_id, "")}


def stats_payload() -> dict:
    questions = load_questions()
    corpus: goldset.Corpus = STATE["corpus"]
    g = goldset.test_group_counts(questions)
    cfg = STATE["cfg"]
    ann = cfg["annotation"]
    by_domain = Counter(q.get("domain", "?") for q in questions)
    by_type = Counter(q.get("evidence_type", "?") for q in questions)
    skeletons = STATE.get("skeletons", [])
    return {
        "questions": len(questions),
        "dev": g["counts"]["dev"],
        "test": g["counts"]["test"],
        "dev_groups": g["groups"]["dev"],
        "test_groups": g["groups"]["test"],
        "manual_reviewed": g["manual_reviewed"],
        "prefilled": g["prefilled"],
        "targets": {
            "total": ann["target_questions_total"],
            "dev": ann["target_dev_questions"],
            "test": ann["target_test_questions"],
            "test_groups": ann["target_test_groups_min"],
        },
        "by_domain": dict(by_domain),
        "by_type": dict(by_type),
        "skeletons_available": len(skeletons),
        "corpus_version": corpus.corpus_version,
        "corpus_docs": len(corpus.by_doc),
        "lock": STATE.get("lock"),
    }


def validate_payload(question: dict) -> dict:
    corpus: goldset.Corpus = STATE["corpus"]
    cfg = STATE["cfg"]
    taxonomy = STATE["taxonomy"]
    problems = goldset.validate_question(
        question, corpus,
        valid_evidence_types=cfg["annotation"]["evidence_types"],
        valid_domains=list(taxonomy.get("domains", {}).keys()),
        valid_levels=cfg["annotation"]["source_levels"],
        require_manual_review_for_test=True,
    )
    # group crossing check against the stored set
    others = [q for q in load_questions() if q.get("qid") != question.get("qid")]
    gid = str(question.get("group_id", ""))
    same = [q for q in others if str(q.get("group_id", "")) == gid]
    if gid and same:
        splits = {q.get("split") for q in same} | {question.get("split")}
        if len(splits) > 1:
            problems.append(f"group {gid} would appear in both splits {sorted(splits)}")
    return {"ok": not problems, "problems": problems}


# --------------------------------------------------------------------- frontend
INDEX_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Gold annotation workbench</title>
<style>
 :root{--bg:#0f1115;--panel:#171a21;--line:#2a2f3a;--fg:#e6e9ef;--dim:#98a2b3;
       --accent:#4f8cff;--ok:#3fb950;--warn:#d29922;--bad:#f85149}
 *{box-sizing:border-box}
 body{margin:0;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
      background:var(--bg);color:var(--fg);height:100vh;display:flex;flex-direction:column}
 header{padding:8px 12px;border-bottom:1px solid var(--line);display:flex;gap:14px;
        align-items:center;background:var(--panel);flex-wrap:wrap}
 header b{color:var(--accent)}
 .stat{color:var(--dim)} .stat span{color:var(--fg)}
 main{display:grid;grid-template-columns:1fr 460px;flex:1;min-height:0}
 #left{display:flex;flex-direction:column;min-width:0;border-right:1px solid var(--line)}
 #docbar{padding:6px 10px;border-bottom:1px solid var(--line);display:flex;gap:8px;
         align-items:center;background:var(--panel)}
 #docbar select{width:340px}
 #docbar input{width:120px}
 #ev{flex:1;overflow:auto;padding:14px 18px;white-space:pre-wrap;word-wrap:break-word;
     font-size:13px;line-height:1.65}
 .cl{border-radius:3px;cursor:pointer;padding:1px 0}
 .cl.usable{background:rgba(79,140,255,.07)}
 .cl.exc{background:rgba(210,153,34,.12);box-shadow:inset 0 -2px 0 rgba(210,153,34,.45)}
 .cl.xr{background:rgba(63,185,80,.10)}
 .cl.alt{background:rgba(190,120,255,.10)}
 .cl.boiler{opacity:.55;text-decoration:line-through}
 .cl.hover{outline:1px solid var(--accent)}
 #right{display:flex;flex-direction:column;min-height:0;background:var(--panel)}
 #form{overflow:auto;padding:12px;flex:1}
 label{display:block;margin:8px 0 3px;color:var(--dim);font-size:11px;
       text-transform:uppercase;letter-spacing:.04em}
 input,select,textarea{width:100%;background:#0d1015;color:var(--fg);border:1px solid var(--line);
       border-radius:4px;padding:5px 7px;font:inherit}
 textarea{resize:vertical}
 .row{display:flex;gap:8px}.row>div{flex:1}
 button{background:var(--accent);color:#fff;border:0;border-radius:4px;padding:6px 11px;
        font:inherit;cursor:pointer}
 button.ghost{background:#222834;color:var(--fg)}
 button:disabled{opacity:.5;cursor:not-allowed}
 #sets{margin-top:6px}
 .set{border:1px solid var(--line);border-radius:5px;padding:6px;margin-bottom:6px}
 .set h4{margin:0 0 4px;font-size:11px;color:var(--dim);display:flex;justify-content:space-between}
 .ref{background:#0d1015;border-left:2px solid var(--accent);padding:3px 6px;margin:3px 0;
      font-size:11px;max-height:80px;overflow:auto;cursor:pointer}
 .ref .cid{color:var(--accent)}
 #msg{padding:8px 12px;border-top:1px solid var(--line);min-height:34px;font-size:12px}
 .ok{color:var(--ok)} .bad{color:var(--bad)} .warn{color:var(--warn)}
 #sublist{max-height:160px;overflow:auto;border:1px solid var(--line);border-radius:4px;
          margin-top:6px}
 .subitem{padding:4px 7px;border-bottom:1px solid var(--line);cursor:pointer;font-size:11px}
 .subitem:hover{background:#1d2230}
 .subitem .meta{color:var(--dim)}
</style></head><body>
<header>
  <b>Gold annotation workbench</b>
  <div class="stat">corpus <span id="cv">-</span></div>
  <div class="stat">questions <span id="sq">0</span></div>
  <div class="stat">dev/test <span id="sdt">0/0</span></div>
  <div class="stat">groups test <span id="sg">0</span></div>
  <div class="stat">reviewed <span id="sr">0</span></div>
  <div class="stat" id="lockstate"></div>
</header>
<main>
  <div id="left">
    <div id="docbar">
      <select id="docsel"></select>
      <input id="clausefilter" placeholder="filter clauses">
      <button class="ghost" id="loadskel">load skeleton</button>
      <button class="ghost" id="newq">new blank</button>
    </div>
    <div id="ev"></div>
  </div>
  <div id="right">
    <div id="form">
      <label>qid</label><input id="qid" placeholder="Q0001">
      <label>question (write it yourself)</label><textarea id="question" rows="3"></textarea>
      <label>reference answer (write it yourself)</label><textarea id="answer" rows="3"></textarea>
      <div class="row">
        <div><label>group_id</label><input id="group"></div>
        <div><label>split</label><select id="split"><option>dev</option><option>test</option></select></div>
      </div>
      <div class="row">
        <div><label>evidence type</label><select id="etype"></select></div>
        <div><label>domain</label><select id="domain"></select></div>
      </div>
      <div class="row">
        <div><label>source level</label><select id="level"></select></div>
        <div><label>condition</label><select id="cond"></select></div>
      </div>
      <label>evidence sets</label>
      <div id="sets"></div>
      <div class="row" style="margin-top:6px">
        <div><button class="ghost" id="addset">+ sufficient set</button></div>
        <div><button class="ghost" id="addsel">+ selection as clause in set 1</button></div>
      </div>
      <label style="margin-top:10px">reviewed by me</label>
      <select id="reviewed"><option value="false">false (prefilled)</option>
        <option value="true">true (I verified every field)</option></select>
      <div class="row" style="margin-top:12px">
        <div><button id="save">save &amp; validate</button></div>
        <div><button class="ghost" id="delete">delete this qid</button></div>
      </div>
      <div id="sublist"></div>
    </div>
    <div id="msg"></div>
  </div>
</main>
<script>
const S={corpus:null,doc:null,sets:[],skeletons:[],skelIdx:0};
const $=id=>document.getElementById(id);
const esc=t=>t.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

async function api(path,body){
  const opt=body?{method:'POST',headers:{'Content-Type':'application/json'},
                  body:JSON.stringify(body)}:{};
  const r=await fetch(path,opt); return r.json();
}
function fill(sel,values){$(sel).innerHTML=values.map(v=>`<option>${v}</option>`).join('');}

async function boot(){
  const c=await api('/api/corpus'); S.corpus=c;
  $('cv').textContent=c.corpus_version;
  $('docsel').innerHTML=c.documents.map(d=>
    `<option value="${d.doc_id}">${d.doc_id.slice(0,58)} [${d.tier}] ${d.usable}c</option>`).join('');
  const cfg=await api('/api/config');
  fill('etype',cfg.evidence_types); fill('domain',cfg.domains);
  fill('level',cfg.source_levels); fill('cond',cfg.conditions);
  const sk=await api('/api/skeletons'); S.skeletons=sk.skeletons||[];
  await loadDoc($('docsel').value);
  await refreshStats();
}

async function loadDoc(docId){
  const d=await api('/api/doc?doc_id='+encodeURIComponent(docId)); S.doc=d;
  renderEvidence();
}

function renderEvidence(){
  const d=S.doc; if(!d) return;
  const cls={}; (d.clauses||[]).forEach(c=>cls[c.clause_id]=c);
  // walk clauses in order; any text not covered by a clause is emitted verbatim
  let html='',cursor=0;
  const ordered=[...d.clauses].sort((a,b)=>a.char_start-b.char_start);
  for(const c of ordered){
    if(c.char_start<cursor) continue;
    if(c.char_start>cursor) html+=esc(d.text.slice(cursor,c.char_start));
    const kinds=['cl'];
    if(c.usable) kinds.push('usable');
    if(c.has_exception) kinds.push('exc');
    if(c.has_crossref) kinds.push('xr');
    if(c.has_alt_evidence) kinds.push('alt');
    if(c.boilerplate) kinds.push('boiler');
    html+=`<span class="${kinds.join(' ')}" data-cid="${c.clause_id}"
             title="${esc(c.clause_id)}">${esc(d.text.slice(c.char_start,c.char_end))}</span>`;
    cursor=c.char_end;
  }
  html+=esc(d.text.slice(cursor));
  $('ev').innerHTML=html;
  $('ev').querySelectorAll('.cl').forEach(el=>{
    el.onclick=()=>addClauseRef(el.dataset.cid);
  });
}

function offsetIn(root,node,off){
  const w=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n,total=0;
  while(n=w.nextNode()){ if(n===node) return total+off; total+=n.textContent.length; }
  return total;
}

function currentSelection(){
  const sel=window.getSelection(); if(!sel.rangeCount) return null;
  const r=sel.getRangeAt(0); const ev=$('ev');
  if(!ev.contains(r.startContainer)||!ev.contains(r.endContainer)) return null;
  const a=offsetIn(ev,r.startContainer,r.startOffset);
  const b=offsetIn(ev,r.endContainer,r.endOffset);
  if(b<=a) return null;
  return {start:a,end:b,quote:S.doc.text.slice(a,b)};
}

function clauseById(id){
  return (S.doc.clauses||[]).find(c=>c.clause_id===id);
}

function addClauseRef(cid){
  const c=clauseById(cid)||{clause_id:cid,char_start:0,char_end:0};
  addRef({doc_id:S.doc.doc_id,version:S.corpus.corpus_version,clause_id:cid,
          section_path:c.section_path||'',char_start:c.char_start,char_end:c.char_end,
          quote_evidence:S.doc.text.slice(c.char_start,c.char_end),
          token_start:null,token_end:null,source:'clause'});
}

function addRef(ref){
  if(!S.sets.length) S.sets.push({set_id:'S1',clauses:[]});
  S.sets[0].clauses.push(ref); renderSets();
}

function addSelectionRef(){
  const s=currentSelection(); if(!s){ msg('select text inside the document first','warn'); return; }
  addRef({doc_id:S.doc.doc_id,version:S.corpus.corpus_version,clause_id:'',
          section_path:'',char_start:s.start,char_end:s.end,quote_evidence:s.quote,
          token_start:null,token_end:null,source:'selection'});
  msg(`added custom span ${s.start}-${s.end} (needs a clause_id before saving)`,'warn');
}

function renderSets(){
  const box=$('sets');
  if(!S.sets.length){ box.innerHTML='<div class="stat">no evidence sets yet</div>'; return; }
  box.innerHTML=S.sets.map((s,si)=>`
    <div class="set"><h4>set ${si+1} · ${s.set_id}
      <span><button class="ghost" data-rm="${si}" style="padding:1px 6px">remove</button></span></h4>
      ${(s.clauses||[]).map((r,ri)=>`
        <div class="ref" title="click to locate">
          <span class="cid">${esc(r.clause_id||'(no clause_id)')}</span>
          <span class="stat"> ${r.char_start}-${r.char_end}</span><br>
          ${esc((r.quote_evidence||'').slice(0,220))}
          <button class="ghost" data-rmref="${si}:${ri}" style="float:right;padding:0 5px">x</button>
        </div>`).join('')}
    </div>`).join('');
  box.querySelectorAll('[data-rm]').forEach(b=>b.onclick=()=>{
    S.sets.splice(+b.dataset.rm,1); renderSets(); });
  box.querySelectorAll('[data-rmref]').forEach(b=>b.onclick=e=>{
    e.stopPropagation(); const [a,c]=b.dataset.rmref.split(':').map(Number);
    S.sets[a].clauses.splice(c,1); renderSets(); });
}

function collect(){
  return {qid:$('qid').value.trim(),question:$('question').value.trim(),
    reference_answer:$('answer').value.trim(),group_id:$('group').value.trim(),
    split:$('split').value,evidence_type:$('etype').value,domain:$('domain').value,
    source_level:$('level').value,condition:$('cond').value,
    evidence_sets:S.sets.map(s=>({set_id:s.set_id,clauses:s.clauses})),
    prefilled:true,manual_reviewed:$('reviewed').value==='true',
    provenance:{generated_by:'09_annotate_server.py',skeleton_id:S.currentSkeleton||null}};
}

function loadInto(q){
  $('qid').value=q.qid||''; $('question').value=q.question||'';
  $('answer').value=q.reference_answer||''; $('group').value=q.group_id||'';
  $('split').value=q.split||'dev'; $('etype').value=q.evidence_type||'single-clause';
  $('domain').value=q.domain||'grading'; $('level').value=q.source_level||'institution';
  $('cond').value=q.condition||'baseline';
  $('reviewed').value=q.manual_reviewed?'true':'false';
  S.sets=(q.evidence_sets||[]).map(s=>({set_id:s.set_id,clauses:s.clauses||[]}));
  renderSets();
}

function msg(t,cls){ const m=$('msg'); m.className=cls||''; m.textContent=t; }

async function refreshStats(){
  const s=await api('/api/stats');
  $('sq').textContent=`${s.questions}/${s.targets.total}`;
  $('sdt').textContent=`${s.dev}/${s.targets.dev} · ${s.test}/${s.targets.test}`;
  $('sg').textContent=`${s.test_groups}/${s.targets.test_groups}`;
  $('sr').textContent=`${s.manual_reviewed}/${s.questions}`;
  $('lockstate').innerHTML=s.lock
    ? `<span class="warn">LOCKED ${s.lock.locked_at}</span>` : '';
  // suggestions: existing qids to load
  const qs=await api('/api/questions');
  $('sublist').innerHTML=(qs.questions||[]).slice(-40).reverse().map(q=>
    `<div class="subitem" data-qid="${esc(q.qid||'')}">
       <span class="cid">${esc(q.qid||'')}</span> <span class="meta">${esc((q.question||'').slice(0,70))}</span>
     </div>`).join('');
  $('sublist').querySelectorAll('.subitem').forEach(el=>el.onclick=async()=>{
    const q=await api('/api/question?qid='+encodeURIComponent(el.dataset.qid));
    if(q.question) loadInto(q.question);
  });
}

$('save').onclick=async()=>{
  const q=collect();
  if(!q.qid){ msg('qid required','bad'); return; }
  const r=await api('/api/save',q);
  if(r.ok){ msg('saved. '+((r.warnings||[]).join(' | ')||'no problems'),'ok'); refreshStats(); }
  else { msg('REJECTED: '+r.problems.join(' | '),'bad'); }
};
$('delete').onclick=async()=>{
  const qid=$('qid').value.trim(); if(!qid) return;
  const r=await api('/api/delete',{qid}); msg(r.ok?'deleted':'not found',r.ok?'ok':'bad');
  refreshStats();
};
$('addset').onclick=()=>{ S.sets.push({set_id:'S'+(S.sets.length+1),clauses:[]}); renderSets(); };
$('addsel').onclick=addSelectionRef;
$('newq').onclick=()=>{ S.sets=[]; loadInto({split:'dev'}); msg('blank form',''); };
$('loadskel').onclick=()=>{
  if(!S.skeletons.length){ msg('no skeletons; run 08_prefill_candidates.py','warn'); return; }
  const s=S.skeletons[S.skelIdx % S.skeletons.length]; S.skelIdx++; S.currentSkeleton=s.skeleton_id;
  loadInto({qid:'',question:s.skeleton_question,reference_answer:'',group_id:s.group_id,
    split:'dev',evidence_type:s.evidence_type,domain:s.domain,source_level:s.source_level,
    condition:s.condition,evidence_sets:[{set_id:'S1',clauses:s.candidate_clause_refs}]});
  $('docsel').value=s.candidate_clause_refs[0]?.doc_id||$('docsel').value;
  loadDoc($('docsel').value);
  msg(`loaded ${s.skeleton_id} (${s.confidence} confidence). Rewrite the question and answer, then set reviewed=true.`,'warn');
};
$('docsel').onchange=()=>loadDoc($('docsel').value);
$('clausefilter').oninput=e=>{
  const q=e.target.value.toLowerCase();
  $('ev').querySelectorAll('.cl').forEach(el=>{
    el.style.outline=(q && el.dataset.cid.toLowerCase().includes(q))?'1px solid #4f8cff':'';
  });
};
boot();
</script></body></html>
"""


# --------------------------------------------------------------------- HTTP layer
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter console
        return

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        qs = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif u.path == "/api/corpus":
                self._json(corpus_payload())
            elif u.path == "/api/config":
                cfg = STATE["cfg"]
                self._json({
                    "evidence_types": cfg["annotation"]["evidence_types"],
                    "domains": list(STATE["taxonomy"].get("domains", {}).keys()),
                    "source_levels": cfg["annotation"]["source_levels"],
                    "conditions": cfg["annotation"]["conditions"],
                })
            elif u.path == "/api/doc":
                self._json(doc_payload(qs.get("doc_id", [""])[0]))
            elif u.path == "/api/stats":
                self._json(stats_payload())
            elif u.path == "/api/skeletons":
                self._json({"skeletons": STATE.get("skeletons", [])})
            elif u.path == "/api/questions":
                self._json({"questions": load_questions()})
            elif u.path == "/api/question":
                qid = qs.get("qid", [""])[0]
                found = next((q for q in load_questions() if q.get("qid") == qid), None)
                self._json({"question": found})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError as e:
            self._json({"ok": False, "problems": [f"bad JSON: {e}"]}, 400)
            return

        try:
            if u.path == "/api/save":
                lock = STATE.get("lock")
                if lock and body.get("split") == "test":
                    existing = [q for q in load_questions() if q.get("qid") == body.get("qid")]
                    if not existing:
                        self._json({"ok": False, "problems": [
                            f"held-out test split is locked ({lock.get('locked_at')}); "
                            "unlock with 05_audit_splits.py --unlock before adding test questions"]})
                        return
                result = validate_payload(body)
                if not result["ok"]:
                    self._json({"ok": False, "problems": result["problems"]})
                    return
                questions = load_questions()
                idx = next((i for i, q in enumerate(questions)
                            if q.get("qid") == body.get("qid")), None)
                if idx is None:
                    questions.append(body)
                else:
                    questions[idx] = body
                save_questions(questions)
                warnings = []
                if body.get("split") == "test" and not body.get("manual_reviewed"):
                    warnings.append("test question is not manual_reviewed yet")
                self._json({"ok": True, "warnings": warnings})
            elif u.path == "/api/delete":
                qid = body.get("qid")
                questions = [q for q in load_questions() if q.get("qid") != qid]
                save_questions(questions)
                self._json({"ok": True})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            self._json({"ok": False, "problems": [f"{type(e).__name__}: {e}"]}, 500)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    paths = project_paths(root)
    cfg = load_settings(root)
    taxonomy = load_taxonomy(root)

    corpus = goldset.load_corpus(root)
    if not corpus.clauses:
        print("no corpus loaded; run 02_extract_clean.py and curate first", file=sys.stderr)
        return 2

    skel_path = paths["annotations"] / "question_skeletons.jsonl"
    skeletons = read_jsonl(skel_path) if skel_path.exists() else []
    lock_path = paths["annotations"] / "LOCKED" / "LOCKED.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else None

    STATE.update({"paths": paths, "cfg": cfg, "taxonomy": taxonomy,
                  "corpus": corpus, "skeletons": skeletons, "lock": lock})

    if not questions_path().exists():
        save_questions([])

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print("=== annotation workbench ===")
    print(f"  corpus      : {corpus.corpus_version} "
          f"({len(corpus.by_doc)} documents, {len(corpus.clauses)} clauses)")
    print(f"  skeletons   : {len(skeletons)}")
    print(f"  questions   : {len(load_questions())}")
    print(f"  lock        : {'LOCKED ' + str(lock.get('locked_at')) if lock else 'not locked'}")
    print(f"  serving     : {url}  (127.0.0.1 only)")
    print("  Ctrl+C to stop")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
