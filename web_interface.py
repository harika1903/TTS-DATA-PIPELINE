#!/usr/bin/env python3
"""Web interface for the TTS dataset pipeline (for testing / demo).

A FastAPI app that lets you upload audio files, runs them through the full
custom-STT pipeline (STT + quality + music detection + speaker analysis +
dedup) as a background job, shows live progress and results, and lets you
download the resulting dataset.

RUN IT (on the GPU machine, in your venv):
    pip install fastapi uvicorn python-multipart
    python web_interface.py

Then open http://localhost:8200 in a browser. To expose it to your senior
remotely, use a Cloudflare Tunnel (see WEB_INTERFACE_README.md).

This wraps the SAME pipeline you've been running from the command line -- it
does not reimplement anything. It calls tts_pipeline_custom.process_file etc.
so the results are identical to the CLI.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ---- pipeline imports ----
from pipeline.config import PipelineConfig
from pipeline import audio_utils
import tts_pipeline_custom as pipe

app = FastAPI(title="TTS Dataset Pipeline")

# ---- server-side config (edit these to match your setup) ----
BASE = Path(__file__).resolve().parent
WORK = BASE / "web_workdir"
WORK.mkdir(exist_ok=True)

DEFAULTS = {
    "stt_url": "http://0.0.0.0:8123",
    "dnsmos_model": str(BASE / "dnsmos_model.onnx"),
    "hf_token": "",                     # set via the UI or here
    "silence_db": -32.0,
    "min_silence": 0.20,
    "speaker_device": "cuda",
    "speaker_db": str(BASE / "speaker_db_global.json"),
}

# ---- job tracking (in memory) ----
JOBS: Dict[str, dict] = {}


def _run_job(job_id: str, input_dir: Path, output_dir: Path, opts: dict):
    """Background worker: run the full pipeline on the uploaded files."""
    job = JOBS[job_id]
    try:
        cfg = PipelineConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            force=True,
            use_custom_stt=True,
            custom_stt_url=opts["stt_url"],
            force_language=opts.get("force_language") or None,
            silence_db_threshold=float(opts["silence_db"]),
            min_silence_sec=float(opts["min_silence"]),
            enable_dnsmos=bool(opts.get("dnsmos_model")),
            dnsmos_model_path=opts.get("dnsmos_model") or None,
            enable_bak_gate=True,
            dnsmos_reject_bak=3.0,   # reject clips with background music/noise (BAK below this)
            dnsmos_review_bak=4.0,   # review clips with some background (raise for stricter/cleaner)
            enable_speaker_analysis=bool(opts.get("hf_token")),
            hf_token=opts.get("hf_token") or None,
            speaker_device=opts.get("speaker_device", "cuda"),
            dominant_speaker_min_fraction=0.85,
            speaker_db_path=opts.get("speaker_db") or None,
            flag_numbers_for_review=False,
        )
        cfg.ensure_dirs()

        # check STT server reachable
        from pipeline import custom_stt
        ok, msg = custom_stt.check_server(cfg.custom_stt_url)
        if not ok:
            job["status"] = "error"
            job["error"] = f"STT server not reachable: {msg}"
            return

        files = pipe.discover_audio_files(input_dir)
        job["total_files"] = len(files)
        job["status"] = "running"

        def _progress(source_id, done, total):
            job["clip_done"] = done
            job["clip_total"] = total

        for i, f in enumerate(files, 1):
            job["current_file"] = f.name
            job["files_done"] = i - 1
            job["clip_done"] = 0
            job["clip_total"] = 0
            try:
                pipe.process_file(cfg, f, progress_callback=_progress)
            except audio_utils.AudioIntegrityError as e:
                job.setdefault("file_errors", []).append(f"{f.name}: integrity: {e}")
            except Exception as e:  # noqa: BLE001
                job.setdefault("file_errors", []).append(f"{f.name}: {type(e).__name__}: {e}")
            job["files_done"] = i

        pipe.run_dataset_postprocessing(cfg)

        # collect results
        from pipeline import manifest as m
        accepted = m.load_records(cfg.manifests_dir / "accepted.jsonl")
        quarantined = m.load_records(cfg.manifests_dir / "quarantined.jsonl")
        rejected = m.load_records(cfg.manifests_dir / "rejected.jsonl")
        # separate WHOLE-FILE rejections (multiple speakers, duplicate speaker,
        # etc.) so the UI can show them clearly, rather than burying them in
        # the per-clip reason counts.
        file_rejections = []
        for r in rejected:
            for reason in r.get("rejection_reasons", []):
                if reason.startswith("file_level:"):
                    detail = reason.split(":", 1)[1]
                    label = _file_reject_label(detail)
                    file_rejections.append({"file": r.get("source_filename", ""), "reason": label})
        job["results"] = {
            "accepted": len(accepted),
            "quarantined": len(quarantined),
            "rejected": len(rejected),
            "file_rejections": file_rejections,
            "accepted_clips": [
                {"id": r["utterance_id"], "source": r["source_filename"],
                 "text": r["text"], "duration": round(r["duration_sec"], 1),
                 "dnsmos_ovrl": r.get("dnsmos_ovrl"), "dnsmos_bak": r.get("dnsmos_bak")}
                for r in accepted[:200]
            ],
            "rejected_summary": _reason_summary(rejected, "rejection_reasons"),
            "quarantined_summary": _reason_summary(quarantined, "review_reasons"),
        }
        # zip the output for download
        zip_path = output_dir.parent / f"{output_dir.name}_dataset.zip"
        if zip_path.exists():
            zip_path.unlink()
        shutil.make_archive(str(zip_path.with_suffix("")), "zip", output_dir)
        job["download"] = str(zip_path)
        job["status"] = "done"
    except Exception as e:  # noqa: BLE001
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"


def _file_reject_label(detail):
    """Turn a file_level rejection detail into a readable sentence."""
    if detail.startswith("multiple_speakers_no_dominant"):
        return "Multiple speakers — no single dominant speaker (looks like a conversation/interview)"
    if detail.startswith("speaker_already_in_dataset"):
        return "This speaker already appears in another accepted file (duplicate speaker)"
    if detail.startswith("speaker_borderline_match"):
        return "Possibly the same speaker as another file (flagged for review)"
    if detail.startswith("speaker_analysis_failed"):
        return "Speaker analysis could not run on this file"
    return detail


def _reason_summary(records, field):
    from collections import Counter
    c = Counter()
    for r in records:
        for reason in r.get(field, []):
            c[reason.split(":")[0]] += 1
    # map the internal reason codes to human-readable descriptions
    descriptions = {
        "background_music_or_noise": "Background music or noise detected",
        "dnsmos": "Low audio quality (muffled / distorted / noisy)",
        "clipping": "Audio distortion from clipping (recorded too loud)",
        "exceeds_hard_cap": "Clip too long (speaker didn't pause; over 20s)",
        "below_min_clip_duration": "Clip too short (under 3 seconds)",
        "transcript": "Empty or unreadable transcript (no clear speech)",
        "stt_failed": "Transcription failed (server error or unsupported language)",
        "clip_overlaps_second_speaker": "A second speaker's voice appears in this clip",
        "file_level": "Whole file rejected (multiple speakers, or speaker already in dataset)",
        "speaker_borderline_match": "Possibly the same speaker as another file (needs review)",
        "cut_failed": "Could not extract the audio for this clip",
    }
    return [{"count": n, "code": code, "label": descriptions.get(code, code)}
            for code, n in c.most_common()]


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...),
                 hf_token: str = Form(""),
                 stt_url: str = Form(""),
                 language: str = Form(""),
                 silence_db: str = Form(""),
                 min_silence: str = Form("")):
    job_id = uuid.uuid4().hex[:12]
    job_dir = WORK / job_id
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for uf in files:
        # keep the original filename (the language prefix matters!)
        dest = input_dir / Path(uf.filename).name
        with open(dest, "wb") as out:
            shutil.copyfileobj(uf.file, out)
        saved.append(dest.name)

    opts = dict(DEFAULTS)
    if hf_token: opts["hf_token"] = hf_token
    if stt_url: opts["stt_url"] = stt_url
    if language: opts["force_language"] = language
    if silence_db: opts["silence_db"] = silence_db
    if min_silence: opts["min_silence"] = min_silence

    JOBS[job_id] = {
        "status": "queued", "files": saved, "total_files": len(saved),
        "files_done": 0, "current_file": None, "created": time.time(),
    }
    t = threading.Thread(target=_run_job, args=(job_id, input_dir, output_dir, opts), daemon=True)
    t.start()
    return {"job_id": job_id, "files": saved}


@app.post("/reset_speakers")
def reset_speakers():
    """Clear the global speaker database. Useful before re-testing files
    (otherwise a file's speaker is 'already registered' from a prior run and
    the file gets rejected as a duplicate). Does NOT touch any produced data.
    """
    from pathlib import Path as _P
    db_path = _P(DEFAULTS.get("speaker_db") or (BASE / "speaker_db_global.json"))
    existed = db_path.exists()
    try:
        if existed:
            db_path.unlink()
        # also clear any other speaker_db*.json next to the pipeline
        removed = 0
        for p in BASE.glob("speaker_db*.json"):
            p.unlink(missing_ok=True)
            removed += 1
        return {"ok": True, "message": f"Speaker database cleared ({removed} file(s) removed)."}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "message": f"Could not clear: {e}"}, status_code=500)


@app.get("/status/{job_id}")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return job


@app.get("/download/{job_id}")
def download(job_id: str):
    job = JOBS.get(job_id)
    if not job or "download" not in job:
        return JSONResponse({"error": "not ready"}, status_code=404)
    p = Path(job["download"])
    return FileResponse(str(p), filename=p.name, media_type="application/zip")


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TTS Pipeline</title>
<style>
body{margin:0;font-family:system-ui,sans-serif;background:#f4f5f7;color:#222;line-height:1.5}
.wrap{max-width:640px;margin:40px auto;padding:0 20px}
h1{font-size:22px;margin:0 0 24px;text-align:center}
.card{background:#fff;border:1px solid #e2e4e9;border-radius:10px;padding:24px;margin-bottom:16px}
.drop{border:1px solid #dde1ea;border-radius:12px;padding:36px 20px;text-align:center;cursor:pointer;transition:.15s;background:#fafbfc}
.drop:hover,.drop.over{border-color:#4c8bf5;background:#f2f7ff}
.drop input{display:none}
.dropicon{font-size:36px;margin-bottom:10px}
.droptitle{font-size:16px;font-weight:600;color:#2b3040;margin-bottom:2px}
.dropsub{font-size:13px;color:#8a90a0;margin-bottom:16px}
.choosebtn{display:inline-block;background:#4c8bf5;color:#fff;border-radius:8px;padding:9px 22px;font-size:14px;font-weight:600}
.hintline{color:#9aa0b0;font-size:12px;margin-top:14px}
.files{margin:14px 0;font-size:14px;color:#555}
.files span{display:inline-block;background:#f4f5f7;border-radius:5px;padding:3px 8px;margin:2px}
.btn{width:100%;background:#4c8bf5;color:#fff;border:0;border-radius:8px;padding:12px;font-size:16px;font-weight:600;cursor:pointer;margin-top:8px}
.btn:disabled{opacity:.5;cursor:not-allowed}
.tokenrow{margin:14px 0}
.tokenrow input{width:100%;padding:9px 11px;border:1px solid #e2e4e9;border-radius:8px;font-size:14px}
.tokenrow label{font-size:13px;color:#888;display:block;margin-bottom:4px}
.bar{height:10px;background:#e2e4e9;border-radius:5px;overflow:hidden;margin:14px 0 6px}
.bar>div{height:100%;background:#4c8bf5;width:0;transition:width .3s}
.stat{display:flex;gap:12px;margin:18px 0}
.stat .b{flex:1;text-align:center;padding:16px 8px;border-radius:8px;background:#f4f5f7}
.stat .n{font-size:26px;font-weight:700}
.ok .n{color:#2ea043}.wa .n{color:#bf8700}.ba .n{color:#e5484d}
.stat .l{font-size:12px;color:#888}
.dl{width:100%;background:#2ea043;color:#fff;border:0;border-radius:8px;padding:12px;font-size:15px;font-weight:600;cursor:pointer;margin-top:8px}
.hide{display:none}
.reasons{font-size:13px;color:#666;margin-top:14px}
.reasons b{color:#333}
.rgroup{margin:10px 0}
.rgroup .rhead{font-weight:600;font-size:13px;margin-bottom:6px}
.rgroup .rhead.rej{color:#e5484d}
.rgroup .rhead.rev{color:#bf8700}
.rrow{display:flex;align-items:center;gap:10px;padding:5px 0;border-top:1px solid #f0f0f0}
.rcount{flex:0 0 auto;min-width:34px;text-align:center;background:#f4f5f7;border-radius:6px;padding:2px 6px;font-size:12px;font-weight:700;color:#555}
.rlabel{flex:1;font-size:13.5px;color:#444}
.clip{border-top:1px solid #eee;padding:8px 0;font-size:14px}
.clip .m{color:#999;font-size:12px}
details summary{cursor:pointer;color:#4c8bf5;font-size:14px;margin-top:12px}
.err{color:#e5484d;font-size:14px}
.resetrow{margin-top:14px;text-align:center}
.resetlink{color:#8a90a0;font-size:13px;text-decoration:none;border-bottom:1px dashed #c5cad5}
.resetlink:hover{color:#4c8bf5}
.reset-hint{color:#b0b5c0;font-size:12px}
.filerej{background:#fdeeee;border:1px solid #f5c9c9;border-radius:8px;padding:12px 14px;margin:12px 0}
.frtitle{font-weight:600;color:#c23b3b;font-size:13.5px;margin-bottom:6px}
.frrow{font-size:13px;color:#555;padding:3px 0}
</style></head>
<body><div class="wrap">
<h1>TTS Dataset Pipeline</h1>

<div class="card" id="upcard">
  <input type="file" id="fileinput" multiple accept="audio/*,.wav,.mp3,.flac,.m4a" style="position:absolute;width:1px;height:1px;opacity:0;overflow:hidden;left:-9999px">
  <div class="drop" id="drop">
    <div class="dropicon">🎵</div>
    <div class="droptitle">Upload audio files</div>
    <div class="dropsub">Drag &amp; drop here, or click to browse</div>
    <div class="hintline">Name files with a language code — tel_ (Telugu), kan_ (Kannada), hin_ (Hindi)</div>
  </div>
  <div class="files" id="filelist"></div>
  <div class="tokenrow">
    <label>Language of these files</label>
    <select id="language" style="width:100%;padding:9px 11px;border:1px solid #e2e4e9;border-radius:8px;font-size:14px;background:#fff">
      <option value="">Auto (from filename prefix)</option>
      <option value="hindi">Hindi</option>
      <option value="telugu">Telugu</option>
      <option value="kannada">Kannada</option>
      <option value="tamil">Tamil</option>
      <option value="malayalam">Malayalam</option>
      <option value="bengali">Bengali</option>
      <option value="gujarati">Gujarati</option>
      <option value="english">English</option>
    </select>
    <div style="font-size:12px;color:#9aa0b0;margin-top:4px">Pick a language to process all uploaded files as that language — no filename prefix needed.</div>
  </div>
  <div class="tokenrow">
    <label>Hugging Face token — needed for speaker checks (leave blank to skip)</label>
    <input id="hf_token" placeholder="hf_...">
  </div>
  <button class="btn" id="runbtn" disabled>Run</button>
  <div class="resetrow"><a href="#" id="resetbtn" class="resetlink">Reset speaker database</a> <span class="reset-hint">— clear before re-testing the same files</span></div>
</div>

<div class="card hide" id="progcard">
  <div id="progtitle" style="font-weight:600">Processing…</div>
  <div class="bar"><div id="progbar"></div></div>
  <div class="hintline" id="progtext"></div>
</div>

<div class="card hide" id="rescard">
  <div class="stat">
    <div class="b ok"><div class="n" id="r_acc">0</div><div class="l">Accepted</div></div>
    <div class="b wa"><div class="n" id="r_quar">0</div><div class="l">Review</div></div>
    <div class="b ba"><div class="n" id="r_rej">0</div><div class="l">Rejected</div></div>
  </div>
  <div class="filerej hide" id="filerej"></div>
  <button class="dl" id="dlbtn">Download dataset (.zip)</button>
  <details><summary>Why clips were rejected / flagged</summary><div class="reasons" id="reasons"></div></details>
  <details><summary>Accepted transcripts</summary><div id="clips"></div></details>
</div>

<div class="card hide" id="errcard"><div class="err" id="errmsg"></div></div>

<script>
let chosen=[];
const drop=document.getElementById('drop'),fi=document.getElementById('fileinput'),fl=document.getElementById('filelist'),runbtn=document.getElementById('runbtn');
drop.onclick=()=>fi.click();
fi.onchange=()=>setFiles([...fi.files]);
['dragover','dragenter'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('over')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('over')}));
drop.addEventListener('drop',ev=>setFiles([...ev.dataTransfer.files]));
function setFiles(f){chosen=f;fl.innerHTML=f.map(x=>'<span>'+x.name+'</span>').join('');runbtn.disabled=f.length===0;}

runbtn.onclick=async()=>{
  const fd=new FormData();
  chosen.forEach(f=>fd.append('files',f));
  fd.append('hf_token',document.getElementById('hf_token').value);
  fd.append('language',document.getElementById('language').value);
  runbtn.disabled=true;
  document.getElementById('progcard').classList.remove('hide');
  document.getElementById('rescard').classList.add('hide');
  document.getElementById('errcard').classList.add('hide');
  const r=await fetch('/upload',{method:'POST',body:fd});
  const j=await r.json();
  poll(j.job_id);
};

async function poll(id){
  const r=await fetch('/status/'+id);const j=await r.json();
  const total=j.total_files||1,done=j.files_done||0;
  const ctot=j.clip_total||0, cdone=j.clip_done||0;
  // if we're inside a file processing clips, show clip-level progress
  let pct, txt;
  if(ctot>0 && done<total){
    pct=Math.round(cdone/ctot*100);
    txt=(j.current_file?j.current_file+' — ':'')+'clip '+cdone+' / '+ctot;
  } else {
    pct=Math.round(done/total*100);
    txt=(j.current_file?('Processing '+j.current_file+'  ·  '):'')+done+' / '+total+' files';
  }
  document.getElementById('progbar').style.width=pct+'%';
  document.getElementById('progtext').textContent=txt;
  if(j.status==='done'){showResults(id,j);return;}
  if(j.status==='error'){document.getElementById('progcard').classList.add('hide');document.getElementById('errcard').classList.remove('hide');document.getElementById('errmsg').textContent=j.error;return;}
  setTimeout(()=>poll(id),1000);
}

function showResults(id,j){
  document.getElementById('progcard').classList.add('hide');
  document.getElementById('rescard').classList.remove('hide');
  const res=j.results;
  r_acc.textContent=res.accepted;r_quar.textContent=res.quarantined;r_rej.textContent=res.rejected;
  // whole-file rejections (shown prominently at the top)
  const fz=document.getElementById('filerej');
  if(res.file_rejections&&res.file_rejections.length){
    let h='<div class="frtitle">Whole files rejected:</div>';
    res.file_rejections.forEach(fr=>{h+='<div class="frrow"><b>'+escapeHtml(fr.file)+'</b> — '+escapeHtml(fr.reason)+'</div>';});
    fz.innerHTML=h;fz.classList.remove('hide');
  } else { fz.classList.add('hide'); }
  const rz=document.getElementById('reasons');rz.innerHTML='';
  const rej=res.rejected_summary,quar=res.quarantined_summary;
  function renderGroup(title, cls, items){
    if(!items||!items.length)return '';
    let h='<div class="rgroup"><div class="rhead '+cls+'">'+title+'</div>';
    items.forEach(it=>{h+='<div class="rrow"><span class="rcount">'+it.count+'</span><span class="rlabel">'+escapeHtml(it.label)+'</span></div>';});
    return h+'</div>';
  }
  let html='';
  html+=renderGroup('Rejected — removed from dataset','rej',rej);
  html+=renderGroup('Flagged for review — set aside for a human to check','rev',quar);
  if(!html)html='<div>Nothing rejected or flagged — all clips accepted.</div>';
  rz.innerHTML=html;
  const cz=document.getElementById('clips');cz.innerHTML='';
  res.accepted_clips.forEach(c=>{cz.innerHTML+='<div class="clip">'+escapeHtml(c.text||'(no text)')+'<div class="m">'+c.source+' · '+c.duration+'s · MOS '+(c.dnsmos_ovrl?c.dnsmos_ovrl.toFixed(2):'—')+'</div></div>';});
  if(!res.accepted_clips.length)cz.innerHTML='No accepted clips.';
  document.getElementById('dlbtn').onclick=()=>location.href='/download/'+id;
}
function escapeHtml(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

document.getElementById('resetbtn').onclick=async(e)=>{
  e.preventDefault();
  if(!confirm('Clear the speaker database? This lets you re-test the same files without them being rejected as duplicate speakers.'))return;
  const r=await fetch('/reset_speakers',{method:'POST'});
  const j=await r.json();
  alert(j.message||(j.ok?'Cleared.':'Failed.'));
};
</script>
</div></body></html>"""


if __name__ == "__main__":
    import uvicorn
    print("Starting TTS Pipeline web interface on http://0.0.0.0:8200")
    print("Open http://localhost:8200 in your browser.")
    uvicorn.run(app, host="0.0.0.0", port=8200)
