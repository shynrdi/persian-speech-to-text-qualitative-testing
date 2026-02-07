#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_asr_report_v3.py  (FAIR benchmark + REALTIME selection + clean report)

Folder:
test_samples/
  Ardalan/Ardalan01.mp3 ... Ardalan18.mp3
  Shayan/Shayan01.mp3 ... Shayan18.mp3

Backends:
A) transformers_whisper (HF pipeline)  GPU: fp16 -> bf16 -> fp32  | CPU: fp32
   - Adds decoding guards to reduce repetitions ("این این این...")

B) faster_whisper (CTranslate2)
   - GPU: float16 + int8_float16
   - CPU: int8
   - Skips tiny + int8_float16 on GPU (often flaky)

Fairness & measurement:
- Measures model load_time separately from inference_time.
- Does warmup runs (NOT counted in latency stats).
- Latency measured per sample: inference_only_time (after warmup, sync).

Outputs in out_dir:
- asr_merged_results.csv              (per-sample results)
- asr_summary_all.csv                 (grouped summary)
- asr_pareto.csv                      (pareto frontier)
- asr_report.html                     (tables + SVG charts)
- asr_realtime_recommendation.txt     (simple suggestion for realtime)

Run:
  python make_asr_report_v3.py --samples_dir ./test_samples --out_dir ./report_gpu --device gpu --normalize
  python make_asr_report_v3.py --samples_dir ./test_samples --out_dir ./report_cpu --device cpu --normalize
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import librosa
from jiwer import wer, cer
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline as hf_pipeline

try:
    from faster_whisper import WhisperModel as FWWhisperModel
    _HAS_FASTER = True
except Exception:
    _HAS_FASTER = False


# ============================================================
# 1) Ground Truth sentence bank (1..18)
# ============================================================
SENTENCES_1_TO_18 = {
    1: "امروز هوا خیلی خوبه و من می‌خوام پیاده‌روی کنم.",
    2: "من دارم یک مدل گفتار به متن فارسی رو تست می‌کنم.",
    3: "لطفاً این جمله را دقیق و بدون اشتباه بنویس.",
    4: "راستش دیگه حوصله‌ی ترافیک و شلوغی رو ندارم.",
    5: "الان دقیقاً نمی‌دونم چی می‌خوام بگم ولی ادامه می‌دم!",
    6: "گفت بیا ساعت هشت، ولی خب طبق معمول دیر کرد.",
    7: "جلسه‌ی بعدی ساعت ۹ و ۴۵ دقیقه‌ی صبح برگزار می‌شه.",
    8: "من بین سال‌های ۱۳۹۸ تا ۱۴۰۲ روی پروژه‌های هوش مصنوعی کار کردم.",
    9: "قیمت این لپ‌تاپ حدود بیست و هفت میلیون تومنه.",
    10: "شیر، شیلنگ، شیشه و شیراز خیلی شبیه هم شروع می‌شن.",
    11: "ساقه‌ی درخت کنار سکو شکسته بود.",
    12: "سرِ سبد سیب سرخ سنگین شد.",
    13: "من می‌خواستم سریع توضیح بدم ولی چون هیجان‌زده بودم کلمات قاطی شد و جمله طولانی‌تر از حد انتظار شد.",
    14: "وقتی سیستم گفتار به متن همزمان با نویز محیط و صدای چند نفر کار می‌کنه، دقتش واقعاً به چالش کشیده می‌شه.",
    15: "پروژه‌ی Whisper روی پردازش گفتار چندزبانه تمرکز داره.",
    16: "من با پایتون، PyTorch و ترنسفورمرها کار می‌کنم.",
    17: "این فایل رو توی گیت‌هاب کامیت کردم ولی پوش ندادم.",
    18: "دیروز ساعت پنج عصر، وقتی داشتم روی مدل جدید کار می‌کردم، اینترنت قطع شد و کل تمرکزم پرید.",
}

WHISPER_HF_MODELS = [
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small",
    "openai/whisper-medium",
    "openai/whisper-large-v3",
]

FASTER_WHISPER_MAP = {
    "openai/whisper-tiny": "Systran/faster-whisper-tiny",
    "openai/whisper-base": "Systran/faster-whisper-base",
    "openai/whisper-small": "Systran/faster-whisper-small",
    "openai/whisper-medium": "Systran/faster-whisper-medium",
    "openai/whisper-large-v3": "Systran/faster-whisper-large-v3",
}


# ============================================================
# 2) Helpers
# ============================================================
def id_to_idx(audio_id: str) -> Optional[int]:
    m = re.search(r"(\d+)$", str(audio_id))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


_PERSIAN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_ARABIC_VARIANTS = {
    "ي": "ی",
    "ك": "ک",
    "ۀ": "ه",
    "ة": "ه",
    "ؤ": "و",
    "أ": "ا",
    "إ": "ا",
    "ٱ": "ا",
    "‌": " ",  # ZWNJ -> space
}

def normalize_fa(text: str, keep_punct: bool = False) -> str:
    if text is None:
        return ""
    t = str(text).strip()
    if not t:
        return ""
    for a, b in _ARABIC_VARIANTS.items():
        t = t.replace(a, b)
    t = t.translate(_PERSIAN_DIGITS)
    t = re.sub(r"\s+", " ", t).strip()
    if not keep_punct:
        t = re.sub(r"[^\w\s\u0600-\u06FF]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
    return t


def load_audio_16k_mono(path: Path) -> Tuple[np.ndarray, float]:
    y, _sr = librosa.load(str(path), sr=16000, mono=True)
    y = y.astype(np.float32)
    dur = float(len(y) / 16000.0)
    return y, dur


def sync_device(run_device: str):
    if run_device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def safe_empty_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_device(device_arg: str) -> Tuple[str, str, str]:
    """
    Returns: (run_device_for_torch, fw_device_for_faster_whisper, device_tag_for_reports)
    """
    if device_arg == "gpu" and torch.cuda.is_available():
        return "cuda", "cuda", "gpu"
    return "cpu", "cpu", "cpu"


def torch_env_report(run_device: str, fw_device: str, device_arg: str) -> Dict[str, str]:
    info = {
        "torch": str(torch.__version__),
        "torch_cuda_build": str(torch.version.cuda),
        "cuda_available_system": str(torch.cuda.is_available()),
        "requested_device": device_arg,
        "run_device_used": run_device,
        "fw_device_used": fw_device,
    }
    if run_device == "cuda" and torch.cuda.is_available():
        info["gpu_name"] = str(torch.cuda.get_device_name(0))
        info["capability"] = str(torch.cuda.get_device_capability(0))
    return info


# ============================================================
# 3) Transformers Whisper (auto fallback precision) + guards
# ============================================================
_TF_PIPE_CACHE: Dict[Tuple[str, str, str], object] = {}

def _dtype_candidates(run_device: str) -> List[Tuple[str, torch.dtype]]:
    if run_device == "cuda" and torch.cuda.is_available():
        return [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]
    return [("fp32", torch.float32)]


def get_transformers_pipe(model_id: str, run_device: str, prefer: Optional[str] = None):
    use_gpu = (run_device == "cuda" and torch.cuda.is_available())
    pipe_device = 0 if use_gpu else -1

    candidates = _dtype_candidates(run_device)
    if prefer:
        candidates = sorted(candidates, key=lambda x: 0 if x[0] == prefer else 1)

    last_err = None
    for precision_tag, dtype in candidates:
        key = (model_id, run_device, precision_tag)
        if key in _TF_PIPE_CACHE:
            return _TF_PIPE_CACHE[key], precision_tag, 0.0  # load_time unknown once cached

        try:
            t0 = time.perf_counter()
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=dtype)

            if use_gpu:
                model = model.to("cuda")

            asr = hf_pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=pipe_device,
            )
            load_time = time.perf_counter() - t0

            _TF_PIPE_CACHE[key] = asr
            return asr, precision_tag, load_time

        except Exception as e:
            last_err = e
            safe_empty_cache()
            continue

    raise RuntimeError(f"Failed to init transformers pipeline for {model_id}. Last error: {last_err}")


def run_transformers_whisper(model_id: str, audio: np.ndarray, run_device: str, beam_size: int) -> Tuple[str, str]:
    """
    Returns: (text, precision_tag_used)

    Adds decoding guards to reduce repetitions ("این این این...")
    """
    asr, prec, _load = get_transformers_pipe(model_id, run_device)

    out = asr(
        audio,
        generate_kwargs={
            "task": "transcribe",
            "language": "fa",
            "num_beams": int(beam_size),
            "do_sample": False,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3,
            "max_new_tokens": 256,
        },
    )
    text = str(out.get("text", "")).strip() if isinstance(out, dict) else str(out).strip()
    return text, prec


# ============================================================
# 4) faster-whisper (cache + skip tiny int8_float16 on GPU)
# ============================================================
_FW_CACHE: Dict[Tuple[str, str, str], FWWhisperModel] = {}

def get_fw_model(ct2_id: str, fw_device: str, compute_type: str) -> Tuple[FWWhisperModel, float]:
    """
    Returns: (model, load_time) ; cached models => load_time=0
    """
    key = (ct2_id, fw_device, compute_type)
    if key in _FW_CACHE:
        return _FW_CACHE[key], 0.0

    t0 = time.perf_counter()
    fw = FWWhisperModel(ct2_id, device=fw_device, compute_type=compute_type)
    load_time = time.perf_counter() - t0
    _FW_CACHE[key] = fw
    return fw, load_time


def run_faster_whisper(ct2_id: str, audio: np.ndarray, fw_device: str, compute_type: str, beam_size: int) -> str:
    if not _HAS_FASTER:
        raise RuntimeError("faster-whisper not installed. pip install faster-whisper ctranslate2")
    fw, _ = get_fw_model(ct2_id, fw_device, compute_type)
    segments, info = fw.transcribe(audio, language="fa", beam_size=int(beam_size), vad_filter=False)
    return " ".join([s.text for s in segments]).strip()


def fw_variants_for_device(fw_device: str) -> List[str]:
    if fw_device == "cuda":
        return ["float16", "int8_float16"]
    return ["int8"]


def should_skip_fw_variant(model_id: str, fw_device: str, compute_type: str) -> bool:
    # known flaky on some setups: tiny + int8_float16 on GPU
    if fw_device == "cuda" and compute_type == "int8_float16" and model_id.endswith("whisper-tiny"):
        return True
    return False


# ============================================================
# 5) Reporting utils (summary, pareto, charts)
# ============================================================
def summarize(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    g = df.groupby(group_cols, dropna=False)
    out = g.agg(
        n=("audio_id", "count"),
        success_rate=("success", "mean"),
        mean_load_sec=("load_sec", "mean"),
        mean_infer_sec=("infer_sec", "mean"),
        p50_infer_sec=("infer_sec", "median"),
        mean_latency_sec=("latency_sec", "mean"),     # load + infer (for completeness)
        mean_rtf=("rtf", "mean"),
        p50_rtf=("rtf", "median"),
        mean_WER=("WER", "mean"),
        mean_CER=("CER", "mean"),
    ).reset_index()

    # Fair ranking: accuracy first, then inference speed (not load)
    out["rank_score_fair"] = out["mean_CER"].fillna(10.0) * 1000 + out["mean_infer_sec"].fillna(1e9)

    # Realtime ranking: prioritize rtf then CER
    out["rank_score_realtime"] = out["mean_rtf"].fillna(1e9) * 1000 + out["mean_CER"].fillna(10.0)

    return out


def pareto_frontier(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    pts = df[[x, y]].to_numpy()
    keep = []
    for i, (xi, yi) in enumerate(pts):
        dominated = False
        for j, (xj, yj) in enumerate(pts):
            if j == i:
                continue
            if (xj <= xi and yj <= yi) and (xj < xi or yj < yi):
                dominated = True
                break
        keep.append(not dominated)
    return df.loc[keep].copy()


def build_analysis(summary_all: pd.DataFrame, frontier_fair: pd.DataFrame, frontier_rt: pd.DataFrame, env: Dict[str, str]) -> str:
    lines = []
    lines.append("ENV")
    lines.append("-" * 40)
    for k, v in env.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("Notes:")
    lines.append("- mean_infer_sec excludes warmup and excludes model load_time.")
    lines.append("- mean_latency_sec includes (amortized) load_sec + infer_sec (mostly for completeness).")
    lines.append("- For Persian, CER is usually more stable than WER.")
    lines.append("")
    lines.append("TOP-5 (FAIR): sort by mean_CER then mean_infer_sec")
    cols = ["device","backend","model_id","precision","mean_CER","mean_WER","mean_infer_sec","mean_rtf","success_rate","n"]
    cols = [c for c in cols if c in summary_all.columns]
    fair = summary_all.sort_values(["rank_score_fair","mean_CER","mean_infer_sec"], ascending=True).head(5)
    lines.append(fair[cols].to_string(index=False))
    lines.append("")
    lines.append("TOP-5 (REALTIME): sort by mean_rtf then mean_CER")
    rt = summary_all.sort_values(["rank_score_realtime","mean_rtf","mean_CER"], ascending=True).head(5)
    lines.append(rt[cols].to_string(index=False))
    lines.append("")
    lines.append("Pareto (FAIR: CER vs infer_sec)")
    lines.append(frontier_fair.sort_values(["mean_CER","mean_infer_sec"]).head(10)[cols].to_string(index=False))
    lines.append("")
    lines.append("Pareto (REALTIME: CER vs RTF)")
    lines.append(frontier_rt.sort_values(["mean_CER","mean_rtf"]).head(10)[cols].to_string(index=False))
    return "\n".join(lines)


def svg_scatter(df: pd.DataFrame, x: str, y: str, label_cols: List[str], title: str, width: int = 640, height: int = 360) -> str:
    """
    Minimal SVG scatter plot. No external libs.
    """
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        return "<p><em>No chart data</em></p>"

    # Keep finite rows
    d = df[[x, y] + [c for c in label_cols if c in df.columns]].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[x, y])
    if d.empty:
        return "<p><em>No finite points</em></p>"

    # ranges with padding
    xvals = d[x].to_numpy()
    yvals = d[y].to_numpy()
    xmin, xmax = float(np.min(xvals)), float(np.max(xvals))
    ymin, ymax = float(np.min(yvals)), float(np.max(yvals))
    if xmax == xmin:
        xmax = xmin + 1e-9
    if ymax == ymin:
        ymax = ymin + 1e-9

    pad = 0.08
    xr = xmax - xmin
    yr = ymax - ymin
    xmin -= xr * pad
    xmax += xr * pad
    ymin -= yr * pad
    ymax += yr * pad

    # plot area
    m = 40
    pw, ph = width - 2*m, height - 2*m

    def sx(v): return m + (float(v) - xmin) / (xmax - xmin) * pw
    def sy(v): return m + (1.0 - (float(v) - ymin) / (ymax - ymin)) * ph

    # axes ticks
    def ticks(lo, hi, n=5):
        return [lo + (hi - lo) * i / (n - 1) for i in range(n)]

    xt = ticks(xmin, xmax)
    yt = ticks(ymin, ymax)

    # points
    circles = []
    labels = []
    for _, r in d.iterrows():
        cx, cy = sx(r[x]), sy(r[y])
        circles.append(f"<circle cx='{cx:.2f}' cy='{cy:.2f}' r='4' opacity='0.75'></circle>")
        lab = " | ".join([str(r[c]) for c in label_cols if c in d.columns])
        # short label
        lab = (lab[:48] + "…") if len(lab) > 49 else lab
        labels.append(f"<text x='{cx+6:.2f}' y='{cy-6:.2f}' font-size='10'>{escape_html(lab)}</text>")

    # grid + ticks
    grid = []
    for v in xt:
        xpx = sx(v)
        grid.append(f"<line x1='{xpx:.2f}' y1='{m}' x2='{xpx:.2f}' y2='{height-m}' stroke='#ddd' stroke-width='1'/>")
        grid.append(f"<text x='{xpx:.2f}' y='{height-10}' font-size='10' text-anchor='middle'>{v:.3g}</text>")
    for v in yt:
        ypx = sy(v)
        grid.append(f"<line x1='{m}' y1='{ypx:.2f}' x2='{width-m}' y2='{ypx:.2f}' stroke='#ddd' stroke-width='1'/>")
        grid.append(f"<text x='10' y='{ypx+4:.2f}' font-size='10'>{v:.3g}</text>")

    return f"""
    <div style="margin:12px 0">
      <div style="font-weight:600;margin:6px 0">{escape_html(title)}</div>
      <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="border:1px solid #e5e7eb;border-radius:12px;background:#fff">
        {''.join(grid)}
        <rect x="{m}" y="{m}" width="{pw}" height="{ph}" fill="none" stroke="#999" stroke-width="1"/>
        {''.join(circles)}
        {''.join(labels)}
      </svg>
      <div style="font-size:12px;color:#444;margin-top:6px">
        x: <code>{escape_html(x)}</code> , y: <code>{escape_html(y)}</code>
      </div>
    </div>
    """


def escape_html(s: str) -> str:
    return (s.replace("&", "&amp;")
              .replace("<", "&lt;")
              .replace(">", "&gt;")
              .replace('"', "&quot;")
              .replace("'", "&#39;"))


def df_to_html(df: pd.DataFrame, max_rows: int = 300) -> str:
    if df is None or df.empty:
        return "<p><em>No data</em></p>"
    return df.head(max_rows).to_html(index=False, escape=True)


def write_html(out_dir: Path,
               summary_all: pd.DataFrame,
               frontier_fair: pd.DataFrame,
               frontier_rt: pd.DataFrame,
               hard: pd.DataFrame,
               analysis_text: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "asr_report.html"

    head = """
    <html><head>
      <meta charset="utf-8"/>
      <title>Persian ASR Benchmark Report</title>
      <style>
        body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; }
        h1,h2,h3 { margin-top: 28px; }
        table { border-collapse: collapse; width: 100%; margin: 12px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; font-size: 13px; vertical-align: top; }
        th { background: #f7f7f7; position: sticky; top: 0; }
        .analysis { white-space: pre-wrap; background: #f8fafc; border: 1px solid #e5e7eb; padding: 12px; border-radius: 10px; }
        .note { color: #444; }
        code { background: #f2f2f2; padding: 1px 4px; border-radius: 6px; }
        .grid { display: grid; grid-template-columns: 1fr; gap: 16px; }
        @media (min-width: 1100px) { .grid { grid-template-columns: 1fr 1fr; } }
      </style>
    </head><body>
    """

    # charts
    label_cols = ["backend","model_id","precision"]
    chart_fair = svg_scatter(frontier_fair, x="mean_CER", y="mean_infer_sec", label_cols=label_cols,
                             title="Pareto (FAIR): mean_CER vs mean_infer_sec (lower is better)")
    chart_rt = svg_scatter(frontier_rt, x="mean_CER", y="mean_rtf", label_cols=label_cols,
                           title="Pareto (REALTIME): mean_CER vs mean_rtf (lower is better)")

    body = []
    body.append("<h1>Persian ASR Benchmark Report</h1>")
    body.append("<p class='note'>Scoring: lower CER/WER is better. Speed: lower infer/RTF is better. Warmup excluded.</p>")

    body.append("<h2>Analysis</h2>")
    body.append("<div class='analysis'>" + escape_html(analysis_text) + "</div>")

    body.append("<h2>Pareto charts</h2>")
    body.append("<div class='grid'>" + chart_fair + chart_rt + "</div>")

    body.append("<h2>Overall summary (all variants)</h2>")
    body.append(df_to_html(summary_all, 500))

    body.append("<h2>Pareto frontier tables</h2>")
    body.append("<h3>FAIR (CER vs infer_sec)</h3>" + df_to_html(frontier_fair, 300))
    body.append("<h3>REALTIME (CER vs rtf)</h3>" + df_to_html(frontier_rt, 300))

    body.append("<h2>Hardest samples (top 25 CER)</h2>")
    body.append(df_to_html(hard, 25))

    html_path.write_text(head + "\n".join(body) + "</body></html>", encoding="utf-8")
    return html_path


def realtime_recommendation(summary_all: pd.DataFrame) -> str:
    """
    Simple recommender for realtime:
    - filter success_rate == 1
    - pick best by mean_rtf then mean_CER
    """
    if summary_all is None or summary_all.empty:
        return "No summary data."

    s = summary_all.copy()
    s = s[s["success_rate"] >= 0.999].copy()
    if s.empty:
        return "No fully-successful variant found (success_rate < 1)."

    s = s.sort_values(["rank_score_realtime","mean_rtf","mean_CER"], ascending=True)
    top = s.iloc[0].to_dict()

    lines = []
    lines.append("Realtime recommendation (heuristic):")
    lines.append("- prioritize mean_rtf (lower better), then mean_CER")
    lines.append("")
    keys = ["device","backend","model_id","precision","mean_rtf","mean_infer_sec","mean_CER","mean_WER","n"]
    for k in keys:
        if k in top:
            lines.append(f"{k}: {top[k]}")
    lines.append("")
    lines.append("Rule of thumb:")
    lines.append("- If you need RTF < 0.2: consider faster-whisper on GPU (float16 or int8_float16).")
    lines.append("- If CPU-only: faster-whisper int8 is usually your best speed baseline.")
    return "\n".join(lines)


# ============================================================
# 6) Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_dir", required=True, help="Path to test_samples/")
    ap.add_argument("--out_dir", default="./report", help="Output dir")
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--skip_transformers", action="store_true")
    ap.add_argument("--skip_faster_whisper", action="store_true")
    ap.add_argument("--only_models", default=None, help="Comma-separated list of HF whisper model ids")
    ap.add_argument("--warmup_n", type=int, default=1, help="Warmup runs per model/variant (excluded from latency)")
    args = ap.parse_args()

    samples_dir = Path(args.samples_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_device, fw_device, device_tag = resolve_device(args.device)
    env = torch_env_report(run_device, fw_device, args.device)

    # gather audio
    audio_paths = sorted(list(samples_dir.rglob("*.mp3")) + list(samples_dir.rglob("*.wav")))
    if not audio_paths:
        raise SystemExit(f"No audio files found under: {samples_dir}")

    # preload audio
    audio_cache: Dict[str, np.ndarray] = {}
    rows_audio = []
    for p in audio_paths:
        y, dur = load_audio_16k_mono(p)
        audio_cache[str(p)] = y
        speaker = p.parent.name
        audio_id = p.stem
        idx = id_to_idx(audio_id)
        ref = SENTENCES_1_TO_18.get(idx, "")
        rows_audio.append({
            "speaker": speaker,
            "audio_id": audio_id,
            "audio_path": str(p),
            "duration_sec": dur,
            "ref_text": ref,
        })
    df_audio = pd.DataFrame(rows_audio)

    # models
    models = WHISPER_HF_MODELS[:]
    if args.only_models:
        want = [s.strip() for s in args.only_models.split(",") if s.strip()]
        models = [m for m in models if m in want]
        if not models:
            raise SystemExit("No models left after --only_models.")

    results: List[Dict] = []

    # -----------------------
    # A) Transformers Whisper
    # -----------------------
    if not args.skip_transformers:
        print(f"\n### Running backend: transformers_whisper on {device_tag} (run_device={run_device})")
        for model_id in models:
            print(f"\nModel: {model_id}")

            # init + measure load_time (first build)
            load_sec = 0.0
            try:
                _pipe, _prec, load_sec = get_transformers_pipe(model_id, run_device)
                print(f"Init OK | load_sec={load_sec:.3f}")
            except Exception as e:
                print(f"Init failed for {model_id}: {e}")
                # log failures for all samples
                for _, r in df_audio.iterrows():
                    results.append({
                        "device": device_tag,
                        "backend": "transformers_whisper",
                        "precision": "unknown",
                        "model_id": model_id,
                        "speaker": r["speaker"],
                        "audio_id": r["audio_id"],
                        "duration_sec": r["duration_sec"],
                        "load_sec": np.nan,
                        "infer_sec": np.nan,
                        "latency_sec": np.nan,
                        "rtf": np.nan,
                        "pred_text": "",
                        "ref_text": r["ref_text"],
                        "error": f"init_failed: {e}",
                    })
                safe_empty_cache()
                continue

            # warmup (excluded)
            try:
                y0 = audio_cache[df_audio.iloc[0]["audio_path"]]
                for _ in range(max(0, int(args.warmup_n))):
                    sync_device(run_device)
                    _ = run_transformers_whisper(model_id, y0, run_device, args.beam_size)
                    sync_device(run_device)
                print(f"Warmup OK (n={args.warmup_n})")
            except Exception as e:
                print(f"Warmup failed for {model_id}: {e}")

            # inference per sample
            for _, r in df_audio.iterrows():
                y = audio_cache[r["audio_path"]]
                try:
                    sync_device(run_device)
                    t0 = time.perf_counter()
                    text, prec = run_transformers_whisper(model_id, y, run_device, args.beam_size)
                    print(f"text: {text}")
                    sync_device(run_device)
                    infer_sec = time.perf_counter() - t0

                    # amortize load across N samples of this model (fair-ish)
                    load_amort = load_sec / max(1, len(df_audio))

                    latency_sec = load_amort + infer_sec
                    rtf = infer_sec / r["duration_sec"] if r["duration_sec"] > 0 else np.nan

                    results.append({
                        "device": device_tag,
                        "backend": "transformers_whisper",
                        "precision": prec,
                        "model_id": model_id,
                        "speaker": r["speaker"],
                        "audio_id": r["audio_id"],
                        "duration_sec": r["duration_sec"],
                        "load_sec": load_amort,
                        "infer_sec": infer_sec,
                        "latency_sec": latency_sec,
                        "rtf": rtf,
                        "pred_text": text,
                        "ref_text": r["ref_text"],
                        "error": "",
                    })
                    print(f"{model_id} | {r['audio_id']} | {prec} | infer={infer_sec:.3f}s | rtf={rtf:.3f}")
                except Exception as e:
                    results.append({
                        "device": device_tag,
                        "backend": "transformers_whisper",
                        "precision": "unknown",
                        "model_id": model_id,
                        "speaker": r["speaker"],
                        "audio_id": r["audio_id"],
                        "duration_sec": r["duration_sec"],
                        "load_sec": np.nan,
                        "infer_sec": np.nan,
                        "latency_sec": np.nan,
                        "rtf": np.nan,
                        "pred_text": "",
                        "ref_text": r["ref_text"],
                        "error": str(e),
                    })
                    print(f"ERROR {model_id} | {r['audio_id']}: {e}")

            safe_empty_cache()

    # -----------------------
    # B) faster-whisper
    # -----------------------
    if not args.skip_faster_whisper:
        if not _HAS_FASTER:
            print("\n[SKIP] faster-whisper not installed.")
        else:
            compute_types = fw_variants_for_device(fw_device)
            print(f"\n### Running backend: faster_whisper on {device_tag} (fw_device={fw_device}) types={compute_types}")

            for model_id in models:
                ct2_id = FASTER_WHISPER_MAP.get(model_id)
                if not ct2_id:
                    print(f"[SKIP] No CT2 mapping for {model_id}")
                    continue

                for compute_type in compute_types:
                    if should_skip_fw_variant(model_id, fw_device, compute_type):
                        print(f"[SKIP] {model_id} + {compute_type} on GPU (known flaky)")
                        continue

                    print(f"\nCT2 model: {ct2_id} | compute_type={compute_type}")

                    # init + load time
                    load_sec = 0.0
                    try:
                        _, load_sec = get_fw_model(ct2_id, fw_device, compute_type)
                        print(f"Init OK | load_sec={load_sec:.3f}")
                    except Exception as e:
                        print(f"Init failed for {ct2_id} ({compute_type}): {e}")
                        for _, r in df_audio.iterrows():
                            results.append({
                                "device": device_tag,
                                "backend": "faster_whisper",
                                "precision": compute_type,
                                "model_id": model_id,
                                "speaker": r["speaker"],
                                "audio_id": r["audio_id"],
                                "duration_sec": r["duration_sec"],
                                "load_sec": np.nan,
                                "infer_sec": np.nan,
                                "latency_sec": np.nan,
                                "rtf": np.nan,
                                "pred_text": "",
                                "ref_text": r["ref_text"],
                                "error": f"init_failed: {e}",
                            })
                            
                        safe_empty_cache()
                        continue

                    # warmup
                    try:
                        y0 = audio_cache[df_audio.iloc[0]["audio_path"]]
                        for _ in range(max(0, int(args.warmup_n))):
                            sync_device(run_device)
                            _ = run_faster_whisper(ct2_id, y0, fw_device, compute_type, args.beam_size)
                            sync_device(run_device)
                        print(f"Warmup OK (n={args.warmup_n})")
                    except Exception as e:
                        print(f"Warmup failed for {ct2_id} ({compute_type}): {e}")

                    # inference per sample
                    for _, r in df_audio.iterrows():
                        y = audio_cache[r["audio_path"]]
                        try:
                            sync_device(run_device)
                            t0 = time.perf_counter()
                            text = run_faster_whisper(ct2_id, y, fw_device, compute_type, args.beam_size)
                            sync_device(run_device)
                            infer_sec = time.perf_counter() - t0

                            load_amort = load_sec / max(1, len(df_audio))
                            latency_sec = load_amort + infer_sec
                            rtf = infer_sec / r["duration_sec"] if r["duration_sec"] > 0 else np.nan

                            results.append({
                                "device": device_tag,
                                "backend": "faster_whisper",
                                "precision": compute_type,
                                "model_id": model_id,
                                "speaker": r["speaker"],
                                "audio_id": r["audio_id"],
                                "duration_sec": r["duration_sec"],
                                "load_sec": load_amort,
                                "infer_sec": infer_sec,
                                "latency_sec": latency_sec,
                                "rtf": rtf,
                                "pred_text": text,
                                "ref_text": r["ref_text"],
                                "error": "",
                            })
                            print(f"{model_id} | {r['audio_id']} | {compute_type} | infer={infer_sec:.3f}s | rtf={rtf:.3f}")
                        except Exception as e:
                            results.append({
                                "device": device_tag,
                                "backend": "faster_whisper",
                                "precision": compute_type,
                                "model_id": model_id,
                                "speaker": r["speaker"],
                                "audio_id": r["audio_id"],
                                "duration_sec": r["duration_sec"],
                                "load_sec": np.nan,
                                "infer_sec": np.nan,
                                "latency_sec": np.nan,
                                "rtf": np.nan,
                                "pred_text": "",
                                "ref_text": r["ref_text"],
                                "error": str(e),
                            })
                            print(f"ERROR {model_id} | {r['audio_id']} ({compute_type}): {e}")

                    safe_empty_cache()

    # ========================================================
    # Final metrics & report
    # ========================================================
    df = pd.DataFrame(results)
    if df.empty:
        raise SystemExit("No results were generated. (All backends skipped or failed)")

    df["success"] = df["error"].fillna("").astype(str).str.len().eq(0)

    if args.normalize:
        df["pred_norm"] = df["pred_text"].map(normalize_fa)
        df["ref_norm"] = df["ref_text"].map(normalize_fa)
    else:
        df["pred_norm"] = df["pred_text"].fillna("").astype(str)
        df["ref_norm"] = df["ref_text"].fillna("").astype(str)

    # WER/CER
    df["WER"] = [float(wer(r, h)) if r is not None else np.nan for r, h in zip(df["ref_norm"], df["pred_norm"])]
    df["CER"] = [float(cer(r, h)) if r is not None else np.nan for r, h in zip(df["ref_norm"], df["pred_norm"])]

    merged_csv = out_dir / "asr_merged_results.csv"
    df.to_csv(merged_csv, index=False)

    summary_all = summarize(df, group_cols=["device","backend","model_id","precision"])
    # sort both views for convenience in file
    summary_all_fair = summary_all.sort_values(["rank_score_fair","mean_CER","mean_infer_sec"], ascending=True)
    summary_all_rt = summary_all.sort_values(["rank_score_realtime","mean_rtf","mean_CER"], ascending=True)

    summary_all.to_csv(out_dir / "asr_summary_all.csv", index=False)

    frontier_fair = pareto_frontier(summary_all, x="mean_CER", y="mean_infer_sec")
    frontier_fair.to_csv(out_dir / "asr_pareto_fair.csv", index=False)

    frontier_rt = pareto_frontier(summary_all, x="mean_CER", y="mean_rtf")
    frontier_rt.to_csv(out_dir / "asr_pareto_realtime.csv", index=False)

    hard = df[df["success"]].sort_values(["CER", "WER"], ascending=[False, False]).head(25)
    hard_cols = [c for c in ["device","backend","model_id","precision","speaker","audio_id","infer_sec","rtf","WER","CER","ref_text","pred_text"] if c in hard.columns]
    hard = hard[hard_cols]

    analysis_text = build_analysis(summary_all, frontier_fair, frontier_rt, env)
    html_path = write_html(out_dir, summary_all, frontier_fair, frontier_rt, hard, analysis_text)

    # realtime recommendation text
    rt_txt = realtime_recommendation(summary_all)
    (out_dir / "asr_realtime_recommendation.txt").write_text(rt_txt, encoding="utf-8")

    print("\n=== TOP 10 (FAIR) ===")
    print(summary_all_fair.head(10).to_string(index=False))
    print("\n=== TOP 10 (REALTIME) ===")
    print(summary_all_rt.head(10).to_string(index=False))

    print(f"\nSaved: {merged_csv}")
    print(f"Saved: {out_dir / 'asr_summary_all.csv'}")
    print(f"Saved: {out_dir / 'asr_pareto_fair.csv'}")
    print(f"Saved: {out_dir / 'asr_pareto_realtime.csv'}")
    print(f"Saved: {out_dir / 'asr_realtime_recommendation.txt'}")
    print(f"Saved: {html_path}")


if __name__ == "__main__":
    main()
