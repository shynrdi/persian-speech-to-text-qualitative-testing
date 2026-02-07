# Persian ASR Benchmark (Whisper-based)

A reproducible benchmark for Persian Automatic Speech Recognition (ASR),
focused on **accuracy vs latency trade-offs** on both **CPU and GPU**.

This repository compares multiple Whisper model sizes using different
inference backends, with special attention to **real-time feasibility**.

---

## Key Goals

- Fair and reproducible ASR benchmarking for Persian
- Separate evaluation for:
  - **Accuracy-oriented use cases**
  - **Real-time deployment scenarios**
- Transparent comparison across:
  - CPU vs GPU
  - Transformers vs faster-whisper backends

---

## Models Evaluated

- `openai/whisper-tiny`
- `openai/whisper-base`
- `openai/whisper-small`
- `openai/whisper-medium`
- `openai/whisper-large-v3`

---

## Inference Backends

- **transformers (HuggingFace Whisper)**
  - High accuracy
  - Reference implementation
- **faster-whisper (CTranslate2)**
  - Optimized for low latency
  - Suitable for production and real-time

---

## Dataset (Test Samples)

- Language: Persian (Farsi)
- Speakers: 2
- Total samples: 36
- Sampling rate: 16kHz mono
- Each audio file has a fixed, verified ground-truth transcript

Directory structure:

test_samples/
├── Ardalan/
└── Shayan/

---

## Metrics

For each audio sample:

- Word Error Rate (WER)
- Character Error Rate (CER)
- Inference latency (seconds)
- Real-Time Factor (RTF)

> Note: For Persian, CER is considered more stable than WER
> and is prioritized in final rankings.

---

## Results

All results are stored as CSV files and can be re-aggregated or re-plotted:

- `asr_merged_results.csv`  
  Per-sample inference results

- `asr_summary_all.csv`  
  Aggregated metrics per model / backend / precision

- `asr_pareto.csv`  
  Pareto-optimal configurations (accuracy vs speed)

An HTML report with tables and Pareto plots is also generated automatically.

---

## Hardware & Environment

All benchmarks were executed on controlled hardware.
A full hardware and software snapshot is available under:

hw_info/


This includes CPU, GPU, OS, driver versions, and package versions.

---

## Academic-Style Methodology

For a detailed description of:
- experimental protocol
- decoding settings
- fairness constraints
- hardware assumptions
- statistical interpretation

see:

docs/benchmark_methodology.md


---

## Reproducibility

- No batching (batch size = 1)
- Warmup excluded from latency measurements
- Model load time measured separately
- Deterministic decoding (no sampling)

---

## License

MIT