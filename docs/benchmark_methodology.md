# Persian ASR Benchmark: Experimental Methodology

## 1. Objective

This benchmark aims to provide a systematic and reproducible evaluation of
Persian Automatic Speech Recognition (ASR) systems based on OpenAI Whisper.

The study addresses two complementary objectives:

1. **Fair accuracy benchmarking**
   Evaluate transcription quality independently of deployment constraints.

2. **Real-time model selection**
   Identify models suitable for low-latency inference on CPU and GPU.

Rather than assuming a single “best” model, the benchmark explicitly analyzes
accuracy–latency trade-offs.

---

## 2. Dataset

### 2.1 Composition

The evaluation dataset consists of a small but controlled set of Persian speech
samples:

- Total samples: 36
- Speakers: 2
- Language: Persian
- Audio format: WAV, 16kHz, mono

Each audio file is paired with a manually verified reference transcription.

### 2.2 Rationale

The dataset is intentionally limited in size to:

- enable rapid iteration
- avoid stochastic variance from large-scale batching
- allow per-sample latency inspection

This setup is particularly suitable for real-time analysis.

---

## 3. Models

All experiments use official Whisper checkpoints distributed via HuggingFace:

- whisper-tiny
- whisper-base
- whisper-small
- whisper-medium
- whisper-large-v3

Model sizes span a wide range of parameter counts, enabling evaluation across
accuracy and efficiency regimes.

---

## 4. Inference Backends

### 4.1 Transformers Whisper

The HuggingFace Transformers implementation is used as a reference backend.
It prioritizes transcription quality and algorithmic clarity.

To mitigate known decoding pathologies (e.g., repeated tokens in small models),
custom generation parameters are applied:

- Beam search decoding
- Repetition penalty
- N-gram repetition blocking
- Sampling disabled

These constraints are essential for stable WER/CER computation.

---

### 4.2 faster-whisper (CTranslate2)

The faster-whisper backend provides an optimized inference pipeline designed for
low-latency and real-time deployment.

Compute configurations depend on hardware:

| Device | Compute Type |
|------|-------------|
| GPU  | float16 |
| GPU  | int8_float16 |
| CPU  | int8 |

A known instability affecting `whisper-tiny + int8_float16` on GPU was observed
and excluded from final recommendations due to zero success rate.

---

## 5. Experimental Protocol

### 5.1 Warmup and Timing

- A warmup pass is executed for each model
- Warmup latency is excluded from reported results
- Load time and inference time are measured separately

### 5.2 Inference Conditions

- Batch size: 1 (no batching)
- Offline inference
- Deterministic decoding

This setup closely approximates real-time usage patterns.

---

## 6. Evaluation Metrics

For each audio sample, the following metrics are computed:

- Word Error Rate (WER)
- Character Error Rate (CER)
- Inference latency (seconds)
- Real-Time Factor (RTF)

For Persian ASR, CER is treated as the primary accuracy metric due to
morphological and tokenization variability affecting WER.

---

## 7. Aggregation and Ranking

Results are aggregated across samples to compute:

- Mean and median latency
- Mean CER and WER
- Success rate
- Composite ranking scores

Two ranking regimes are defined:

1. **Fair ranking**
   Prioritizes CER, then inference latency.

2. **Real-time ranking**
   Prioritizes RTF, then CER.

Pareto frontiers are computed for both regimes.

---

## 8. Hardware and Software Environment

All experiments were conducted on a fixed hardware configuration.
A complete snapshot including CPU, GPU, OS, driver, CUDA, and package versions
is stored alongside the results to ensure reproducibility.

---

## 9. Interpretation Guidelines

- No single model is optimal for all scenarios.
- Large models dominate accuracy-oriented tasks.
- Smaller models offer superior real-time performance.
- Backend choice significantly impacts latency.

Model selection should therefore be driven by deployment constraints rather
than raw accuracy alone.

---

## 10. Reproducibility Statement

All scripts, parameters, and raw results are provided.
Any result reported in this benchmark can be independently reproduced using
the supplied CSV files and hardware specifications.
