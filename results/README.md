# Saved Result Artifacts

Reproducible JSON artifacts for the headline numbers cited in the project
documentation. Each subdirectory corresponds to one experiment run; the
directory name is `<exp-name>_<UTC-timestamp>`.

## `legacy_phase1_beir/`

Phase-1 KV-Embedding pipeline (HuggingFace-based, code in `src/legacy/`,
runner: `experiments/legacy/4_benchmark_beir.py` and
`experiments/legacy/7_benchmark_beir_storage.py`). All runs from
2026-01-21 / 2026-01-23.

| Run | Dataset | Model | Config | Headline |
|---|---|---|---|---|
| `exp4_beir_20260121_230957` | scifact | Qwen3-VL-8B-Instruct (4bit) | no whitening, no zscore | NDCG@10 = 0.0 (degenerate run, retained as failure record) |
| `exp4_beir_20260121_231316` | scifact | Qwen3-VL-8B-Instruct (4bit) | no whitening, no zscore | **NDCG@10 = 0.397** -- the "0.40 baseline" |
| `exp4_beir_20260121_232031` | scifact | Qwen3-VL-8B-Instruct (4bit) | **whitening + zscore** | **NDCG@10 = 0.821** -- the headline number |
| `exp7_beir_storage_20260123_213956` | scifact | (storage variant) | INT8 quantization | NDCG@10 = 0.556 (vs 0.821 unquantized) -- 4x storage at ~32% NDCG drop |
| `exp7_beir_storage_20260123_215423` | nfcorpus | (storage variant) | INT8 quantization | NDCG@10 = 0.130 |

Headline claim in `README.md` and the cover letter ("NDCG@10 0.82 on BEIR
SciFact with an 8B model; whitening + z-score normalization moved it from
0.40 to 0.82") is supported by these artifacts.

## `phase3_e2e_gate/`

Phase-3 TransformerLens pipeline TriviaQA / needle-in-haystack end-to-end
gate (current pipeline, code in `experiments/nnm/`, runner family
`exp17*`). All runs from 2026-04-25 / 2026-04-26.

| Run | Test | Model | Headline |
|---|---|---|---|
| `nnm_exp17c_20260425_184841` | TriviaQA, n=100 x 3 seeds | Qwen2.5-7B-Instruct | nnma EM = **0.76** > rag 0.72 > cold 0.52 > random 0.50 (random matches cold floor -> control passes) |
| `nnm_exp17d_20260425_191856` | TriviaQA with RAG length ablation | Qwen2.5-7B-Instruct | nnma 0.76 vs rag_short_{50,100,200,400} |
| `nnm_exp17f_20260425_202048` | TriviaQA post-RoPE variants, n=50 | Qwen2.5-7B-Instruct | nnma_baseline 0.78 > rag 0.76 > nnma_in_chat 0.58 > nnma_post_rope 0.28 |
| `nnm_exp17g_20260426_130615` | Needle-in-Haystack, n=50, 2.5-8k chars | Qwen2.5-7B-Instruct | nnma needle=1.00 vs rag needle=0.86; McNemar p=0.0156 |
| `nnm_exp17g_20260426_134444` | Needle-in-Haystack variant | Qwen2.5-7B-Instruct | nnma both=0.98 vs rag both=0.82 |
| `nnm_exp17h_20260426_135808` | Password-extraction robustness | Qwen2.5-7B-Instruct | vulnerability_rate by attack pattern across 48 trials |

Each run folder typically contains `summary.json`, `per_item.json`, and
`log.txt`. Multi-seed runs also have `summary_seed_{42,43,44}.json`.

## Reproducibility notes

- Phase-1 artifacts were produced with the pre-restructure
  `kv_llm_vectorstore` codebase. The current `experiments/legacy/`
  scripts have been patched (commit `d0fd154`) so the same runs reproduce
  on a clean clone, but exact byte-equality of new runs vs. these archived
  files is not guaranteed (model weights, BEIR data versions etc.).
- Phase-3 artifacts were produced with the TransformerLens-based pipeline
  in `experiments/nnm/`. See `docs/e2e-gate-results-2026-04-18.md` for
  the discussion of these runs.
- The 0.82 / 0.40 / 0.76 / McNemar numbers cited in the cover letter and
  README all map to specific files above.
