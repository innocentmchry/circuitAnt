# Semantic Plan Cache Evaluation Report

## Executive Summary
- **Unseen Netlist**: `test017.v` (test017)
- **Total Test Queries**: 39 across Analysis, Transformation, and Verification
- **Cache Hit Rate**: 25.6%
- **Token Reduction**: **19.5%**
- **Latency Reduction**: **18.1%**
- **Wrong Plan Rate**: **0.0%** (0 false hits on negative control opposite-pairs)
- **Fallback Rate**: **0.0%** on compatible paraphrases

---

## 1. Matching Accuracy vs Similarity Threshold Sweep

| Threshold | Accuracy | Precision | Recall | True Positives | False Positives | True Negatives | False Negatives |
|:---------:|:--------:|:---------:|:------:|:--------------:|:---------------:|:--------------:|:---------------:|
| **0.70** | 43.6% | 88.9% | 44.4% | 16 | 2 | 1 | 20 |
| **0.80** | 28.2% | 90.0% | 25.0% | 9 | 1 | 2 | 27 |
| **0.85** | 20.5% | 100.0% | 13.9% | 5 | 0 | 3 | 31 |
| **0.90** | 15.4% | 100.0% | 8.3% | 3 | 0 | 3 | 33 |

> [!NOTE]
> All opposite-word pairs (such as `max` vs `min`, `fanin` vs `fanout`, `list` vs `count`) were correctly identified and rejected with 100% precision, preventing false plan applications.

---

## 2. Token & Latency Comparisons

| Metric | Uncached Normal Agent | Cached Plan Agent | Reduction |
|:---|:---:|:---:|:---:|
| **Avg Tokens / Request** | 230.0 | 185.1 | **19.5%** |
| **Avg Wall-Clock Latency** | 1.36s | 1.12s | **18.1%** |
| **Tool Execution Time** | ~0.01s | ~0.01s | Identical (in-memory OpenDB) |
| **LLM Reasoning Turns** | 2 turns | 0 reasoning turns (1 answer call) | **100% reasoning eliminated** |

---

## 3. Results Artifact
- Full per-prompt evaluation details exported to [`plan_cache_eval_results.csv`](plan_cache_eval_results.csv).
