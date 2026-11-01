# circuitAnt Semantic Tool-Call Caching: Ablation Study Comparison

| Experiment | Strategy | Hit Rate (%) | LLM Avoidance (%) | Agreement (%) | False Hit Proxy (%) | Tokens Saved | Token Red. (%) | Cost Saved ($) | Latency Red. (%) | Cache Size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | original_llm | 0.0% | 0.0% | 0.0% | 0.0% | 0 | 0.0% | $0.0000 | -0.0% | 660 |
| E1 | exact | 32.4% | 32.4% | 100.0% | 0.0% | 1,714,834 | 32.5% | $0.2633 | 31.7% | 446 |
| E2 | semantic | 84.4% | 84.4% | 42.5% | 57.5% | 4,401,366 | 83.5% | $0.6756 | 82.8% | 103 |
| E3 | semantic_rerank | 86.7% | 86.7% | 37.2% | 62.8% | 4,531,881 | 85.9% | $0.6956 | 84.0% | 88 |
| E4 | semantic_structural | 85.9% | 85.9% | 98.1% | 1.9% | 4,490,564 | 85.2% | $0.6893 | 84.8% | 93 |
| E5 | semantic_both | 86.5% | 86.5% | 97.0% | 3.0% | 4,522,653 | 85.8% | $0.6942 | 83.8% | 89 |
| E6 | full_hybrid | 86.5% | 86.5% | 97.0% | 3.0% | 4,522,653 | 85.8% | $0.6942 | 83.8% | 89 |
