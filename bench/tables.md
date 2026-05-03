## Benchmark Tables

### level1_in_distribution

| Strategy | ExactMatch | BLEU | chrF | OverCorr | Unchanged@Correct | ChangedRate |
|---|---|---|---|---|---|---|
| t5_top1 | **0.3548** | **83.6648** | **93.1841** | **0.2857** | 0.7143 | 0.7584 |
| bleu_ranker | 0.1909 | 80.7889 | 92.3426 | 0.1429 | **0.8571** | 0.5280 |
| conservative | 0.3265 | 83.2768 | 93.0883 | 0.1905 | 0.8095 | 0.7077 |
| conservative_no_detconf | 0.3338 | 83.2326 | 93.0265 | 0.2381 | 0.7619 | 0.7281 |
| conservative_no_edit | 0.3476 | 83.2655 | 92.9719 | 0.2381 | 0.7619 | **0.7610** |

### level2_stress

| Strategy | ExactMatch | BLEU | chrF | OverCorr | Unchanged@Correct | ChangedRate |
|---|---|---|---|---|---|---|
| t5_top1 | **0.3419** | **82.6579** | **92.7328** | **0.4737** | 0.5263 | 0.7622 |
| bleu_ranker | 0.1871 | 79.5341 | 91.8224 | 0.1579 | **0.8421** | 0.5507 |
| conservative | 0.3083 | 82.1731 | 92.5407 | 0.3158 | 0.6842 | 0.7194 |
| conservative_no_detconf | 0.3142 | 82.0981 | 92.4648 | **0.4737** | 0.5263 | 0.7477 |
| conservative_no_edit | 0.3254 | 82.1539 | 92.4804 | **0.4737** | 0.5263 | **0.7734** |

### level3_synthetic

| Strategy | ExactMatch | BLEU | chrF | OverCorr | Unchanged@Correct | ChangedRate |
|---|---|---|---|---|---|---|
| t5_top1 | **0.3720** | 75.0411 | 90.5544 | 0.1304 | 0.8696 | 0.8460 |
| bleu_ranker | 0.1420 | 65.7371 | 86.9297 | 0.0652 | **0.9348** | 0.6800 |
| conservative | 0.3440 | 74.8672 | 90.4820 | 0.1087 | 0.8913 | 0.8280 |
| conservative_no_detconf | 0.3520 | **75.2174** | **90.5931** | 0.1087 | 0.8913 | 0.8340 |
| conservative_no_edit | 0.3340 | 75.1148 | 90.4999 | **0.2174** | 0.7826 | **0.8560** |
