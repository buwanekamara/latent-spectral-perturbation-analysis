# LSPA Detailed Evaluation Report

Weights: `fast_lspa_classifier_ds.pth`  |  normalize=True  |  deep_supervision=True
Deployment threshold (frozen on imagenet/adm): 1.0000

## imagenet

| Generator | Seen | #real | #fake | AUC % | EER % | Acc@0.5 % | Cal.Acc % (±std) | Deploy Acc % | Prec % | Rec % | F1 % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| adm | seen | 5000 | 5000 | 100.00 | 0.00 | 100.00 | 100.00 ± 0.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| sdv1 | unseen | 5000 | 10000 | 100.00 | 0.00 | 99.53 | 100.00 ± 0.00 | 100.00 | 98.60 | 100.00 | 99.30 |

## tiny_genimage

| Generator | Seen | #real | #fake | AUC % | EER % | Acc@0.5 % | Cal.Acc % (±std) | Deploy Acc % | Prec % | Rec % | F1 % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| adm | seen | 500 | 500 | 99.70 | 2.50 | 81.30 | 97.25 ± 0.72 | 59.60 | 100.00 | 62.60 | 77.00 |
| biggan | unseen | 500 | 500 | 95.89 | 9.20 | 78.10 | 90.99 ± 1.08 | 58.80 | 99.30 | 56.60 | 72.10 |
| glide | unseen | 500 | 500 | 99.99 | 0.10 | 80.70 | 99.75 ± 0.24 | 60.40 | 100.00 | 61.40 | 76.08 |
| midjourney | unseen | 500 | 500 | 82.49 | 24.90 | 71.30 | 78.40 ± 1.09 | 57.10 | 76.56 | 61.40 | 68.15 |
| sdv5 | unseen | 500 | 500 | 97.70 | 7.00 | 81.20 | 93.19 ± 1.05 | 59.80 | 98.15 | 63.60 | 77.18 |
| vqdm | unseen | 500 | 500 | 96.48 | 10.00 | 81.50 | 89.49 ± 0.88 | 59.80 | 98.17 | 64.20 | 77.63 |
| wukong | unseen | 500 | 500 | 94.23 | 11.60 | 77.40 | 88.83 ± 1.22 | 58.20 | 94.19 | 58.40 | 72.10 |

## Summary

| Group | #subsets | Avg AUC % | Avg Cal.Acc % | Avg Deploy Acc % |
|---|---|---|---|---|
| Seen (in-domain) | 2 | 99.85 | 98.62 | 79.80 |
| Unseen (generalization) | 7 | 95.25 | 91.52 | 64.87 |
| tiny_genimage (all) | 7 | 95.21 | 91.13 | 59.10 |
| ALL subsets | 9 | 96.28 | 93.10 | 68.19 |

**For the thesis:** lead with AUC (threshold-free). Report the generalization row (unseen average) as the headline number, and the in-domain seen row as a one-line 'fits training distribution' note.