# Validation Metrics Summary

Metrics source: YOLOv8n `best.pt` from training run `vehicle_satellite_640` (see README). Trained weights and the full `runs/` tree are not in the repository; the tables match the evaluation logs `eval_640.txt` and `eval_1024.txt` in this folder.

Dataset split: validation (`config/data_dota_car.yaml` maps `test` to `val`, so these are not separate test results).

| Inference size | Precision | Recall | F1-score | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 640 | 0.2349 | 0.2619 | 0.2477 | 0.1355 | 0.0392 |
| 1024 | 0.3157 | 0.3417 | 0.3282 | 0.2104 | 0.0709 |

Interpretation: the same 640-trained YOLOv8n checkpoint performs better when evaluated at 1024 inference size, suggesting that preserving more pixel detail helps with small vehicle localization. This is an inference-resolution analysis, not a separately trained 1024 model.
