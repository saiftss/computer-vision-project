# Vehicle Detection in Aerial Imagery

COE-486 computer vision project for single-class vehicle detection in aerial/satellite imagery using YOLOv8n.

## Project Summary

This project trains one YOLOv8n baseline on a DOTA-derived vehicle dataset. DOTA `small-vehicle` annotations are converted from oriented quadrilaterals to YOLO horizontal boxes and mapped to one class: `car`.

The final implemented experiment is:

- Train YOLOv8n at image size `640` for 20 epochs.
- Evaluate the same trained checkpoint at inference sizes `640` and `1024`.
- Report validation metrics, training curves, and qualitative prediction images.

The `1024` result is an inference-resolution analysis, not a separately trained 1024 model.

## Setup

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If a copied virtual environment fails on macOS because of code-signing errors, recreate the environment or reinstall the native packages:

```bash
pip install --force-reinstall numpy opencv-python torch torchvision ultralytics
```

## Dataset Layout

Raw DOTA-style data should look like:

```text
DOTA/
  train/
    images/
    labelTxt/
  val/
    images/
    labelTxt/
```

Convert it to YOLO format:

```bash
python -m satvehicle.preprocessing.dota_to_yolo \
  --dota-root "/path/to/DOTA" \
  --out-root "/path/to/dota_vehicle_yolo" \
  --splits train,val
```

Update `config/data_dota_car.yaml` so `path` points to the converted YOLO dataset root:

```yaml
path: /path/to/dota_vehicle_yolo
train: images/train
val: images/val
test: images/val
```

The `test` field maps to validation data in this project, so final results should be described as validation results.

## Training

The completed baseline command was:

```bash
python -m satvehicle.train_yolov8 \
  --data config/data_dota_car.yaml \
  --model n \
  --epochs 20 \
  --imgsz 640 \
  --batch 8 \
  --device mps \
  --name vehicle_satellite_640
```

Main outputs are saved under:

```text
runs/detect/runs/detect/vehicle_satellite_640/
```

Important files:

```text
weights/best.pt
weights/last.pt
results.csv
results.png
```

## Evaluation

Evaluate the trained checkpoint at the training image size:

```bash
python -m satvehicle.evaluate \
  --weights runs/detect/runs/detect/vehicle_satellite_640/weights/best.pt \
  --data config/data_dota_car.yaml \
  --split val \
  --imgsz 640 \
  --batch 16 \
  --device cpu
```

Evaluate the same checkpoint at higher inference resolution:

```bash
python -m satvehicle.evaluate \
  --weights runs/detect/runs/detect/vehicle_satellite_640/weights/best.pt \
  --data config/data_dota_car.yaml \
  --split val \
  --imgsz 1024 \
  --batch 8 \
  --device cpu
```

Final validation summary:

| Inference size | Precision | Recall | F1-score | mAP@0.5 |
| --- | ---: | ---: | ---: | ---: |
| 640 | 0.2349 | 0.2619 | 0.2477 | 0.1355 |
| 1024 | 0.3157 | 0.3417 | 0.3282 | 0.2104 |

## Figures and Demo

Generate training curves:

```bash
python -m satvehicle.visualize curves \
  --results runs/detect/runs/detect/vehicle_satellite_640/results.csv \
  --out figures/vehicle_satellite_640
```

Run prediction visualization on validation images:

```bash
python -m satvehicle.visualize predict \
  --weights runs/detect/runs/detect/vehicle_satellite_640/weights/best.pt \
  --source "/path/to/dota_vehicle_yolo/images/val" \
  --out figures/predictions_1024 \
  --imgsz 1024 \
  --conf 0.25
```

The generated prediction images are useful for a live demo and for the qualitative section of the report.
With the current Ultralytics settings, relative prediction projects are written under `runs/detect/`, for example:

```text
runs/detect/figures/predictions_1024/pred/
```

## Known Limitations

- The model is trained on DOTA as a proxy dataset, not on UAE-specific aerial imagery.
- DOTA oriented boxes are converted to horizontal boxes, which loses orientation detail.
- The project reports validation results because no separate held-out test split is used.
- The single trained model is a baseline; higher-resolution training, tiling, and confidence-threshold tuning are future work.
