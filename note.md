| Requirement      | Status | Catatan                  |
| ---------------- | ------ | ------------------------ |
| 3 kelas          | ✅      | `metal / plastik / muka` |
| 300 data         | ✅      | via `dataset_raw`        |
| Split 70/20/10   | ✅      | sudah diverifikasi       |
| CNN              | ✅      | MobileNetV2              |
| YOLO             | ✅      | YOLOv8 pretrained        |
| Grayscale        | ✅      | grayscale → 3 channel    |
| Resize           | ✅      | 224×224                  |
| Normalisasi      | ✅      | ImageNet mean/std        |
| Augmentasi       | ✅      | flip + jitter            |
| Confusion matrix | ✅      | `eval.py`                |
| Webcam realtime  | ✅      | YOLO + CNN               |
