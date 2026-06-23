# Trash Detection YOLO11

使用 YOLO11 訓練的垃圾分類偵測專案，資料集格式為 YOLO detection。類別包含：

- General waste
- glass
- metal
- paper
- plastic

資料集來源為 Roboflow Universe：<https://universe.roboflow.com/trash-tqsr9/trash-5jtgs>，授權為 CC BY 4.0。

## 專案結構

```text
.
├── data.yaml              # YOLO 資料集設定
├── train_model.py         # 訓練模型
├── webcam_detect.py       # 使用 webcam 即時偵測
├── requirements.txt       # Python 套件
├── train/                 # 訓練資料，101 張
├── valid/                 # 驗證資料，13 張
├── test/                  # 測試資料，13 張
└── weights/
    └── best.pt            # 已訓練權重，如有提供
```

`runs/`、`.venv/`、快取檔與影片輸出已放進 `.gitignore`，避免把本機環境與訓練暫存結果上傳到 GitHub。

## 安裝

建議使用 Python 3.10 以上版本。

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux：

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## 資料集

資料集設定在 `data.yaml`：

```yaml
path: .
train: train/images
val: valid/images
test: test/images
```

如果移動資料夾，請確認 `train/`、`valid/`、`test/` 仍然和 `data.yaml` 在同一層。

## 訓練模型

使用 YOLO11s 訓練：

```bash
python train_model.py --model yolo11s.pt --epochs 100 --imgsz 640 --batch 16
```

如果電腦效能較低，可以改用 YOLO11n：

```bash
python train_model.py --model yolo11n.pt --epochs 100 --imgsz 640 --batch 8
```

訓練結果會輸出到 `runs/TRASH_yolov11s/weights/`。其中 `best.pt` 是驗證表現最佳的模型。

## Webcam 即時偵測

如果 repo 中已有 `weights/best.pt`：

```bash
python webcam_detect.py
```

指定模型、鏡頭或信心門檻：

```bash
python webcam_detect.py --model weights/best.pt --camera 0 --conf 0.25
```

按 `q` 可以關閉偵測視窗。


