"""Treino do detector de EPI (YOLO11n) e exportação pra ONNX.

Uso:
    python train/train.py --data /caminho/Dataset --epochs 30 --imgsz 512

`--data` aponta pra uma pasta no formato YOLO com train/, valid/ e test/,
cada uma com images/ e labels/.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

CLASSES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="pasta do dataset (train/valid/test)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--out", default="models")
    ap.add_argument("--fraction", type=float, default=1.0, help="fração do treino usada (pra treinar rápido em CPU)")
    args = ap.parse_args()

    from ultralytics import YOLO

    data_dir = Path(args.data).resolve()
    yaml_path = Path("runs_data.yaml").resolve()
    yaml_path.write_text(
        f"path: {data_dir}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: {len(CLASSES)}\nnames: {json.dumps(CLASSES)}\n"
    )

    model = YOLO(args.model)
    model.train(
        data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device="cpu", workers=2, project="runs", name="epi", exist_ok=True,
        fraction=args.fraction, patience=10, plots=True, verbose=False,
        mosaic=1.0, close_mosaic=5, cos_lr=True,
    )

    best = Path("runs/epi/weights/best.pt")
    final = YOLO(str(best))
    metrics = {}
    for split in ("val", "test"):
        r = final.val(data=str(yaml_path), split=split, imgsz=args.imgsz, device="cpu", plots=split == "test",
                      project="runs", name=f"eval_{split}", exist_ok=True, verbose=False)
        metrics[split] = {
            "mAP50": round(float(r.box.map50), 4),
            "mAP50-95": round(float(r.box.map), 4),
            "precision": round(float(r.box.mp), 4),
            "recall": round(float(r.box.mr), 4),
            "per_class_mAP50": {CLASSES[i]: round(float(v), 4) for i, v in zip(r.box.ap_class_index, r.box.ap50)},
        }

    onnx_path = Path(final.export(format="onnx", imgsz=args.imgsz, simplify=True, opset=12))
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    shutil.copy(onnx_path, out / "epi_yolo11n.onnx")
    (out / "epi_yolo11n.json").write_text(json.dumps(
        {"classes": CLASSES, "imgsz": args.imgsz, "epochs": args.epochs, "base": args.model, "metrics": metrics},
        indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
