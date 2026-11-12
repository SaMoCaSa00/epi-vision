"""Inferência YOLO em ONNX, sem PyTorch.

Implementa o pré-processamento (letterbox), a decodificação da saída do YOLO11
e o NMS em NumPy puro. Assim o app roda leve em qualquer lugar (inclusive no
Streamlit Community Cloud) só com `onnxruntime`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

DEFAULT_CLASSES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle",
]


@dataclass
class Detection:
    cls: str
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 em pixels da imagem original

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Redimensiona mantendo proporção e completa com cinza (114) até size x size."""
    import cv2

    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, r, (left, top)


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU entre uma caixa `a` (4,) e várias caixas `b` (N, 4)."""
    x1 = np.maximum(a[0], b[:, 0])
    y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2])
    y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        order = rest[iou(boxes[i], boxes[rest]) < thr]
    return keep


def decode(output: np.ndarray, conf_thr: float, iou_thr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Saída do YOLO11 (1, 4+nc, N) -> caixas xyxy, scores e classes após NMS por classe."""
    pred = output[0].T  # (N, 4+nc)
    xywh, cls_scores = pred[:, :4], pred[:, 4:]
    cls_ids = cls_scores.argmax(1)
    scores = cls_scores[np.arange(len(cls_ids)), cls_ids]
    mask = scores >= conf_thr
    xywh, scores, cls_ids = xywh[mask], scores[mask], cls_ids[mask]
    if not len(scores):
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    xyxy = np.column_stack([
        xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
        xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2,
    ])
    keep: list[int] = []
    for c in np.unique(cls_ids):  # NMS separado por classe
        idx = np.where(cls_ids == c)[0]
        keep.extend(idx[k] for k in nms(xyxy[idx], scores[idx], iou_thr))
    keep_arr = np.array(sorted(keep, key=lambda i: -scores[i]), dtype=int)
    return xyxy[keep_arr], scores[keep_arr], cls_ids[keep_arr]


class EPIDetector:
    """Carrega o modelo ONNX e devolve detecções em coordenadas da imagem original."""

    def __init__(self, model_path: str | Path, classes: list[str] | None = None, imgsz: int | None = None):
        model_path = Path(model_path)
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.imgsz = imgsz or (shape[2] if isinstance(shape[2], int) else 640)
        meta_file = model_path.with_suffix(".json")
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        self.classes = classes or meta.get("classes", DEFAULT_CLASSES)

    def __call__(self, img_bgr: np.ndarray, conf: float = 0.35, iou_thr: float = 0.5) -> list[Detection]:
        lb, r, (dx, dy) = letterbox(img_bgr, self.imgsz)
        x = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # BGR->RGB, NCHW
        out = self.session.run(None, {self.input_name: np.ascontiguousarray(x)})[0]
        boxes, scores, cls_ids = decode(out, conf, iou_thr)
        h, w = img_bgr.shape[:2]
        dets = []
        for b, s, c in zip(boxes, scores, cls_ids):
            x1 = float(np.clip((b[0] - dx) / r, 0, w))
            y1 = float(np.clip((b[1] - dy) / r, 0, h))
            x2 = float(np.clip((b[2] - dx) / r, 0, w))
            y2 = float(np.clip((b[3] - dy) / r, 0, h))
            dets.append(Detection(self.classes[int(c)], float(s), (x1, y1, x2, y2)))
        return dets
