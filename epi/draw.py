"""Desenho das caixas e rótulos sobre a imagem."""
from __future__ import annotations

import cv2
import numpy as np

from .compliance import Status, Worker
from .detector import Detection

GREEN, RED, AMBER, GREY = (80, 200, 60), (40, 40, 230), (30, 170, 240), (160, 160, 160)
PPE_COLORS = {
    "Hardhat": (60, 200, 60), "Safety Vest": (0, 200, 255), "Mask": (220, 160, 40),
    "NO-Hardhat": (40, 40, 230), "NO-Safety Vest": (40, 40, 230), "NO-Mask": (40, 40, 230),
    "Safety Cone": (0, 140, 255), "machinery": (200, 120, 0), "vehicle": (200, 120, 0),
}
NAMES_PT = {
    "Hardhat": "capacete", "Safety Vest": "colete", "Mask": "mascara",
    "NO-Hardhat": "sem capacete", "NO-Safety Vest": "sem colete", "NO-Mask": "sem mascara",
    "Safety Cone": "cone", "machinery": "maquina", "vehicle": "veiculo", "Person": "pessoa",
}


def _label(img, text, x, y, color):
    scale = max(0.45, min(img.shape[:2]) / 1100)
    th = max(1, int(scale * 2))
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    y = max(y, tht + 6)
    cv2.rectangle(img, (int(x), int(y - tht - 6)), (int(x + tw + 6), int(y)), color, -1)
    cv2.putText(img, text, (int(x + 3), int(y - 4)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), th, cv2.LINE_AA)


def annotate(img: np.ndarray, dets: list[Detection], workers: list[Worker], exigidos: list[str],
             indeterminado_conta: bool = False, show_ppe: bool = True, ids: list[int] | None = None) -> np.ndarray:
    out = img.copy()
    lw = max(2, int(min(img.shape[:2]) / 300))
    if show_ppe:
        for d in dets:
            if d.cls == "Person":
                continue
            x1, y1, x2, y2 = map(int, d.box)
            cv2.rectangle(out, (x1, y1), (x2, y2), PPE_COLORS.get(d.cls, GREY), max(1, lw - 1))
    for k, w in enumerate(workers):
        x1, y1, x2, y2 = map(int, w.box)
        falt = w.faltando(exigidos, indeterminado_conta)
        indet = [i for i in exigidos if w.itens.get(i) == Status.INDETERMINADO]
        color = RED if falt else (AMBER if indet else GREEN)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, lw + 1)
        tag = f"#{ids[k]} " if ids else ""
        if falt:
            text = tag + "SEM " + "+".join(f.upper() for f in falt)
        elif indet:
            text = tag + "verificar " + "+".join(indet)
        else:
            text = tag + "OK"
        _label(out, text, x1, y1, color)
    return out
