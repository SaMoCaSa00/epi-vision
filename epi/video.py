"""Processamento de vídeo com rastreamento simples por IoU.

Em vídeo, a mesma pessoa aparece em dezenas de quadros. Pra contar *pessoas*
irregulares (e não quadros), cada trabalhador recebe um ID que segue entre os
quadros pela maior sobreposição de caixas. Uma pessoa só conta como irregular
depois de aparecer irregular em `min_hits` quadros, o que filtra piscadas do
detector.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from .compliance import Worker, assess
from .detector import EPIDetector, iou
from .draw import annotate


@dataclass
class Track:
    tid: int
    box: tuple
    last_frame: int
    hits: int = 0
    bad_hits: int = 0
    faltas: dict = field(default_factory=dict)


class IoUTracker:
    def __init__(self, iou_thr: float = 0.3, max_age: int = 15):
        self.iou_thr, self.max_age = iou_thr, max_age
        self.tracks: list[Track] = []
        self.next_id = 1

    def update(self, boxes: list[tuple], frame: int) -> list[int]:
        alive = [t for t in self.tracks if frame - t.last_frame <= self.max_age]
        ids: list[int] = [-1] * len(boxes)
        if alive and boxes:
            tb = np.array([t.box for t in alive], dtype=float)
            pairs = []
            for i, b in enumerate(boxes):
                ious = iou(np.array(b, dtype=float), tb)
                for j, v in enumerate(ious):
                    if v >= self.iou_thr:
                        pairs.append((v, i, j))
            used_i, used_j = set(), set()
            for v, i, j in sorted(pairs, reverse=True):
                if i in used_i or j in used_j:
                    continue
                used_i.add(i)
                used_j.add(j)
                alive[j].box, alive[j].last_frame = boxes[i], frame
                ids[i] = alive[j].tid
        for i, b in enumerate(boxes):
            if ids[i] == -1:
                t = Track(self.next_id, b, frame)
                self.next_id += 1
                self.tracks.append(t)
                ids[i] = t.tid
        return ids

    def get(self, tid: int) -> Track:
        return next(t for t in self.tracks if t.tid == tid)


def process_video(
    src: str | Path, dst: str | Path, detector: EPIDetector, exigidos: list[str],
    conf: float = 0.35, stride: int = 2, min_hits: int = 3, indeterminado_conta: bool = False,
    progress: Callable[[float], None] | None = None, max_seconds: float = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Anota o vídeo e devolve (linha do tempo por quadro, resumo por pessoa)."""
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    limit = min(total, int(fps * max_seconds))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, 960 / max(w, h))
    ow, oh = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (ow, oh))

    tracker = IoUTracker()
    rows = []
    f = 0
    while f < limit:
        ok, frame = cap.read()
        if not ok:
            break
        if f % stride == 0:
            frame = cv2.resize(frame, (ow, oh))
            dets = detector(frame, conf=conf)
            workers: list[Worker] = assess(dets)
            ids = tracker.update([wk.box for wk in workers], f)
            n_bad = 0
            for wk, tid in zip(workers, ids):
                t = tracker.get(tid)
                t.hits += 1
                falt = wk.faltando(exigidos, indeterminado_conta)
                if falt:
                    t.bad_hits += 1
                    n_bad += 1
                    for it in falt:
                        t.faltas[it] = t.faltas.get(it, 0) + 1
            writer.write(annotate(frame, dets, workers, exigidos, indeterminado_conta, ids=ids))
            rows.append({"tempo_s": round(f / fps, 2), "pessoas": len(workers), "irregulares": n_bad})
            if progress:
                progress(min(1.0, f / limit))
        f += 1
    cap.release()
    writer.release()

    people = []
    for t in tracker.tracks:
        if t.hits < min_hits:
            continue
        people.append({
            "id": t.tid,
            "quadros": t.hits,
            "irregular": t.bad_hits >= min_hits,
            "faltas": ", ".join(k for k, v in t.faltas.items() if v >= min_hits) or "-",
        })
    return pd.DataFrame(rows), pd.DataFrame(people, columns=["id", "quadros", "irregular", "faltas"])
