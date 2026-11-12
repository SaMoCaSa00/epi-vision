"""Regra de conformidade: associa EPIs a cada pessoa detectada.

O modelo detecta objetos soltos (pessoa, capacete, colete, "sem capacete"...).
Pra saber *quem* está irregular, cada EPI é ligado à pessoa cujo corpo contém
o centro da caixa do EPI, na região do corpo onde ele deveria estar:

- capacete / sem capacete  -> terço superior da pessoa (cabeça)
- colete / sem colete      -> faixa do tronco
- máscara / sem máscara    -> terço superior (rosto)

Cada item fica em um de três estados: OK (EPI visto), AUSENTE (o modelo viu a
falta do EPI) ou INDETERMINADO (nada visível, ex.: pessoa de costas/cortada).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .detector import Detection


class Status(str, Enum):
    OK = "ok"
    AUSENTE = "ausente"
    INDETERMINADO = "indeterminado"


# item -> (classe positiva, classe negativa, faixa vertical relativa na pessoa)
ITEMS = {
    "capacete": ("Hardhat", "NO-Hardhat", (-0.10, 0.40)),
    "colete": ("Safety Vest", "NO-Safety Vest", (0.15, 0.80)),
    "mascara": ("Mask", "NO-Mask", (-0.05, 0.40)),
}


@dataclass
class Worker:
    box: tuple[float, float, float, float]
    conf: float
    itens: dict[str, Status] = field(default_factory=dict)
    inferido: bool = False  # True quando a pessoa foi inferida só a partir de um EPI/falta

    def faltando(self, exigidos: list[str], indeterminado_conta: bool = False) -> list[str]:
        bad = {Status.AUSENTE} | ({Status.INDETERMINADO} if indeterminado_conta else set())
        return [i for i in exigidos if self.itens.get(i, Status.INDETERMINADO) in bad]

    def conforme(self, exigidos: list[str], indeterminado_conta: bool = False) -> bool:
        return not self.faltando(exigidos, indeterminado_conta)


def _center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _inside_region(item_box, person_box, band: tuple[float, float], pad_x: float = 0.15) -> bool:
    cx, cy = _center(item_box)
    x1, y1, x2, y2 = person_box
    w, h = x2 - x1, y2 - y1
    return (x1 - pad_x * w <= cx <= x2 + pad_x * w) and (y1 + band[0] * h <= cy <= y1 + band[1] * h)


def assess(detections: list[Detection], min_person_conf: float = 0.35) -> list[Worker]:
    persons = [d for d in detections if d.cls == "Person" and d.conf >= min_person_conf]
    workers = [Worker(p.box, p.conf) for p in persons]
    used: set[int] = set()

    for item, (pos_cls, neg_cls, band) in ITEMS.items():
        for wk in workers:
            best_pos = best_neg = 0.0
            for i, d in enumerate(detections):
                if d.cls not in (pos_cls, neg_cls) or not _inside_region(d.box, wk.box, band):
                    continue
                used.add(i)
                if d.cls == pos_cls:
                    best_pos = max(best_pos, d.conf)
                else:
                    best_neg = max(best_neg, d.conf)
            if best_pos == 0 and best_neg == 0:
                wk.itens[item] = Status.INDETERMINADO
            else:
                wk.itens[item] = Status.OK if best_pos >= best_neg else Status.AUSENTE

    # "sem capacete"/"sem colete" que não caiu em nenhuma pessoa detectada:
    # ainda é uma infração, então vira um trabalhador inferido
    neg_to_item = {neg: item for item, (_, neg, _) in ITEMS.items()}
    for i, d in enumerate(detections):
        if i in used or d.cls not in neg_to_item:
            continue
        wk = Worker(d.box, d.conf, {it: Status.INDETERMINADO for it in ITEMS}, inferido=True)
        wk.itens[neg_to_item[d.cls]] = Status.AUSENTE
        workers.append(wk)
    return workers


def summarize(workers: list[Worker], exigidos: list[str], indeterminado_conta: bool = False) -> dict:
    total = len(workers)
    irregulares = [w for w in workers if not w.conforme(exigidos, indeterminado_conta)]
    por_item = {i: sum(1 for w in workers if i in w.faltando(exigidos, indeterminado_conta)) for i in exigidos}
    return {
        "pessoas": total,
        "conformes": total - len(irregulares),
        "irregulares": len(irregulares),
        "taxa_conformidade": (total - len(irregulares)) / total if total else 1.0,
        "faltas_por_item": por_item,
    }
