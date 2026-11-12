import numpy as np

from epi.compliance import Status, assess, summarize
from epi.detector import Detection, decode, nms
from epi.video import IoUTracker

PERSON = (100, 100, 200, 400)  # x1, y1, x2, y2 (300 px de altura)


def test_helmet_on_head_is_ok():
    dets = [Detection("Person", 0.9, PERSON), Detection("Hardhat", 0.8, (125, 95, 175, 140))]
    [w] = assess(dets)
    assert w.itens["capacete"] == Status.OK


def test_helmet_at_feet_does_not_count():
    dets = [Detection("Person", 0.9, PERSON), Detection("Hardhat", 0.8, (125, 360, 175, 400))]
    [w] = assess(dets)
    assert w.itens["capacete"] == Status.INDETERMINADO


def test_no_hardhat_marks_missing():
    dets = [Detection("Person", 0.9, PERSON), Detection("NO-Hardhat", 0.7, (130, 100, 170, 140))]
    [w] = assess(dets)
    assert w.itens["capacete"] == Status.AUSENTE
    assert w.faltando(["capacete", "colete"]) == ["capacete"]


def test_vest_on_torso():
    dets = [Detection("Person", 0.9, PERSON), Detection("Safety Vest", 0.8, (110, 180, 190, 290))]
    [w] = assess(dets)
    assert w.itens["colete"] == Status.OK


def test_ppe_goes_to_the_right_person():
    a = (100, 100, 200, 400)
    b = (300, 100, 400, 400)
    dets = [Detection("Person", 0.9, a), Detection("Person", 0.9, b),
            Detection("Hardhat", 0.8, (325, 95, 375, 140)),      # capacete só na pessoa B
            Detection("NO-Hardhat", 0.8, (125, 100, 175, 140))]  # A sem capacete
    wa, wb = assess(dets)
    assert wa.itens["capacete"] == Status.AUSENTE
    assert wb.itens["capacete"] == Status.OK


def test_orphan_violation_becomes_inferred_worker():
    dets = [Detection("NO-Safety Vest", 0.6, (500, 200, 560, 300))]
    [w] = assess(dets)
    assert w.inferido and w.itens["colete"] == Status.AUSENTE


def test_indeterminate_policy():
    dets = [Detection("Person", 0.9, PERSON)]
    [w] = assess(dets)
    assert w.conforme(["capacete"]) is True
    assert w.conforme(["capacete"], indeterminado_conta=True) is False


def test_summary():
    dets = [Detection("Person", 0.9, (0, 0, 100, 300)), Detection("Hardhat", 0.9, (25, -5, 75, 40)),
            Detection("Safety Vest", 0.9, (10, 80, 90, 200)),
            Detection("Person", 0.9, (200, 0, 300, 300)), Detection("NO-Hardhat", 0.9, (225, 0, 275, 40))]
    s = summarize(assess(dets), ["capacete", "colete"])
    assert s["pessoas"] == 2 and s["irregulares"] == 1 and s["faltas_por_item"]["capacete"] == 1


def test_nms_removes_duplicates():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], float)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5)
    assert sorted(keep) == [0, 2]


def test_decode_yolo_output():
    nc, n = 10, 3
    out = np.zeros((1, 4 + nc, n), np.float32)
    out[0, :4, 0] = [50, 50, 20, 20]; out[0, 4 + 5, 0] = 0.9   # pessoa
    out[0, :4, 1] = [51, 50, 20, 20]; out[0, 4 + 5, 1] = 0.6   # duplicata
    out[0, :4, 2] = [10, 10, 4, 4]; out[0, 4 + 0, 2] = 0.1     # abaixo do limiar
    boxes, scores, cls = decode(out, 0.25, 0.5)
    assert len(boxes) == 1 and cls[0] == 5
    assert np.allclose(boxes[0], [40, 40, 60, 60])


def test_tracker_keeps_id_when_moving():
    tr = IoUTracker()
    a = tr.update([(0, 0, 100, 200)], 0)
    b = tr.update([(5, 0, 105, 200)], 1)
    c = tr.update([(500, 0, 600, 200)], 2)
    assert a == b and c != a
