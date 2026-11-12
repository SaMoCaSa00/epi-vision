"""EPI Vision — detector de Equipamentos de Proteção Individual com YOLO.

Rode com: streamlit run app.py
"""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from epi.compliance import ITEMS, Status, assess, summarize
from epi.detector import EPIDetector
from epi.draw import NAMES_PT, annotate
from epi.video import process_video

ROOT = Path(__file__).parent
MODEL = ROOT / "models" / "epi_yolo11n.onnx"
SAMPLES = sorted((ROOT / "samples").glob("*.jpg"))
RED, GREEN, AMBER = "#FF2A1F", "#22C55E", "#F59E0B"

st.set_page_config(page_title="EPI Vision", page_icon="🦺", layout="wide")


@st.cache_resource(show_spinner="Carregando modelo…")
def load_detector() -> EPIDetector:
    return EPIDetector(MODEL)


def read_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Arquivo de imagem inválido")
    return img


def to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


det = load_detector()
meta_path = MODEL.with_suffix(".json")
meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

# ───────────────────────── Sidebar ─────────────────────────
with st.sidebar:
    st.markdown("## 🦺 EPI Vision")
    st.caption("Detecção de EPI com YOLO11 + regra de conformidade por trabalhador.")
    exigidos = st.multiselect(
        "EPIs obrigatórios", list(ITEMS), default=["capacete", "colete"],
        format_func=lambda s: {"capacete": "Capacete", "colete": "Colete refletivo", "mascara": "Máscara"}[s],
    )
    conf = st.slider("Confiança mínima", 0.10, 0.90, 0.35, 0.05)
    indet = st.toggle("Contar 'não visível' como irregular", value=False,
                      help="Pessoa de costas ou cortada pode não mostrar o EPI. Ligado = postura mais rigorosa.")
    show_ppe = st.toggle("Mostrar caixas dos EPIs", value=True)
    if meta.get("metrics"):
        m = meta["metrics"].get("test") or meta["metrics"].get("val")
        st.divider()
        st.caption(f"Modelo: YOLO11n treinado {meta.get('epochs', '?')} épocas · {meta.get('imgsz', '?')} px")
        st.caption(f"mAP50 (teste): **{m['mAP50']:.2f}** · precisão {m['precision']:.2f} · recall {m['recall']:.2f}")

st.markdown(
    f"<h1 style='margin-bottom:0'>EPI <span style='color:{RED}'>Vision</span></h1>",
    unsafe_allow_html=True,
)
st.caption("Fiscalização automática de capacete e colete em canteiros de obra e áreas industriais (NR-6).")

tab_img, tab_vid, tab_model = st.tabs(["🖼️ Imagem", "🎞️ Vídeo", "📊 Sobre o modelo"])

# ───────────────────────── Imagem ─────────────────────────
with tab_img:
    c1, c2 = st.columns([1, 2])
    with c1:
        up = st.file_uploader("Envie uma foto", type=["jpg", "jpeg", "png", "webp"])
        sample = None
        if SAMPLES:
            sample = st.selectbox("…ou escolha um exemplo", ["—"] + [p.name for p in SAMPLES])
        cam = st.toggle("Usar câmera")
        snap = st.camera_input("Tire uma foto") if cam else None

    src = snap or up
    if src is not None:
        img = read_image(src.getvalue())
    elif sample and sample != "—":
        img = cv2.imread(str(ROOT / "samples" / sample))
    else:
        img = cv2.imread(str(SAMPLES[0])) if SAMPLES else None

    if img is not None:
        dets = det(img, conf=conf)
        workers = assess(dets)
        res = summarize(workers, exigidos, indet)
        with c2:
            st.image(to_rgb(annotate(img, dets, workers, exigidos, indet, show_ppe)), width="stretch")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Pessoas", res["pessoas"])
        k2.metric("Conformes", res["conformes"])
        k3.metric("Irregulares", res["irregulares"])
        k4.metric("Conformidade", f"{res['taxa_conformidade'] * 100:.0f}%")

        if res["irregulares"]:
            faltas = ", ".join(f"{v} sem {k}" for k, v in res["faltas_por_item"].items() if v)
            st.error(f"🚨 {res['irregulares']} trabalhador(es) irregular(es): {faltas}.")
        elif res["pessoas"]:
            st.success("✅ Todos os trabalhadores visíveis estão com os EPIs exigidos.")

        rows = []
        for i, w in enumerate(workers, 1):
            row = {"#": i, "confiança": round(w.conf, 2), "situação": "IRREGULAR" if w.faltando(exigidos, indet) else "ok"}
            for it in exigidos:
                row[it] = {Status.OK: "✅", Status.AUSENTE: "❌", Status.INDETERMINADO: "❔"}[w.itens.get(it, Status.INDETERMINADO)]
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True, width="stretch")
            st.download_button("Baixar relatório (CSV)", df.to_csv(index=False).encode(), "relatorio_epi.csv", "text/csv")

        with st.expander("Detecções brutas do modelo"):
            st.dataframe(pd.DataFrame([{"classe": NAMES_PT.get(d.cls, d.cls), "confiança": round(d.conf, 2)} for d in dets]),
                         hide_index=True, width="stretch")

# ───────────────────────── Vídeo ─────────────────────────
with tab_vid:
    st.markdown("Envie um vídeo curto (até 60 s são analisados). Cada pessoa ganha um **ID** que a segue entre os "
                "quadros, então o relatório conta **pessoas** irregulares, não quadros.")
    vid = st.file_uploader("Vídeo", type=["mp4", "mov", "avi", "mkv"], key="vid")
    stride = st.select_slider("Analisar 1 a cada N quadros", [1, 2, 3, 4, 5], value=3)
    if vid is not None and st.button("Analisar vídeo", type="primary"):
        with tempfile.TemporaryDirectory() as tmp:
            src_path = Path(tmp) / f"in{Path(vid.name).suffix}"
            src_path.write_bytes(vid.getvalue())
            raw = Path(tmp) / "raw.mp4"
            bar = st.progress(0.0, "Processando…")
            timeline, people = process_video(src_path, raw, det, exigidos, conf, stride, 3, indet,
                                             progress=lambda p: bar.progress(p, "Processando…"))
            bar.empty()
            final = Path(tmp) / "out.mp4"
            try:  # re-encoda em H.264 pra tocar no navegador
                import imageio_ffmpeg, subprocess
                subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(raw),
                                "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(final)], check=True)
            except Exception:
                final = raw
            st.video(final.read_bytes())

        n_bad = int(people["irregular"].sum()) if len(people) else 0
        k1, k2, k3 = st.columns(3)
        k1.metric("Pessoas rastreadas", len(people))
        k2.metric("Irregulares", n_bad)
        k3.metric("Conformidade", f"{(1 - n_bad / len(people)) * 100:.0f}%" if len(people) else "—")
        if len(timeline):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=timeline["tempo_s"], y=timeline["pessoas"], name="pessoas", line=dict(color="#A1A1AA")))
            fig.add_trace(go.Scatter(x=timeline["tempo_s"], y=timeline["irregulares"], name="irregulares",
                                     line=dict(color=RED), fill="tozeroy"))
            fig.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=30, b=10),
                              xaxis_title="tempo (s)", yaxis_title="pessoas", legend=dict(orientation="h", y=1.15))
            st.plotly_chart(fig, width="stretch")
        if len(people):
            st.dataframe(people, hide_index=True, width="stretch")
            st.download_button("Baixar relatório (CSV)", people.to_csv(index=False).encode(), "relatorio_video.csv")

# ───────────────────────── Modelo ─────────────────────────
with tab_model:
    if not meta.get("metrics"):
        st.info("Métricas do modelo indisponíveis.")
    else:
        t = meta["metrics"]["test"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("mAP50", f"{t['mAP50']:.2f}")
        c2.metric("mAP50-95", f"{t['mAP50-95']:.2f}")
        c3.metric("Precisão", f"{t['precision']:.2f}")
        c4.metric("Recall", f"{t['recall']:.2f}")
        pc = pd.Series(t["per_class_mAP50"]).sort_values()
        fig = go.Figure(go.Bar(x=pc.values, y=[NAMES_PT.get(k, k) for k in pc.index], orientation="h",
                               marker_color=[RED if k.startswith("NO-") else "#38BDF8" for k in pc.index]))
        fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=40, b=10),
                          title="mAP50 por classe (conjunto de teste)", xaxis_range=[0, 1])
        st.plotly_chart(fig, width="stretch")
        for img_name, cap in [("confusion_matrix_normalized.png", "Matriz de confusão (teste)"),
                              ("results.png", "Curvas de treino")]:
            p = ROOT / "docs" / img_name
            if p.exists():
                st.image(str(p), caption=cap, width="stretch")
    st.markdown(
        "**Como funciona a regra de conformidade:** o modelo detecta pessoas e EPIs separadamente. "
        "Cada capacete é ligado à pessoa cuja *cabeça* contém o centro da caixa, e cada colete à pessoa cujo "
        "*tronco* contém o centro. Assim o sistema diz **quem** está irregular, e não só que existe um "
        "\"sem capacete\" na imagem."
    )
