import json, math
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import torch
import matplotlib.pyplot as plt

from model import SonarDeconvCNN
from dataset_utils import find_dataset_roots, collect_samples, summarize

st.set_page_config(
    page_title="PRJ-44 | SONAR-X",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).parent

MODEL_PATH = BASE / "sonar_prj44.pth"
METRICS = BASE / "metrics.json"
SUMMARY = BASE / "dataset_summary.json"
HISTORY = BASE / "history.json"
CM_PATH = BASE / "confusion_matrix.png"
LOSS_PATH = BASE / "training_loss.png"

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp { background: #050a10; color: #e8f0f7; }
[data-testid="stHeader"] { background: rgba(5,10,16,.88); }
section[data-testid="stSidebar"] { background: #071019; border-right: 1px solid #142838; }
.block-container { padding: 1.25rem 2rem 3rem; max-width: 1500px; }

/* Sidebar */
.brand { padding: 8px 4px 22px; }
.brand-mark { width:42px;height:42px;border:1px solid #1e90b8;border-radius:12px;display:inline-flex;align-items:center;justify-content:center;color:#5ee7ff;font-weight:800;font-size:18px;background:linear-gradient(145deg,#0d2b39,#071019);box-shadow:0 0 24px rgba(38,190,230,.12); }
.brand-title { display:inline-block; vertical-align:middle; margin-left:10px; }
.brand-title b { display:block; font-size:18px; letter-spacing:.12em; }
.brand-title span { font-size:9px; color:#6d8999; letter-spacing:.16em; }
.nav-note { color:#5f7888; font-size:11px; line-height:1.6; margin-top:14px; }

/* Hero */
.hero { position:relative; overflow:hidden; padding:30px 32px; border:1px solid #153346; border-radius:22px; background:radial-gradient(circle at 85% 20%,rgba(16,125,161,.16),transparent 30%),linear-gradient(135deg,#09151f,#071019 65%); margin-bottom:18px; }
.hero:after { content:""; position:absolute; width:280px;height:280px; right:-90px;bottom:-140px;border:1px solid rgba(71,211,246,.13);border-radius:50%;box-shadow:0 0 0 30px rgba(71,211,246,.025),0 0 0 60px rgba(71,211,246,.018); }
.eyebrow { color:#58d9f6; font-size:11px; font-weight:700; letter-spacing:.18em; text-transform:uppercase; }
.hero h1 { margin:8px 0 8px; font-size:35px; line-height:1.1; letter-spacing:-.035em; }
.hero p { color:#8ea7b6; max-width:850px; margin:0; font-size:14px; }
.status-row { margin-top:20px; display:flex; gap:8px; flex-wrap:wrap; }
.pill { display:inline-block;padding:6px 10px;border-radius:999px;font-size:10px;font-weight:700;letter-spacing:.08em;border:1px solid #1d3b4b;background:#0b1a25;color:#87a9b8; }
.pill.ok { color:#79e3ae;border-color:#194c3a;background:#082119; }
.pill.blue { color:#71dfff;border-color:#14506a;background:#071f2a; }
.pill.warn { color:#ffd27b;border-color:#5a451a;background:#211806; }

/* Cards */
.card { background:linear-gradient(145deg,#0a151f,#09131c); border:1px solid #142c3b; border-radius:16px; padding:17px 18px; min-height:112px; box-shadow:0 10px 30px rgba(0,0,0,.12); }
.card-label { color:#6f8795; font-size:10px; letter-spacing:.14em; text-transform:uppercase; font-weight:700; }
.card-value { font-size:28px; font-weight:800; margin-top:7px; letter-spacing:-.03em; }
.card-sub { color:#6c8797; font-size:11px; margin-top:4px; }
.section-title { margin:22px 0 10px; font-size:16px; font-weight:750; letter-spacing:.01em; }
.section-sub { color:#6f8795; font-size:11px; margin-top:-7px; margin-bottom:12px; }
.panel { background:#08131c;border:1px solid #142c3b;border-radius:18px;padding:18px; }

/* pipeline */
.pipeline { display:flex; gap:0; align-items:stretch; overflow:auto; padding:4px 0 10px; }
.step { min-width:145px; padding:14px; background:#091721; border:1px solid #153444; border-radius:13px; }
.step small { color:#4e7485; font-weight:700; letter-spacing:.1em; }
.step b { display:block; margin-top:6px; font-size:12px; }
.arrow { min-width:28px; display:flex; align-items:center; justify-content:center; color:#2d7089; font-size:20px; }

/* Result */
.result { border:1px solid #1b4b5e;border-radius:18px;padding:20px;background:radial-gradient(circle at 90% 10%,rgba(50,190,224,.12),transparent 35%),#08151e; }
.result-label { color:#6d8999; font-size:10px; letter-spacing:.15em; font-weight:700; }
.result-class { font-size:31px;font-weight:850;margin-top:4px; }
.result-good { color:#68e4b0; }
.result-neutral { color:#6edcff; }

/* uploader */
[data-testid="stFileUploaderDropzone"] { background:#07121b; border:1px dashed #245064; }
.stButton > button { border-radius:10px; border:1px solid #1c5267; background:#0b2430; color:#c9f4ff; font-weight:700; }
.stButton > button:hover { border-color:#4fd7f4; color:white; }

/* Hide streamlit clutter */
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


def load_json(path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SonarDeconvCNN()

    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)

    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {
            k.replace("module.", "", 1): v
            for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model, device

def preprocess(img):
    im = img.convert("L").resize((128, 128), Image.Resampling.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def estimate_snr(arr):
    import cv2
    smooth = cv2.GaussianBlur(arr, (0, 0), 2)
    noise = arr - smooth
    return 10 * math.log10((np.mean(smooth ** 2) + 1e-8) / (np.mean(noise ** 2) + 1e-8))


def pct(v):
    return f"{v * 100:.1f}%"


def metric_card(label, value, sub):
    return f'<div class="card"><div class="card-label">{label}</div><div class="card-value">{value}</div><div class="card-sub">{sub}</div></div>'


metrics = load_json(METRICS)
summary = load_json(SUMMARY)
history = load_json(HISTORY)
try:
    loaded = load_model()
    load_error = None
except Exception as e:
    loaded = None
    load_error = str(e)
    
# ---------- Sidebar ----------
with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">◈</span><div class="brand-title"><b>SONAR-X</b><span>PRJ-44 RESEARCH ENGINE</span></div></div>', unsafe_allow_html=True)
    model_state = "MODEL ONLINE" if loaded else "MODEL NOT FOUND"
    dataset_state = "REAL DATA LOADED" if summary else "DATASET NOT INDEXED"
    st.markdown(f'<span class="pill ok">● {model_state}</span> <span class="pill blue">● {dataset_state}</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    page = st.radio("MODULES", ["Command Center", "Sonar Analyzer", "Model Lab", "Dataset Explorer", "Research Method"], label_visibility="visible")
    st.markdown('<div class="nav-note">AUV side-scan sonar hazard classification using a joint CNN + deconvolutional reconstruction architecture.</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.caption("LOCAL INFERENCE • NO FAKE METRICS")

# ---------- Common hero ----------
now = datetime.now().strftime("%d %b %Y • %H:%M")
st.markdown(f'''<div class="hero"><div class="eyebrow">AUV / SIDE-SCAN SONAR / PRJ-44</div><h1>Hazard Classification Command Center</h1><p>Deconvolutional feature de-noising CNN for real sonar imagery — designed to classify MILCO and NOMBO contacts while reconstructing acoustic features.</p><div class="status-row"><span class="pill ok">● SYSTEM READY</span><span class="pill blue">REAL TRAINED MODEL</span><span class="pill">LOCAL INFERENCE</span><span class="pill">{now}</span></div></div>''', unsafe_allow_html=True)

# ---------- Command Center ----------
if page == "Command Center":
    st.markdown('<div class="section-title">System telemetry</div>', unsafe_allow_html=True)
    if summary and metrics:
        cols = st.columns(5)
        vals = [
            ("TEST ACCURACY", pct(metrics["accuracy"]), "held-out test set"),
            ("VAL ACCURACY", pct(metrics.get("best_val_accuracy", metrics.get("val_accuracy", 0))), "best validation epoch"),
            ("REAL IMAGES", str(summary.get("total", "—")), "labelled sonar images"),
            ("TEST SAMPLES", str(metrics.get("test_samples", "—")), "evaluation set"),
            ("DEVICE", str(metrics.get("device", "CPU")).upper(), "inference backend"),
        ]
        for c, (a,b,d) in zip(cols, vals): c.markdown(metric_card(a,b,d), unsafe_allow_html=True)
    else:
        st.warning("Training results are not available in this project folder yet.")

    st.markdown('<div class="section-title">Processing architecture</div><div class="section-sub">Actual inference path used by the PRJ-44 model</div>', unsafe_allow_html=True)
    steps = ["RAW SONAR", "PREPROCESS", "CNN ENCODER", "LATENT FEATURES", "DECONV DECODER", "RECONSTRUCTION", "CLASSIFIER", "MILCO / NOMBO"]
    html = '<div class="pipeline">' + ''.join([f'<div class="step"><small>{i+1:02d}</small><b>{s}</b></div>' + ('<div class="arrow">›</div>' if i < len(steps)-1 else '') for i,s in enumerate(steps)]) + '</div>'
    st.markdown(html, unsafe_allow_html=True)

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown('<div class="section-title">Live analysis workspace</div>', unsafe_allow_html=True)
        if loaded:
            st.info("Upload a real side-scan sonar frame in **Sonar Analyzer** to run local model inference.")
            if CM_PATH.exists():
                st.image(str(CM_PATH), caption="Actual test-set confusion matrix", use_container_width=True)
        else:
            st.error("Trained model could not be loaded.")
            if load_error:
                st.caption(load_error)
    with right:
        st.markdown('<div class="section-title">Research KPIs</div>', unsafe_allow_html=True)
        if metrics:
            for label,key in [("Accuracy","accuracy"),("Precision","precision"),("Recall","recall"),("F1 Score","f1")]:
                st.progress(float(metrics[key]), text=f"{label}  •  {pct(metrics[key])}")
            snr = metrics.get("snr_improvement_db", 0)
            if snr >= 0: st.success(f"SNR improvement: +{snr:.2f} dB")
            else: st.warning(f"SNR change: {snr:.2f} dB — reconstruction needs further optimisation.")

# ---------- Sonar Analyzer ----------
elif page == "Sonar Analyzer":
    st.markdown('<div class="section-title">Real-time sonar inference</div><div class="section-sub">Upload a real labelled/unlabelled sonar image. No fixed prediction values are used.</div>', unsafe_allow_html=True)
    upload = st.file_uploader("DROP SONAR IMAGE HERE", type=["jpg","jpeg","png","bmp"], label_visibility="visible")
    if not upload:
        st.markdown('<div class="panel"><b>Analysis flow</b><br><br>Upload → preprocess → CNN encoder → deconvolutional reconstruction → classifier → hazard result.</div>', unsafe_allow_html=True)
    elif not loaded:
        st.error("Model file not found. Your existing trained model should be at models/sonar_prj44.pth.")
    else:
        img = Image.open(upload).convert("RGB")
        model, device = loaded
        x = preprocess(img).to(device)
        with torch.no_grad():
            # SonarDeconvCNN returns: logits, reconstructed, latent_features
            logits, recon, _ = model(x)
            probs = torch.softmax(logits, 1)[0].cpu().numpy()
            pred = int(np.argmax(probs))
            recon_arr = recon[0,0].cpu().numpy()
        labels = ["MILCO", "NOMBO"]
        label = labels[pred]
        conf = float(probs[pred])
        gray = np.asarray(img.convert("L").resize((128,128)), dtype=np.float32) / 255.0
        snr_before = estimate_snr(gray)
        snr_after = estimate_snr(np.clip(recon_arr,0,1))

        st.markdown(f'<div class="result"><div class="result-label">MODEL DECISION</div><div class="result-class {"result-good" if label=="NOMBO" else "result-neutral"}">{label}</div><div style="color:#7d96a5;font-size:12px;margin-top:4px">{"Mine-Like Contact" if label=="MILCO" else "Non-Mine-like Bottom Object"} • {conf*100:.1f}% model probability</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        a,b,c,d = st.columns(4)
        a.metric("Prediction", label)
        b.metric("Probability", f"{conf*100:.1f}%")
        c.metric("Input SNR", f"{snr_before:.2f} dB")
        d.metric("Reconstruction SNR", f"{snr_after:.2f} dB", delta=f"{snr_after-snr_before:+.2f} dB")

        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**INPUT / RAW SONAR**")
            st.image(img, use_container_width=True)
        with c2:
            st.markdown("**DECONVOLUTIONAL RECONSTRUCTION**")
            st.image(recon_arr, clamp=True, use_container_width=True)
        st.progress(conf, text=f"Model probability • {label}")
        st.caption("Prediction is generated by the trained PyTorch model stored in models/sonar_prj44.pth.")

# ---------- Model Lab ----------
elif page == "Model Lab":
    st.markdown('<div class="section-title">Model performance laboratory</div><div class="section-sub">Only values produced by the training/evaluation pipeline are displayed.</div>', unsafe_allow_html=True)
    if not metrics:
        st.warning("metrics.json not found. Run training first.")
    else:
        c = st.columns(4)
        for col,key in zip(c,["accuracy","precision","recall","f1"]): col.markdown(metric_card(key.upper(),pct(metrics[key]),"test-set metric"),unsafe_allow_html=True)
        st.markdown('<div class="section-title">Evaluation artifacts</div>', unsafe_allow_html=True)
        x,y = st.columns(2)
        with x:
            if CM_PATH.exists(): st.image(str(CM_PATH), caption="Confusion Matrix — held-out test set", use_container_width=True)
        with y:
            if LOSS_PATH.exists(): st.image(str(LOSS_PATH), caption="Training / validation loss", use_container_width=True)
        st.markdown('<div class="section-title">SNR evaluation</div>', unsafe_allow_html=True)
        snr1,snr2,snr3 = st.columns(3)
        snr1.metric("Before", f"{metrics.get('snr_before_db',0):.2f} dB")
        snr2.metric("After", f"{metrics.get('snr_after_db',0):.2f} dB")
        snr3.metric("Change", f"{metrics.get('snr_improvement_db',0):+.2f} dB")
        st.caption("The current experiment's SNR figures are the controlled evaluation values generated by the project pipeline; they should not be described as field-measured underwater SNR.")
        if history:
            st.markdown('<div class="section-title">Training trace</div>', unsafe_allow_html=True)
            if isinstance(history, dict):
                epochs = history.get("epoch", history.get("epochs"))
                train_acc = history.get("train_acc")
                val_acc = history.get("val_acc")
                if epochs and train_acc and val_acc:
                    fig, ax = plt.subplots(figsize=(9,3.5))
                    ax.plot(epochs, train_acc, marker="o", label="Train accuracy")
                    ax.plot(epochs, val_acc, marker="o", label="Validation accuracy")
                    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_ylim(0,1); ax.grid(alpha=.2); ax.legend()
                    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# ---------- Dataset Explorer ----------
elif page == "Dataset Explorer":
    st.markdown('<div class="section-title">Real sonar dataset explorer</div><div class="section-sub">Dataset information is discovered from the existing local dataset; no dataset is copied into this web package.</div>', unsafe_allow_html=True)
    if summary:
        c=st.columns(4)
        for col,(a,b,d) in zip(c,[("TOTAL",summary.get("total","—"),"real images"),("MILCO",summary.get("MILCO","—"),"mine-like"),("NOMBO",summary.get("NOMBO","—"),"non-mine"),("TEST",summary.get("test","—"),"held-out")]): col.markdown(metric_card(a,str(b),d),unsafe_allow_html=True)
        st.markdown('<div class="section-title">Class distribution</div>', unsafe_allow_html=True)
        vals=[summary.get("MILCO",0),summary.get("NOMBO",0)]
        fig,ax=plt.subplots(figsize=(7,3.2)); ax.bar(["MILCO","NOMBO"],vals); ax.set_ylabel("Images"); ax.grid(axis="y",alpha=.2); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    else:
        with st.spinner("Scanning safe project/dataset locations…"):
            roots=find_dataset_roots()
        if roots:
            samples,_,_=collect_samples(roots); st.json(summarize(samples))
            st.write("Detected roots:")
            for r in roots: st.code(str(r))
        else: st.warning("No labelled JPG/TXT dataset was detected near the project.")

# ---------- Research Method ----------
elif page == "Research Method":
    st.markdown('<div class="section-title">PRJ-44 research method</div>', unsafe_allow_html=True)
    tabs = st.tabs(["Problem", "Architecture", "Training", "Evaluation", "Limitations"])
    with tabs[0]:
        st.markdown("""### Problem statement\nAUV side-scan sonar imagery can contain acoustic reflections and seabed clutter. PRJ-44 studies a CNN that reconstructs an acoustic representation using deconvolutional layers before classification into **MILCO** and **NOMBO**.""")
    with tabs[1]:
        st.markdown("""### Model architecture\n**Input 128×128 grayscale → encoder (1→16→32→64→128) → latent feature representation → ConvTranspose2d decoder → reconstructed image.**\n\nThe bottleneck is also passed through adaptive average pooling and fully connected layers for two-class classification.""")
    with tabs[2]:
        st.markdown("""### Joint learning\nThe training objective combines classification loss with reconstruction loss. This encourages the latent representation to support both hazard classification and image reconstruction.""")
    with tabs[3]:
        st.markdown("""### Evaluation\nThe project reports Accuracy, Precision, Recall, F1-score, Confusion Matrix and SNR-related reconstruction measurements on a held-out test set.""")
    with tabs[4]:
        st.warning("Current results should be presented honestly: classification performance is moderate, and the current SNR experiment does not show a positive improvement. Further tuning, augmentation, architecture changes and a stronger denoising objective are suitable future work.")
