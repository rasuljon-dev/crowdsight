"""
CrowdSight Streamlit Dashboard
================================
Real-time crowd density estimation with live webcam support.

Run:
    cd crowdsight
    streamlit run dashboard/app.py

Requirements (in addition to requirements.txt):
    pip install streamlit>=1.32.0 opencv-python-headless>=4.9.0
"""

import base64
import io
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ── make sure CrowdSight project root is on sys.path ────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.inference import CrowdInferenceEngine   # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
#  Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CrowdSight",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* metric cards */
[data-testid="metric-container"] {
    background: #1e2130;
    border: 1px solid #2e3250;
    border-radius: 10px;
    padding: 16px 20px;
}
/* status badge */
.badge-ok   { color: #22c55e; font-weight: 700; }
.badge-warn { color: #f59e0b; font-weight: 700; }
.badge-err  { color: #ef4444; font-weight: 700; }
/* thin divider */
hr { border-color: #2e3250; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar — settings
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    # Model selection
    model_choice = st.selectbox(
        "Model",
        options=["csrnet", "clip_ebc", "mac_cnn"],
        index=0,
        help=(
            "csrnet   — CSRNet + DM-Count (CVPR 2018), MAE ~59.7\n"
            "clip_ebc — CLIP ViT-B/16 (ICME 2025), MAE ~56\n"
            "mac_cnn  — thesis baseline, MAE ~80"
        ),
    )

    # Checkpoint path
    ckpt_default = str(ROOT / "checkpoints" / "best.pth")
    weights_path = st.text_input(
        "Checkpoint path",
        value=ckpt_default,
        help="Leave blank to run with random (untrained) weights.",
    )
    weights_path = weights_path.strip() or None

    st.markdown("---")

    # Webcam settings
    st.subheader("📷 Webcam")
    webcam_fps = st.slider("Target FPS", min_value=1, max_value=30, value=5)
    webcam_resolution = st.selectbox(
        "Resolution",
        options=["640×480", "1280×720", "1920×1080"],
        index=0,
    )
    webcam_w, webcam_h = (int(x) for x in webcam_resolution.split("×"))

    st.markdown("---")

    # Heatmap opacity
    overlay_alpha = st.slider(
        "Heatmap overlay opacity", min_value=0.0, max_value=1.0, value=0.55, step=0.05
    )

    st.markdown("---")
    st.caption("CrowdSight · MAC-CNN → CSRNet → CLIP-EBC")

# ─────────────────────────────────────────────────────────────────────────────
#  Load model (cached so it survives reruns)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_engine(model_name: str, ckpt: str | None) -> CrowdInferenceEngine:
    return CrowdInferenceEngine(weights_path=ckpt, model_name=model_name)


engine: CrowdInferenceEngine = load_engine(model_choice, weights_path)

# ─────────────────────────────────────────────────────────────────────────────
#  Helper: run inference + compose overlay
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(pil_img: Image.Image) -> dict:
    """Returns inference result dict with an extra 'overlay' PIL image."""
    result = engine.analyze(pil_img)

    # Decode base64 heatmap
    heatmap_bytes = base64.b64decode(result["density_map"])
    heatmap_pil   = Image.open(io.BytesIO(heatmap_bytes)).convert("RGBA")

    # Resize heatmap to match original
    heatmap_pil = heatmap_pil.resize(pil_img.size, Image.BILINEAR)

    # Composite overlay
    base_rgba = pil_img.convert("RGBA")
    heat_rgba = heatmap_pil.copy()
    heat_rgba.putalpha(int(overlay_alpha * 255))
    overlay = Image.alpha_composite(base_rgba, heat_rgba).convert("RGB")

    result["overlay"] = overlay
    result["heatmap"] = heatmap_pil.convert("RGB")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Status bar
# ─────────────────────────────────────────────────────────────────────────────
status_col1, status_col2, status_col3, status_col4 = st.columns(4)
with status_col1:
    st.metric("Model", model_choice.upper())
with status_col2:
    st.metric("Device", engine.device_name.upper())
with status_col3:
    w_status = "✅ Loaded" if engine.weights_loaded else "⚠️ Random"
    st.metric("Weights", w_status)
with status_col4:
    st.metric("Output stride", str(getattr(engine.model, "output_stride", "?")))

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  Tabs: Image upload | Live webcam
# ─────────────────────────────────────────────────────────────────────────────
tab_upload, tab_webcam = st.tabs(["📁 Image Upload", "📷 Live Webcam"])

# ═══════════════════════════════════════════════════════════════════════════
#  TAB 1 — Image upload
# ═══════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.subheader("Upload an image")
    uploaded = st.file_uploader(
        "Drop a crowd image here (JPEG / PNG)",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("RGB")

        with st.spinner("Running inference…"):
            result = run_inference(pil_img)

        # ── Metrics row ───────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        m1.metric("🧮 Estimated count", f"{result['count']:,.0f}")
        m2.metric("⚡ Inference time", f"{result['inference_time_ms']:.1f} ms")
        m3.metric("📐 Image size", f"{pil_img.width} × {pil_img.height}")

        # ── Image panels ─────────────────────────────────────────────────
        col_orig, col_heat, col_overlay = st.columns(3)
        with col_orig:
            st.markdown("**Original**")
            st.image(pil_img, use_container_width=True)
        with col_heat:
            st.markdown("**Density heatmap**")
            st.image(result["heatmap"], use_container_width=True)
        with col_overlay:
            st.markdown("**Overlay**")
            st.image(result["overlay"], use_container_width=True)

        # ── Download overlay ──────────────────────────────────────────────
        buf = io.BytesIO()
        result["overlay"].save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download overlay PNG",
            data=buf.getvalue(),
            file_name=f"crowdsight_overlay_{int(time.time())}.png",
            mime="image/png",
        )
    else:
        st.info("Upload an image to get started.")

# ═══════════════════════════════════════════════════════════════════════════
#  TAB 2 — Live webcam
# ═══════════════════════════════════════════════════════════════════════════
with tab_webcam:
    st.subheader("Live webcam crowd counting")

    # Check for OpenCV
    try:
        import cv2
        HAS_CV2 = True
    except ImportError:
        HAS_CV2 = False

    if not HAS_CV2:
        st.error(
            "OpenCV is required for webcam mode. Install it with:\n\n"
            "```\npip install opencv-python-headless\n```"
        )
        st.stop()

    # ── Controls ─────────────────────────────────────────────────────────
    ctrl_col1, ctrl_col2 = st.columns([1, 4])
    with ctrl_col1:
        run_webcam = st.toggle("▶ Start / ⏹ Stop", value=False, key="webcam_toggle")
    with ctrl_col2:
        cam_idx = st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)

    # Placeholders updated in the loop
    wc_status   = st.empty()
    wc_metrics  = st.empty()
    wc_col1, wc_col2 = st.columns(2)
    wc_frame    = wc_col1.empty()
    wc_heat     = wc_col2.empty()

    # Count history for mini chart
    if "count_history" not in st.session_state:
        st.session_state.count_history = []
    history_chart = st.empty()

    if run_webcam:
        cap = cv2.VideoCapture(int(cam_idx))
        if not cap.isOpened():
            st.error(f"Cannot open camera {cam_idx}. Check the camera index.")
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  webcam_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_h)
            wc_status.success(f"Camera {cam_idx} opened — streaming at {webcam_fps} FPS target")

            frame_delay = 1.0 / webcam_fps

            while st.session_state.get("webcam_toggle", False):
                t_frame = time.perf_counter()

                ret, frame_bgr = cap.read()
                if not ret:
                    wc_status.warning("Frame read failed — retrying…")
                    time.sleep(0.1)
                    continue

                # BGR → RGB PIL
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)

                result = run_inference(pil_frame)

                # Update history
                st.session_state.count_history.append(result["count"])
                if len(st.session_state.count_history) > 60:
                    st.session_state.count_history = st.session_state.count_history[-60:]

                # Display
                wc_metrics.markdown(
                    f"**Count:** `{result['count']:,.0f}` &nbsp;|&nbsp; "
                    f"**Inference:** `{result['inference_time_ms']:.1f} ms`"
                )
                wc_frame.image(result["overlay"], caption="Live overlay", use_container_width=True)
                wc_heat.image(result["heatmap"],  caption="Density map",  use_container_width=True)

                # Count trend chart
                history_chart.line_chart(
                    {"Crowd count": st.session_state.count_history},
                    height=120,
                )

                # Pace to target FPS
                elapsed = time.perf_counter() - t_frame
                sleep   = max(0.0, frame_delay - elapsed)
                time.sleep(sleep)

            cap.release()
            wc_status.info("Webcam stopped.")
    else:
        wc_status.info("Toggle 'Start' to begin live crowd counting from your webcam.")
        if st.session_state.count_history:
            history_chart.line_chart(
                {"Crowd count": st.session_state.count_history},
                height=120,
            )
