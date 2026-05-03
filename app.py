import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import time

# Page config
st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }

    .stApp {
        background: #0a0a0f;
        color: #e8e8f0;
    }

    .main-header {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        background: linear-gradient(135deg, #00d4ff, #7b2fff, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0;
        line-height: 1.1;
    }

    .subtitle {
        font-family: 'Space Mono', monospace;
        color: #6b6b8a;
        font-size: 0.85rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }

    .card {
        background: linear-gradient(135deg, #12121e, #1a1a2e);
        border: 1px solid #2a2a4a;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #0f0f1e, #1a1a30);
        border: 1px solid #00d4ff33;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }

    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #00d4ff;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #6b6b8a;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    .prediction-bar-container {
        background: #12121e;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        border: 1px solid #2a2a4a;
    }

    .prediction-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        color: #c0c0d8;
        display: flex;
        justify-content: space-between;
        margin-bottom: 4px;
    }

    .badge {
        display: inline-block;
        background: linear-gradient(135deg, #7b2fff, #00d4ff);
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.7rem;
        font-family: 'Space Mono', monospace;
        letter-spacing: 1px;
        color: white;
        font-weight: 700;
    }

    .section-title {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        color: #c0c0d8;
        border-left: 3px solid #7b2fff;
        padding-left: 10px;
        margin-bottom: 1rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #7b2fff, #00d4ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(123, 47, 255, 0.4) !important;
    }

    .stProgress > div > div {
        background: linear-gradient(90deg, #7b2fff, #00d4ff) !important;
    }

    .stSidebar {
        background: #0d0d1a !important;
        border-right: 1px solid #2a2a4a !important;
    }

    div[data-testid="stSidebar"] {
        background: #0d0d1a;
    }

    .stSelectbox > div > div {
        background: #12121e !important;
        border: 1px solid #2a2a4a !important;
        color: #e8e8f0 !important;
    }

    .info-tag {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        color: #00d4ff;
        background: #00d4ff11;
        border: 1px solid #00d4ff33;
        border-radius: 6px;
        padding: 2px 8px;
        display: inline-block;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Lazy import helpers (keep startup fast)
# ─────────────────────────────────────────────
@st.cache_resource
def load_tensorflow():
    import tensorflow as tf
    from tensorflow.keras import datasets, layers, models
    return tf, datasets, layers, models


@st.cache_resource
def build_and_train_model(dataset_name, epochs, _tf_modules):
    tf, datasets, layers, models = _tf_modules

    if dataset_name == "CIFAR-10":
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        class_names = ['airplane','automobile','bird','cat','deer',
                       'dog','frog','horse','ship','truck']
        input_shape = (32, 32, 3)
        num_classes = 10
    else:  # MNIST
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        x_train = x_train[..., np.newaxis]
        x_test  = x_test[..., np.newaxis]
        class_names = [str(i) for i in range(10)]
        input_shape = (28, 28, 1)
        num_classes = 10

    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    # Use small subset for fast demo
    x_train, y_train = x_train[:5000], y_train[:5000]
    x_test,  y_test  = x_test[:1000],  y_test[:1000]

    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=64,
                        validation_split=0.1,
                        verbose=0)

    _, test_acc = model.evaluate(x_test, y_test, verbose=0)

    return model, class_names, history.history, test_acc


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="main-header" style="font-size:1.8rem;">⚡ CNN Lab</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Configuration</p>', unsafe_allow_html=True)

    st.markdown('<p class="section-title">🗂 Dataset</p>', unsafe_allow_html=True)
    dataset_name = st.selectbox("Choose dataset", ["CIFAR-10", "MNIST"])

    st.markdown('<p class="section-title">⚙️ Training</p>', unsafe_allow_html=True)
    epochs = st.slider("Epochs", min_value=1, max_value=20, value=5)
    st.caption("*(5 000 samples used for fast demo)*")

    st.markdown("---")
    train_btn = st.button("🚀 Train Model", use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-title">🛠 Stack</p>', unsafe_allow_html=True)
    for tag in ["TensorFlow/Keras", "OpenCV", "Pillow", "NumPy", "Streamlit"]:
        st.markdown(f'<span class="info-tag">{tag}</span>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<h1 class="main-header">Convolutional Neural Network</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">🧠 Deep Learning · Image Classification · Real-time Inference</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────
if "model" not in st.session_state:
    st.session_state.model        = None
    st.session_state.class_names  = []
    st.session_state.history      = {}
    st.session_state.test_acc     = 0.0
    st.session_state.dataset_name = ""

# ─────────────────────────────────────────────
# Architecture overview
# ─────────────────────────────────────────────
with st.expander("📐 Model Architecture", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Block 1**
```
Conv2D(32) → BN → Conv2D(32) → MaxPool → Dropout(0.25)
```
**Block 2**
```
Conv2D(64) → BN → Conv2D(64) → MaxPool → Dropout(0.25)
```
        """)
    with col2:
        st.markdown("""
**Classifier Head**
```
Flatten → Dense(256) → Dropout(0.5) → Softmax(10)
```
**Optimizer** `Adam` · **Loss** `Sparse Categorical CE`
        """)

st.markdown("---")

# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
if train_btn:
    with st.spinner("Loading TensorFlow…"):
        tf_modules = load_tensorflow()

    progress = st.progress(0, text="Initialising…")
    status   = st.empty()

    status.info(f"📦 Loading **{dataset_name}** dataset…")
    progress.progress(10, text="Loading dataset…")
    time.sleep(0.5)

    status.info("🏋️ Training CNN — this may take a minute…")
    progress.progress(30, text="Training…")

    model, class_names, history, test_acc = build_and_train_model(
        dataset_name, epochs, tf_modules
    )

    progress.progress(90, text="Evaluating…")
    time.sleep(0.3)
    progress.progress(100, text="Done!")
    status.empty()

    st.session_state.model        = model
    st.session_state.class_names  = class_names
    st.session_state.history      = history
    st.session_state.test_acc     = test_acc
    st.session_state.dataset_name = dataset_name

    st.success(f"✅ Model trained!  Test accuracy: **{test_acc*100:.2f}%**")


# ─────────────────────────────────────────────
# Metrics + Charts (after training)
# ─────────────────────────────────────────────
if st.session_state.model is not None:
    history   = st.session_state.history
    test_acc  = st.session_state.test_acc
    n_epochs  = len(history["accuracy"])

    st.markdown('<p class="section-title">📊 Training Results</p>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{test_acc*100:.1f}%</div>
            <div class="metric-label">Test Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        best_val = max(history["val_accuracy"])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{best_val*100:.1f}%</div>
            <div class="metric-label">Best Val Acc</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        final_loss = history["loss"][-1]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{final_loss:.3f}</div>
            <div class="metric-label">Final Loss</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{n_epochs}</div>
            <div class="metric-label">Epochs</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)
    epochs_range = list(range(1, n_epochs + 1))

    with chart_col1:
        import pandas as pd
        acc_df = pd.DataFrame({
            "Epoch": epochs_range,
            "Train Accuracy": history["accuracy"],
            "Val Accuracy":   history["val_accuracy"],
        }).set_index("Epoch")
        st.markdown('<p class="section-title">Accuracy Curve</p>', unsafe_allow_html=True)
        st.line_chart(acc_df)

    with chart_col2:
        loss_df = pd.DataFrame({
            "Epoch": epochs_range,
            "Train Loss": history["loss"],
            "Val Loss":   history["val_loss"],
        }).set_index("Epoch")
        st.markdown('<p class="section-title">Loss Curve</p>', unsafe_allow_html=True)
        st.line_chart(loss_df)

    st.markdown("---")

    # ─────────────────────────────────────────
    # Inference section
    # ─────────────────────────────────────────
    st.markdown('<p class="section-title">🔍 Predict on Your Image</p>', unsafe_allow_html=True)

    upload_col, result_col = st.columns([1, 1], gap="large")

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload an image (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload any image to classify it using the trained CNN."
        )

        if uploaded_file:
            pil_img = Image.open(uploaded_file).convert("RGB")
            st.image(pil_img, caption="Uploaded Image", use_container_width=True)

            # Pre-process with OpenCV + NumPy
            img_np  = np.array(pil_img)
            dataset = st.session_state.dataset_name

            if dataset == "MNIST":
                target_size = (28, 28)
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                img_cv = cv2.resize(img_cv, target_size)
                img_input = img_cv.astype("float32") / 255.0
                img_input = img_input[np.newaxis, ..., np.newaxis]
            else:
                target_size = (32, 32)
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                img_cv = cv2.resize(img_cv, target_size)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                img_input = img_cv.astype("float32") / 255.0
                img_input = img_input[np.newaxis]

            predict_btn = st.button("🎯 Classify Image")

            if predict_btn:
                with st.spinner("Running inference…"):
                    preds = st.session_state.model.predict(img_input, verbose=0)[0]

                with result_col:
                    top_idx   = int(np.argmax(preds))
                    top_label = st.session_state.class_names[top_idx]
                    top_conf  = float(preds[top_idx])

                    st.markdown(f"""
                    <div class="card" style="text-align:center; border-color:#7b2fff55;">
                        <div style="font-size:3rem;">🏆</div>
                        <div style="font-family:'Space Mono',monospace; font-size:1.8rem;
                                    color:#00d4ff; font-weight:700; margin:0.5rem 0;">
                            {top_label.upper()}
                        </div>
                        <div style="font-size:1rem; color:#6b6b8a;">
                            Confidence: <strong style="color:#ff6b6b;">{top_conf*100:.1f}%</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown('<p class="section-title" style="margin-top:1rem;">All Class Probabilities</p>',
                                unsafe_allow_html=True)

                    sorted_idx = np.argsort(preds)[::-1]
                    for i in sorted_idx:
                        label = st.session_state.class_names[i]
                        conf  = float(preds[i])
                        bar_color = "#00d4ff" if i == top_idx else "#2a2a4a"
                        st.markdown(f"""
                        <div class="prediction-bar-container">
                            <div class="prediction-label">
                                <span>{label}</span>
                                <span style="color:#00d4ff;">{conf*100:.1f}%</span>
                            </div>
                        </div>""", unsafe_allow_html=True)
                        st.progress(conf)

else:
    # Placeholder before training
    st.markdown("""
    <div class="card" style="text-align:center; padding:3rem; border-style:dashed;">
        <div style="font-size:4rem; margin-bottom:1rem;">🧠</div>
        <div style="font-family:'Space Mono',monospace; color:#6b6b8a; font-size:0.9rem;">
            Choose a dataset and click <strong style="color:#00d4ff;">Train Model</strong> to begin.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-family:'Space Mono',monospace;
            font-size:0.7rem; color:#3a3a5a; padding:1rem 0;">
    CNN Web App · TensorFlow/Keras · OpenCV · Pillow · NumPy · Streamlit
</div>
""", unsafe_allow_html=True)
