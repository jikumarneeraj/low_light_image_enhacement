import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

import os
import zipfile
import gdown

MODEL_DIR = "mirnet_saved"
ZIP_PATH = "mirnet_saved.zip"
GDRIVE_URL = "https://drive.google.com/file/d/11Yq_mNMNVOGR9O0sHtBWZAPTNPX8UcnE/view?usp=sharing"


# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="MIRNet Image Enhancement",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
html, body { font-family: 'Poppins', sans-serif; }
.title-center { text-align: center; font-size: 40px !important; font-weight: 700 !important; color: #ffffff; }
.stApp { background: linear-gradient(135deg, #1f1c2c, #928dab); }
div[data-testid="stFileUploader"] { background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; border: 2px dashed #ffffff55; }
.stButton > button { background-color: #ff6ec7; color: white; border: none; padding: 12px 26px; border-radius: 10px; font-size: 18px; font-weight: bold; transition: 0.3s; }
.stButton > button:hover { background-color: #ff3fa6; }
.image-box { background: rgba(255,255,255,0.1); padding: 20px; border-radius: 20px; backdrop-filter: blur(10px); }
</style>
""", unsafe_allow_html=True)

# for local test
# ------------------ LOAD MODEL ------------------
# @st.cache_resource
# def load_model():
#     return tf.saved_model.load("mirnet_saved")

@st.cache_resource
def load_model():
    # Download model if not present
    if not os.path.exists(MODEL_DIR):
        with st.spinner("Downloading model..."):
            gdown.download(GDRIVE_URL, ZIP_PATH, quiet=False)

        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(".")

    return tf.saved_model.load(MODEL_DIR)



model = load_model()

# ------------------ TITLE ------------------
st.markdown('<p class="title-center"> MIRNet Image Enhancement </p>', unsafe_allow_html=True)
st.write("")

# ------------------ FILE UPLOAD ------------------
uploaded = st.file_uploader("Upload a Low-Light Image", type=["jpg", "jpeg", "png"])

# ------------------ ENHANCEMENT FUNCTION ------------------
def enhance(img: Image.Image) -> Image.Image:
    arr = tf.keras.utils.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    infer = model.signatures["serving_default"]
    # NOTE: the output tensor name may vary; change "output_0" if needed
    output = infer(tf.constant(arr))
    # pick first tensor in the dict if you don't know the key
    out_tensor = list(output.values())[0].numpy()[0]
    out = np.clip(out_tensor * 255.0, 0, 255).astype("uint8")
    return Image.fromarray(out)

# ------------------ LAYOUT ------------------
if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h4 style='text-align:center; color:white;'> Original Image</h4>", unsafe_allow_html=True)
        st.markdown('<div class="image-box">', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("<h4 style='text-align:center; color:white;'> Enhanced Image</h4>", unsafe_allow_html=True)

        if st.button("Enhance Image"):
            enhanced_img = enhance(img)

            st.markdown('<div class="image-box">', unsafe_allow_html=True)
            st.image(enhanced_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # prepare PNG bytes for download
            buf = io.BytesIO()
            enhanced_img.save(buf, format="PNG")
            buf.seek(0)

            st.download_button(
                label="⬇ Download Enhanced Image",
                data=buf,
                file_name="enhanced.png",
                mime="image/png"
            )

else:
    st.markdown("<h4 style='text-align:center; color:white;'>Upload an image to begin.</h4>", unsafe_allow_html=True)
