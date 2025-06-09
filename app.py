import streamlit as st
import onnxruntime as rt
import numpy as np
import cv2
import json
import pyperclip
from PIL import Image
import time
import io

# Set page config with more appealing settings
st.set_page_config(
    page_title="Handwritten Math to LaTeX",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling and animations
st.markdown("""
<style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .stApp {
        background-color: #f8fafc;
    }
    .main-title {
        font-size: 2.5rem !important;
        color: #2c3e50 !important;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeIn 0.8s ease-out;
    }
    .upload-box {
        border: 2px dashed #6366f1;
        border-radius: 15px;
        padding: 2.5rem;
        text-align: center;
        background-color: rgba(99, 102, 241, 0.05);
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }
    .upload-box:hover {
        border-color: #4f46e5;
        background-color: rgba(99, 102, 241, 0.1);
        transform: translateY(-2px);
    }
    .result-container {
        border-radius: 15px;
        padding: 1.5rem;
        background-color: white;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        animation: fadeIn 0.6s ease-out;
    }
    .latex-rendered {
        font-size: 2rem;
        text-align: center;
        padding: 2rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
        animation: pulse 2s infinite;
    }
    .copy-btn {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    .copy-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .image-preview {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .image-preview:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 20px -5px rgba(0, 0, 0, 0.15);
    }
    .tab-content {
        padding: 1.5rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
    }
    .success-toast {
        background-color: #10b981 !important;
        color: white !important;
    }
    .sidebar .sidebar-content {
        background-color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

# -------- Load ONNX model and LaTeX keys --------
@st.cache_resource
def load_model():
    model = rt.InferenceSession('./model.onnx', providers=['CPUExecutionProvider'])
    ltx_index = json.load(open('keys.json'))
    return model, ltx_index

model, ltx_index = load_model()
input_name = model.get_inputs()[0].name
output_name = model.get_outputs()[0].name

# -------- Helper Functions ----------
def resize(img, w=256, h=256):
    p = max(img.shape[:2] / np.array([h, w]))
    r = img.shape[:2] / p
    img = cv2.resize(img, (int(r[1]), int(r[0])))
    canvas = np.zeros((h, w, 3))
    offset = ((h - img.shape[0]) // 2, (w - img.shape[1]) // 2)
    canvas[offset[0]:offset[0]+img.shape[0], offset[1]:offset[1]+img.shape[1]] = img
    return canvas

def predict_latex(img):
    input_img = resize(img, w=1024, h=192) / 255
    input_tensor = input_img[None].astype(np.float32)
    res = model.run([output_name], {input_name: input_tensor})[0][0]
    res = np.argmax(res, axis=1)
    latex = ''.join([ltx_index.get(str(x - 1), '') if x != 0 else ' ' for x in res])
    return latex.strip()

def fix_latex(latex):
    replacements = {
        '**': '\\times',
        '*': '\\times',
        'frac': '\\frac',
        'sqrt': '\\sqrt',
        'pi': '\\pi',
        'alpha': '\\alpha',
        'beta': '\\beta',
        '\\\\': '\\',
        '+-': '-'
    }
    for old, new in replacements.items():
        latex = latex.replace(old, new)
    return latex

# -------- Streamlit UI --------
# Animated header
st.markdown("""
<div class="main-title">
    <span style="color: #6366f1;">✍️</span> Handwritten Math to LaTeX Converter
</div>
<p style="text-align: center; color: #64748b; margin-bottom: 2rem;">
    Transform your handwritten equations into beautiful LaTeX code instantly
</p>
""", unsafe_allow_html=True)

# File uploader with enhanced styling
with st.container():
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload an image", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="file_uploader"
    )
    st.markdown('<p style="color: #64748b; margin-top: 1rem;">Supports JPG, JPEG, and PNG formats</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Read and process image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Create columns for side-by-side comparison
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown('<h3 style="color: #6366f1; margin-bottom: 1rem;">📷 Your Handwritten Equation</h3>', unsafe_allow_html=True)
        st.markdown('<div class="image-preview">', unsafe_allow_html=True)
        st.image(pil_img, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h3 style="color: #6366f1; margin-bottom: 1rem;">✨ LaTeX Conversion</h3>', unsafe_allow_html=True)
        
        # Processing animation
        with st.spinner('Analyzing your handwriting...'):
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            for i in range(101):
                time.sleep(0.02)
                progress_bar.progress(i)
                progress_text.markdown(f'<p style="color: #64748b;">Processing: {i}%</p>', unsafe_allow_html=True)
                if i == 50:
                    latex_predicted = predict_latex(img_rgb)
                if i == 90:
                    latex_fixed = fix_latex(latex_predicted)
            
            progress_text.empty()
            progress_bar.empty()
        
        # Display rendered LaTeX prominently
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.markdown('<h4 style="color: #6366f1; margin-bottom: 0.5rem;">🔮 Rendered Output</h4>', unsafe_allow_html=True)
        st.markdown('<div class="latex-rendered">', unsafe_allow_html=True)
        st.latex(latex_fixed)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced copy button
        if st.button("📋 Copy Rendered LaTeX", key="copy", type="primary"):
            pyperclip.copy(latex_fixed)
            st.toast("LaTeX code copied to clipboard!", icon="✅")
        
        # Raw results in expander
        with st.expander("🔍 View conversion details"):
            st.markdown('<h4 style="color: #6366f1;">Raw Prediction</h4>', unsafe_allow_html=True)
            st.code(latex_predicted, language='latex')
            
            st.markdown('<h4 style="color: #6366f1; margin-top: 1rem;">Fixed LaTeX</h4>', unsafe_allow_html=True)
            st.code(latex_fixed, language='latex')
        
        st.markdown('</div>', unsafe_allow_html=True)

# Tips section in the sidebar
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #6366f1, #8b5cf6); 
                color: white; 
                padding: 1rem; 
                border-radius: 10px;
                margin-bottom: 1.5rem;">
        <h3 style="color: white; margin-bottom: 0.5rem;">💡 Pro Tips</h3>
        <p style="margin-bottom: 0;">Get the best results with these tips</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    - ✏️ **Write clearly** with dark ink on light background
    - 📐 **Align equations** horizontally for best recognition
    - 🔍 **Crop tightly** around the equation
    - ✨ **Try examples**:
    """)
    
    examples = [
        ("Simple equation", "E = mc^2"),
        ("Fraction", "\\frac{1}{2} + \\frac{1}{4}"),
        ("Integral", "\\int_{a}^{b} x^2 dx"),
        ("Square root", "\\sqrt{1 + x^2}"),
        ("Greek letters", "\\alpha + \\beta = \\gamma")
    ]
    
    for name, example in examples:
        if st.button(f"Try: {name}", key=f"example_{name}"):
            st.session_state.example_latex = example
            st.toast(f"Try writing: {example}", icon="✍️")

# Footer with animation
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 1.5rem; 
            background-color: #f1f5f9; border-radius: 10px;
            animation: fadeIn 1s ease-out;">
    <p style="color: #64748b; margin-bottom: 0.5rem;">
        This model can make mistakes.
    </p>
    <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0;">
        ✨ v3.0 with interactive UI and real-time preview
    </p>
</div>
""", unsafe_allow_html=True)