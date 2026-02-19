# Import necessary libraries for the web app, deep learning, image processing, and encoding.
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# Configures the default settings for the Streamlit web application.
# Sets the browser tab title, favicon, and expands the layout to fill the screen width.
st.set_page_config(page_title="Plant Disease Scanner", page_icon="🌿", layout="wide")

# Injects raw CSS to override Streamlit's default visual theme.
# This hides default headers/footers, styles the main container as a floating card,
# and applies custom colors, padding, and animations to the file uploader and sidebar.
def local_css():
    st.markdown("""
        <style>
            [data-testid="stAppViewContainer"] { background-color: #f3f4f6 !important; }
            [data-testid="stHeader"] { display: none !important; }
            
            /* Main Content Card Styling */
            section[data-testid="stMain"] .block-container { 
                background-color: #ffffff !important; 
                border-radius: 12px !important; 
                padding: 3rem !important; 
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important; 
                border: 1px solid #e5e7eb !important; 
                max-width: 950px !important; 
                margin: 5rem auto !important; /* Switched to rem for stability in hosted iframes */
            }
            section[data-testid="stMain"] { padding-top: 0 !important; }
            #MainMenu, footer { visibility: hidden; }
            
            /* Sidebar Styling */
            [data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e5e7eb !important; }
            [data-testid="stSidebarHeader"] { padding-top: 1rem !important; padding-bottom: 0rem !important; min-height: 0 !important; display: none !important; }
            [data-testid="stSidebarNav"] { display: none !important; height: 0 !important; }
            section[data-testid="stSidebar"] .block-container { padding-top: 2rem !important; }

            /* Toggle Button Styling */
            [data-testid="collapsedControl"] { background-color: #10b981 !important; color: white !important; border-radius: 8px !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2) !important; opacity: 1 !important; top: 1.5rem !important; left: 1.5rem !important; transition: background-color 0.2s ease !important; z-index: 9999 !important; }
            [data-testid="collapsedControl"]:hover { background-color: #059669 !important; }
            [data-testid="collapsedControl"] svg { fill: #ffffff !important; color: #ffffff !important; }

            /* Sidebar HTML Elements Styling */
            .brand { display: flex; align-items: center; gap: 0.75rem; font-size: 1.5rem; font-weight: 700; color: #059669; margin-top: 0rem; margin-bottom: 2rem; }
            .info-section { margin-bottom: 2rem; }
            .info-section h3 { font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; color: #6b7280; margin-bottom: 1rem; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem; }
            .info-content { font-size: 0.9rem; line-height: 1.6; color: #1f2937; margin-bottom: 1.5rem;}
            .tag { background: #ecfdf5; color: #059669; padding: 0.25rem 0.75rem; border-radius: 99px; font-size: 0.75rem; font-weight: 600; border: 1px solid #d1fae5; display: inline-block; margin: 0.25rem 0.25rem 0.25rem 0; }
            .accuracy-badge { display: inline-block; background: #1f2937; color: white; padding: 0.25rem 0.75rem; border-radius: 6px; font-weight: 600; font-size: 0.85rem; }

            /* Header HTML Elements Styling */
            .header-container { text-align: center; margin-bottom: 2.5rem; }
            .header-container h1 { font-size: 2.2rem; font-weight: 700; color: #1f2937; margin-top: 0; margin-bottom: 0.5rem; padding: 0; }
            .header-container p { color: #6b7280; font-size: 1rem; margin: 0; }

            /* Uploader Drag-and-Drop Override */
            /* Using pure CSS hover instead of JS to prevent Streamlit Cloud iframe CORS errors */
            [data-testid="stFileUploadDropzone"] { border: 2px dashed #e5e7eb !important; border-radius: 12px !important; padding: 5rem 2rem !important; background-color: #f9fafb !important; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); display: flex !important; flex-direction: column !important; align-items: center !important; justify-content: center !important; position: relative; z-index: 1; }
            [data-testid="stFileUploadDropzone"]:hover { border-color: #10b981 !important; background-color: #ecfdf5 !important; transform: scale(1.03) !important; z-index: 9999 !important; box-shadow: 0 0 0 20000px rgba(0, 0, 0, 0.5), 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04) !important; }
            [data-testid="stFileUploadDropzone"] * { color: #1f2937 !important; text-align: center !important; }
            [data-testid="stFileUploadDropzone"] svg { fill: #6b7280 !important; color: #6b7280 !important; }
            [data-testid="stFileUploadDropzone"] button { background-color: #ffffff !important; color: #1f2937 !important; border: 1px solid #d1d5db !important; font-weight: 500 !important; }
            [data-testid="stFileUploadDropzone"] button:hover { background-color: #f3f4f6 !important; }

            /* Results Section Styling */
            .image-preview-box { border-radius: 12px; overflow: hidden; background: #000; height: 320px; display: flex; align-items: center; justify-content: center; border: 1px solid #e5e7eb; }
            .image-preview-box img { max-width: 100%; max-height: 100%; object-fit: contain; }
            
            .status-card { padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 1rem; }
            .status-card.healthy { background: #ecfdf5; border: 1px solid #a7f3d0; color: #065f46; }
            .status-card.diseased { background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; }
            
            .prediction-item { margin-bottom: 1rem; }
            .pred-header { display: flex; justify-content: space-between; font-size: 0.9rem; font-weight: 600; color: #1f2937; margin-bottom: 0.25rem; }
            .progress-bg { height: 8px; background: #e5e7eb; border-radius: 4px; overflow: hidden; }
            .progress-fill { height: 100%; transition: width 0.8s ease-in-out; }
            
            div[data-testid="stButton"] button { width: 100%; background-color: #ffffff !important; border: 1px solid #e5e7eb !important; padding: 0.75rem 1.5rem !important; border-radius: 12px !important; color: #1f2937 !important; font-weight: 600 !important; margin-top: 1rem; }
            div[data-testid="stButton"] button:hover { background-color: #f3f4f6 !important; border-color: #d1d5db !important; }
        </style>
    """, unsafe_allow_html=True)

# Executes the CSS injection function to style the page right away.
local_css()

# Loads the TensorFlow Lite model and the labels text file exactly once.
@st.cache_resource
def load_model_and_labels():
    interpreter = tf.lite.Interpreter(model_path="model96.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return interpreter, input_details, output_details, labels

# Resizes and formats the uploaded image for TensorFlow's predictions.
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img.convert('RGB')).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Converts a Pillow image object into a raw Base64 text string for UI rendering.
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Initialize the model using the cached helper function.
interpreter, input_details, output_details, labels = load_model_and_labels()

# Renders the sidebar of the web application.
# ALL HTML is flattened into a strict single line to bypass Streamlit Markdown block rendering bugs.
with st.sidebar:
    plants = ["Apple", "Blueberry", "Cherry", "Corn", "Grape", "Orange", "Peach", "Pepper", "Potato", "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato"]
    tags_html = "".join([f'<span class="tag">{p}</span>' for p in plants])
    sidebar_html = f'<div class="brand">🌿 Disease Scanner</div><div class="info-section"><h3>How to Use</h3><div class="info-content">1. Click the upload box or drag an image of a plant leaf.<br>2. Wait for the AI to analyze the image.<br>3. View the diagnosis and confidence score.</div></div><div class="info-section"><h3>Supported Plants</h3><div style="margin-bottom: 2rem;">{tags_html}</div></div><div class="info-section"><h3>Model Accuracy</h3><div class="info-content" style="margin-bottom: 0.5rem;">Current model performance on validation set:</div><span class="accuracy-badge">96.3% Accuracy</span></div>'
    st.markdown(sidebar_html, unsafe_allow_html=True)

# Renders the main title and subtitle of the application (flattened).
st.markdown('<div class="header-container"><h1>Plant Disease Classifier</h1><p>Upload a leaf image to detect diseases instantly using AI.</p></div>', unsafe_allow_html=True)

# Creates a persistent session state variable to track the uploader widget.
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# Creates the file uploader widget.
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key=f"uploader_{st.session_state.uploader_key}")

# If the user successfully uploaded a file, this block executes.
if uploaded_file is not None:
    
    # Visually hide the Streamlit uploader entirely so our custom results take its place.
    st.markdown('<style>[data-testid="stFileUploader"] { display: none !important; }</style>', unsafe_allow_html=True)
    
    # Read the image and prepare its base64 string for the UI preview.
    image = Image.open(uploaded_file)
    img_b64 = image_to_base64(image)
    
    # Display a loading spinner while the AI runs its calculation.
    with st.spinner("Analyzing..."):
        input_data = preprocess_image(image)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Extract the predictions from the model and find the top 3 highest confidence scores.
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        top_indices = output_data.argsort()[-3:][::-1]
        
        # Determine the primary diagnosis and check if the plant is considered healthy.
        top_pred_idx = top_indices[0]
        result_label = labels[top_pred_idx].replace("___", " ").replace("_", " ")
        is_healthy = "healthy" in result_label.lower()
    
    # Split the main area into two side-by-side columns: Image on left, Results on right.
    col1, col2 = st.columns(2, gap="large")

    # Display the uploaded image inside the left column.
    with col1:
        st.markdown(f'<div class="image-preview-box"><img src="data:image/png;base64,{img_b64}" alt="Plant Leaf"></div>', unsafe_allow_html=True)

    # Build and display the detailed results inside the right column.
    with col2:
        # Generates a green success card if healthy, or a red warning card if diseased.
        icon, color_class, diag_text = ("✅", "healthy", "Plant is Healthy") if is_healthy else ("⚠️", "diseased", result_label)
        
        # HTML strings are flattened to a single line so Streamlit DOES NOT parse them as Markdown Code blocks.
        status_html = f'<div class="status-card {color_class}"><span style="font-size: 24px;">{icon}</span><div><div style="font-size:0.8rem; opacity:0.8">Diagnosis</div><div style="font-weight:700; font-size:1.1rem;">{diag_text}</div></div></div>'
        
        # Generates the horizontal progress bars for the top 3 predictions.
        bars_html = ""
        for i, idx in enumerate(top_indices):
            conf = output_data[idx] * 100
            name = labels[idx].replace("___", " ").replace("_", " ")
            bar_color = "#ef4444" if i == 0 and not is_healthy else "#10b981"
            
            bars_html += f'<div class="prediction-item"><div class="pred-header"><span>{name}</span><span>{conf:.1f}%</span></div><div class="progress-bg"><div class="progress-fill" style="width: {conf}%; background-color: {bar_color};"></div></div></div>'
            
        # Inject the fully constructed status card and progress bars into the UI.
        st.markdown(status_html + bars_html, unsafe_allow_html=True)
        
        # Creates a button to reset the app so the user can analyze another photo.
        if st.button("Analyze Another Image"):
            st.session_state.uploader_key += 1
            
            # Support both new and legacy versions of Streamlit for page rerunning.
            if hasattr(st, 'rerun'):
                st.rerun()
            else:
                st.experimental_rerun()