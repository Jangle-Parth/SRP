import os
import io
import cv2
import numpy as np
import pickle
import torch
import segmentation_models_pytorch as smp
import albumentations as albu
import google.generativeai as genai
import google.ai.generativelanguage as glm
from datetime import datetime
from pathlib import Path
from PIL import Image
import base64

from flask import Flask, request, send_file, jsonify, render_template_string
from werkzeug.utils import secure_filename
import streamlit as st

# API configuration
app = Flask(__name__)
UPLOAD_FOLDER = Path('/tmp/uploads')
RESULT_FOLDER = Path('/tmp/results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create directories if they don't exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

# Set up API key for Gemini-Pro
API_KEY = 'AIzaSyAY4YSt-7GLHGF8NUdbgkfxum-s5Rrh5Xs'
genai.configure(api_key=API_KEY)

# Load MicroNet model from pickle
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def load_model():
    """Load the model from the pickle file"""
    pickle_path = "micronet_model_steel_segmentation.pkl"
    try:
        with open(pickle_path, 'rb') as file:
            model = CPU_Unpickler(file, encoding='latin1').load()
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

# Global model instance
MODEL = load_model()

# Model preprocessing functions
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    return albu.Compose([
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ])

def preprocess(img):
    preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet50', 'imagenet')
    preprocessing_transform = get_preprocessing(preprocessing_fn)
    temp = img / 255.0
    img = preprocessing_transform(image=temp)
    return torch.from_numpy(img['image']).to('cpu').unsqueeze(0)

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path: Path) -> Path | None:
    """
    Process the image with MicroNet model
    
    Args:
        image_path: Path to input image
        
    Returns:
        Path to processed image or None if processing fails
    """
    try:
        # Check if model is loaded
        if MODEL is None:
            raise ValueError("Model not loaded")
        
        # Read image
        image = Image.open(image_path).resize((256, 256)).convert("RGB")
        img_array = np.array(image)
        
        # Generate segmented image
        segmented_tensor = MODEL.predict(preprocess(img_array))
        segmented_image = segmented_tensor[0].cpu().numpy().transpose(1, 2, 0)
        
        # Save processed image
        output_path = RESULT_FOLDER / f'processed_{image_path.name}'
        cv2.imwrite(str(output_path), (segmented_image * 255).astype(np.uint8))
        
        # Also perform basic OpenCV processing for visualization
        cv_image = cv2.imread(str(image_path))
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cv_image, contours, -1, (0, 255, 0), 2)
        cv2.putText(cv_image, f'Objects detected: {len(contours)}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                   
        # Combine segmentation and contour detection
        combined = cv2.addWeighted(
            cv_image, 0.7, 
            (segmented_image * 255).astype(np.uint8), 0.3, 
            0
        )
        
        # Save combined result
        combined_path = RESULT_FOLDER / f'combined_{image_path.name}'
        cv2.imwrite(str(combined_path), combined)
        
        return combined_path
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def analyze_with_gemini(image_path: Path) -> str | None:
    """
    Analyze image with Gemini-Pro API
    
    Args:
        image_path: Path to input image
        
    Returns:
        Analysis text or None if analysis fails
    """
    try:
        with open(image_path, 'rb') as img:
            bytes_data = img.read()
            
        # Prompt for Gemini
        prompt_text = """
            Analyze the microscopic image of low carbon steel dual-phase. Based on the visual features, provide the following details:
            Phases: Identify the primary and secondary phases present (e.g., ferrite, martensite, bainite, etc.).
            Heat Treatment Process: Deduce the likely heat treatment process used to produce the observed microstructure.
            Compositions: Estimate the composition, especially the carbon content and other alloying elements.
        """
        
        # LLM call with the updated prompt
        model_gemini = genai.GenerativeModel('gemini-1.5-flash')
        response = model_gemini.generate_content(
            glm.Content(
                parts=[
                    glm.Part(text=prompt_text),
                    glm.Part(inline_data=glm.Blob(mime_type='image/jpeg', data=bytes_data)),
                ]
            )
        )
        
        return response.text
    except Exception as e:
        print(f"Failed to analyze with Gemini: {e}")
        return None

# API Routes
@app.route('/', methods=['GET'])
def home():
    """Home page with usage instructions"""
    return jsonify({
        'status': 'running',
        'usage': {
            'upload_endpoint': '/upload',
            'method': 'POST',
            'content_type': 'multipart/form-data',
            'parameter': 'file'
        },
        'supported_formats': list(ALLOWED_EXTENSIONS)
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'upload_folder': str(UPLOAD_FOLDER),
        'result_folder': str(RESULT_FOLDER)
    })

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))
        
        # Process image
        processed_image_path = process_image(filepath)
        if processed_image_path is None:
            return jsonify({'error': 'Image processing failed'}), 500
        
        # Return processed image
        return send_file(str(processed_image_path), mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    finally:
        # Cleanup uploaded file
        if filepath.exists():
            filepath.unlink()

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Analyze image with Gemini and return results"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))
        
        # Process image
        processed_image_path = process_image(filepath)
        if processed_image_path is None:
            return jsonify({'error': 'Image processing failed'}), 500
        
        # Analyze with Gemini
        analysis = analyze_with_gemini(filepath)
        if analysis is None:
            return jsonify({'error': 'Image analysis failed'}), 500
        
        # Return results
        with open(processed_image_path, 'rb') as img:
            processed_image_bytes = img.read()
            encoded_image = base64.b64encode(processed_image_bytes).decode('utf-8')
        
        return jsonify({
            'analysis': analysis,
            'processed_image': encoded_image
        })
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    finally:
        # Cleanup uploaded file
        if filepath.exists():
            filepath.unlink()

# Streamlit web interface
def streamlit_app():
    """Streamlit web interface for the API"""
    st.set_page_config(page_title="Microscopic Image Analysis", layout="wide")
    st.title("Microscopic Steel Image Analysis")
    
    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Layout with two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose a Microscopic Image file", type=list(ALLOWED_EXTENSIONS))
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Additional user input
            additional_info = st.text_input("Ask for additional information related to the microscopic steel image (optional):")
            
            # Process buttons
            col1a, col1b = st.columns(2)
            with col1a:
                if st.button("Generate Analysis"):
                    # Save temporary file
                    temp_path = UPLOAD_FOLDER / f"streamlit_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Analyze with Gemini
                    with st.spinner("Analyzing with Gemini-Pro..."):
                        prompt = additional_info if additional_info else ""
                        analysis = analyze_with_gemini(temp_path)
                        
                        if analysis:
                            st.session_state.chat_history.append({"role": "system", "content": analysis})
                            st.success("Analysis generated!")
                        else:
                            st.error("Failed to generate analysis")
                    
                    # Clean up
                    if temp_path.exists():
                        temp_path.unlink()
            
            with col1b:
                if st.button("Generate Mask Image"):
                    # Save temporary file
                    temp_path = UPLOAD_FOLDER / f"streamlit_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Process image
                    with st.spinner("Generating segmentation mask..."):
                        processed_path = process_image(temp_path)
                        
                        if processed_path and processed_path.exists():
                            processed_image = Image.open(processed_path)
                            st.session_state.processed_image = processed_image
                            st.success("Mask generated!")
                        else:
                            st.error("Failed to generate mask")
                    
                    # Clean up
                    if temp_path.exists():
                        temp_path.unlink()
    
    with col2:
        st.header("Analysis Results")
        
        # Display chat history
        st.subheader("Gemini Analysis")
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Analysis:** {message['content']}")
        
        # Display processed image if available
        if "processed_image" in st.session_state:
            st.subheader("Segmented Image")
            st.image(st.session_state.processed_image, caption="Segmented Image", use_column_width=True)
            
            # Convert to downloadable format
            buffered = io.BytesIO()
            st.session_state.processed_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            href = f'<a href="data:file/png;base64,{img_str}" download="segmented_image.png">Download Segmented Image</a>'
            st.markdown(href, unsafe_allow_html=True)

# Run API or Streamlit app based on environment
if __name__ == '__main__':
    if os.environ.get('STREAMLIT_APP', 'false').lower() == 'true':
        # Run Streamlit app
        streamlit_app()
    else:
        # Run Flask API
        port = int(os.environ.get('PORT', 10000))
        app.run(host='0.0.0.0', port=port, debug=False)
