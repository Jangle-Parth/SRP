import os
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import google.ai.generativelanguage as glm
import torch
import numpy as np
import pickle
import io
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as albu
import base64
import gdown
import tempfile
from typing import Optional

# Environment variables
API_KEY = os.getenv('GEMINI_API_KEY')  # Set this in Render environment variables
MODEL_URL = os.getenv('MODEL_URL', 'https://drive.google.com/uc?id=1SWKVLqux46da6VPd9eQK9nLuV2J7r1BA')

# Configure Gemini
genai.configure(api_key=API_KEY)

# Create FastAPI app
app = FastAPI(title="Microscopic Image Analysis API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
global_model = None
model_loaded = False
processing_results = {}

class CPU_Unpickler(pickle.Unpickler):
    """Custom unpickler for loading PyTorch models on CPU"""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

class DummyModel:
    """Fallback model if main model fails to load"""
    def __call__(self, x):
        batch_size = x.shape[0]
        h, w = 256, 256
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Create circular mask
        cy, cx = h//2, w//2
        radius = min(cx, cy) - 10
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        mask_area = x*x + y*y <= radius*radius
        mask[mask_area] = 1
        
        # Add border noise
        noise = np.random.rand(h, w) * 0.3
        border = (x*x + y*y <= (radius+10)**2) & ~mask_area
        mask[border] = noise[border]
        
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        mask_tensor = mask_tensor.repeat(batch_size, 3, 1, 1)
        return mask_tensor

def load_model_from_gdrive(gdrive_url):
    """Download and load model from Google Drive"""
    global global_model, model_loaded
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
            temp_path = temp_file.name
        
        # Download model
        gdown.download(gdrive_url, temp_path, quiet=False)
        print("Model downloaded successfully!")
        
        # Load model
        with open(temp_path, 'rb') as file:
            global_model = CPU_Unpickler(file, encoding='latin1').load()
        
        os.unlink(temp_path)
        
        if isinstance(global_model, albu.Compose):
            print("Warning: Loaded preprocessing pipeline instead of model")
            global_model = DummyModel()
        
        model_loaded = True
        return global_model
        
    except Exception as e:
        print(f"Error in model loading: {str(e)}")
        os.unlink(temp_path)
        return None

def to_tensor(x, **kwargs):
    """Convert to tensor format"""
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Get preprocessing pipeline"""
    return albu.Compose([
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ])

def preprocess_image(img):
    """Preprocess image for model"""
    temp = img / 255.0
    preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet50', 'imagenet')
    preprocessing_transform = get_preprocessing(preprocessing_fn)
    img_processed = preprocessing_transform(image=temp)['image']
    return torch.from_numpy(img_processed).to('cpu').unsqueeze(0)

def analyze_image_with_gemini(image_bytes, additional_info=""):
    """Analyze image using Gemini Pro Vision"""
    prompt_text = """
    Analyze the microscopic image of low carbon steel dual-phase. Based on the visual features, provide the following details:
    Phases: Identify the primary and secondary phases present (e.g., ferrite, martensite, bainite, etc.).
    Heat Treatment Process: Deduce the likely heat treatment process used to produce the observed microstructure.
    Compositions: Estimate the composition, especially the carbon content and other alloying elements.
    """
    
    if additional_info:
        prompt_text += f"\n\nAdditional information request: {additional_info}"
    
    try:
        model_gemini = genai.GenerativeModel('gemini-1.5-flash')
        response = model_gemini.generate_content(
            glm.Content(
                parts=[
                    glm.Part(text=prompt_text),
                    glm.Part(inline_data=glm.Blob(mime_type='image/jpeg', data=image_bytes)),
                ]
            )
        )
        return response.text
    except Exception as e:
        return f"Error in Gemini analysis: {str(e)}"

def generate_segmentation_mask(image_array):
    """Generate segmentation mask from image"""
    global global_model
    
    try:
        # Resize to 256x256
        pil_image = Image.fromarray(image_array)
        resized_image = pil_image.resize((256, 256)).convert("RGB")
        resized_array = np.array(resized_image)
        
        if isinstance(global_model, albu.Compose):
            # Apply preprocessing only
            preprocessed = global_model(image=resized_array)['image']
            processed_image = np.clip(preprocessed.transpose(1, 2, 0), 0, 1)
            overlay = np.ones_like(processed_image) * np.array([0.2, 0.5, 0.2])
            segmented_image = processed_image * 0.7 + overlay * 0.3
        else:
            # Full model processing
            preprocessed_tensor = preprocess_image(resized_array)
            
            with torch.no_grad():
                if hasattr(global_model, 'predict'):
                    segmented_tensor = global_model.predict(preprocessed_tensor)
                else:
                    segmented_tensor = global_model(preprocessed_tensor)
                
                if isinstance(segmented_tensor, torch.Tensor):
                    if len(segmented_tensor.shape) == 4:
                        segmented_image = segmented_tensor[0].cpu().numpy().transpose(1, 2, 0)
                    else:
                        segmented_image = segmented_tensor.cpu().numpy().transpose(1, 2, 0)
                else:
                    segmented_image = np.ones((256, 256, 3)) * 0.5
        
        # Convert to base64
        pil_segmented = Image.fromarray((segmented_image * 255).astype(np.uint8))
        buffered = io.BytesIO()
        pil_segmented.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return img_str, None
        
    except Exception as e:
        print(f"Segmentation error: {str(e)}")
        # Generate placeholder on error
        placeholder = np.ones((256, 256, 3), dtype=np.uint8) * 200
        pil_img = Image.fromarray(placeholder)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str, str(e)

async def process_image_task(task_id: str, image_bytes: bytes, additional_info: str):
    """Background task for image processing"""
    global processing_results
    
    try:
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        # Generate Gemini analysis
        gemini_description = analyze_image_with_gemini(image_bytes, additional_info)
        
        # Generate segmentation mask
        mask_base64, error = generate_segmentation_mask(image_array)
        
        # Store results
        processing_results[task_id] = {
            "status": "completed",
            "gemini_analysis": gemini_description,
            "segmentation_mask": mask_base64
        }
        
        if error:
            processing_results[task_id]["segmentation_warning"] = f"Warning: {error}"
            
    except Exception as e:
        processing_results[task_id] = {
            "status": "error",
            "message": f"Processing error: {str(e)}"
        }

@app.post("/process_image/")
async def process_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    additional_info: Optional[str] = Form("")
):
    """Endpoint for processing uploaded images"""
    global model_loaded
    
    if not model_loaded:
        return {
            "status": "error", 
            "message": "Model is still loading. Please try again later."
        }
    
    task_id = f"task_{len(processing_results) + 1}"
    image_bytes = await file.read()
    
    background_tasks.add_task(process_image_task, task_id, image_bytes, additional_info)
    
    return {
        "status": "processing",
        "task_id": task_id,
        "message": "Image processing started"
    }

@app.get("/get_result/{task_id}")
async def get_result(task_id: str):
    """Endpoint to get processing results"""
    if task_id not in processing_results:
        return {"status": "not_found", "message": "Task ID not found"}
    return processing_results[task_id]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "gemini_configured": API_KEY is not None
    }

# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global global_model, model_loaded
    try:
        model = load_model_from_gdrive(MODEL_URL)
        if not model_loaded:
            print("Using dummy model as fallback")
            global_model = DummyModel()
        model_loaded = True
    except Exception as e:
        print(f"Error loading model: {e}")
        global_model = DummyModel()
        model_loaded = True

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
