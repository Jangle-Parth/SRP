# app.py - Flask API for MongoDB storage and image processing
from flask import Flask, request, jsonify, send_file
import os
import io
import uuid
from datetime import datetime
from pymongo import MongoClient
from bson.objectid import ObjectId
import gridfs
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import pickle
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as albu
import tempfile
import base64

app = Flask(__name__)

# MongoDB setup
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb+srv://username:password@cluster.mongodb.net/steel_analysis')
client = MongoClient(MONGO_URI)
db = client.steel_analysis
fs = gridfs.GridFS(db)

# Google Drive setup
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = 'service-account.json'
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

# Model pickle file IDs in Google Drive
MODEL_FILE_ID = 'your-google-drive-file-id-for-model'
MODEL = None  # Will be loaded on first request

# Image preprocessing functions
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

def load_model_from_drive():
    """Load the model pickle file from Google Drive"""
    global MODEL
    
    if MODEL is not None:
        return MODEL
    
    try:
        request = drive_service.files().get_media(fileId=MODEL_FILE_ID)
        file = io.BytesIO()
        downloader = MediaIoBaseDownload(file, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        
        file.seek(0)
        
        # Custom unpickler to handle GPU/CPU device differences
        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                else: 
                    return super().find_class(module, name)
        
        MODEL = CPU_Unpickler(file, encoding='latin1').load()
        return MODEL
    except Exception as e:
        app.logger.error(f"Failed to load model from Google Drive: {e}")
        return None

@app.route('/upload', methods=['POST'])
def upload_image():
    """Upload an image to MongoDB GridFS"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Store image in MongoDB
        file_data = file.read()
        file_id = fs.put(
            file_data, 
            filename=file.filename,
            metadata={
                "upload_time": datetime.utcnow(),
                "processed": False
            }
        )
        
        return jsonify({
            "status": "success",
            "image_id": str(file_id)
        }), 200
    except Exception as e:
        app.logger.error(f"Upload failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/process', methods=['POST'])
def process_image():
    """Process an image using the MicroNet model"""
    data = request.json
    if not data or 'image_id' not in data:
        return jsonify({"error": "Missing image_id in request"}), 400
    
    image_id = data['image_id']
    
    try:
        # Retrieve image from MongoDB
        file_obj = fs.get(ObjectId(image_id))
        if not file_obj:
            return jsonify({"error": "Image not found"}), 404
        
        # Load model if not already loaded
        model = load_model_from_drive()
        if model is None:
            return jsonify({"error": "Failed to load model"}), 500
        
        # Process the image
        img_data = file_obj.read()
        image = Image.open(io.BytesIO(img_data))
        resized_image = image.resize((256, 256)).convert("RGB")
        np_image = np.array(resized_image)
        
        # Generate segmented image
        with torch.no_grad():
            segmented_tensor = model.predict(preprocess(np_image))
        segmented_np = segmented_tensor[0].cpu().numpy().transpose(1, 2, 0)
        
        # Convert segmented image to PIL and save
        segmented_image = Image.fromarray((segmented_np * 255).astype(np.uint8))
        
        # Save processed image back to MongoDB
        img_byte_arr = io.BytesIO()
        segmented_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        processed_file_id = fs.put(
            img_byte_arr.getvalue(),
            filename=f"processed_{file_obj.filename}",
            metadata={
                "original_image_id": image_id,
                "process_time": datetime.utcnow()
            }
        )
        
        # Update original image metadata
        db.fs.files.update_one(
            {"_id": ObjectId(image_id)},
            {"$set": {"metadata.processed": True, "metadata.processed_id": processed_file_id}}
        )
        
        # Return the processed image URL (this could be a direct download endpoint)
        processed_image_url = f"{request.host_url.rstrip('/')}/download/{processed_file_id}"
        
        return jsonify({
            "status": "success",
            "processed_image_id": str(processed_file_id),
            "processed_image_url": processed_image_url
        }), 200
    except Exception as e:
        app.logger.error(f"Processing failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/download/<file_id>', methods=['GET'])
def download_file(file_id):
    """Download a file from MongoDB GridFS"""
    try:
        file_obj = fs.get(ObjectId(file_id))
        if not file_obj:
            return jsonify({"error": "File not found"}), 404
            
        return send_file(
            io.BytesIO(file_obj.read()),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=file_obj.filename
        )
    except Exception as e:
        app.logger.error(f"Download failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
