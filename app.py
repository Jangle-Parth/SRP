import os
import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
from pathlib import Path

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = Path('/tmp/uploads')
RESULT_FOLDER = Path('/tmp/results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create directories if they don't exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path: Path) -> Path | None:
    """
    Process the image with basic OpenCV operations
    
    Args:
        image_path: Path to input image
        
    Returns:
        Path to processed image or None if processing fails
    """
    try:
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Failed to read image")

        # Basic image processing
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on original image
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
        
        # Add some text
        cv2.putText(image, f'Objects detected: {len(contours)}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)

        # Save processed image
        output_path = RESULT_FOLDER / f'processed_{image_path.name}'
        cv2.imwrite(str(output_path), image)
        return output_path

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

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

        return send_file(str(processed_image_path), mimetype='image/jpeg')

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

    finally:
        # Cleanup uploaded file
        if filepath.exists():
            filepath.unlink()

if __name__ == '__main__':
    port = int(os.getenv('PORT', 10000))
    app.run(host='0.0.0.0', port=port)