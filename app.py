import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
from pathlib import Path

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = Path('/tmp/uploads')
RESULT_FOLDER = Path('/tmp/results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = os.getenv('MODEL_PATH', 'path/to/your/tensorflow/model')

# Create directories if they don't exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

# Load model at startup
try:
    detect_fn = tf.saved_model.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    detect_fn = None

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path: Path) -> Path | None:
    """
    Perform object detection on the image
    
    Args:
        image_path: Path to input image
        
    Returns:
        Path to processed image or None if processing fails
    """
    try:
        if detect_fn is None:
            raise ValueError("Model not loaded")

        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Failed to read image")

        # Convert image to tensor
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Perform detection
        detections = detect_fn(input_tensor)

        # Process detection results
        num_detections = int(detections['num_detections'])
        classes = detections['detection_classes'][0].numpy().astype(np.int32)
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()

        # Draw bounding boxes for high confidence detections
        for i in range(num_detections):
            if scores[i] > 0.5:  # Confidence threshold
                ymin, xmin, ymax, xmax = boxes[i]
                im_height, im_width, _ = image.shape
                left = int(xmin * im_width)
                right = int(xmax * im_width)
                top = int(ymin * im_height)
                bottom = int(ymax * im_height)

                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                label = f'Class: {classes[i]}, Score: {scores[i]:.2f}'
                cv2.putText(image, label, (left, top-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save processed image
        output_path = RESULT_FOLDER / f'processed_{image_path.name}'
        cv2.imwrite(str(output_path), image)
        return output_path

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detect_fn is not None
    })

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and object detection"""
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
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)