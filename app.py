import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = '/tmp/uploads'
RESULT_FOLDER = '/tmp/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load pre-trained TensorFlow object detection model
# Note: You'll need to download a specific model from TensorFlow Model Zoo
MODEL_PATH = 'path/to/your/tensorflow/model'
detect_fn = tf.saved_model.load(MODEL_PATH)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    """Perform object detection on the image"""
    try:
        # Read image
        image = cv2.imread(image_path)
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
        output_path = os.path.join(RESULT_FOLDER, 'processed_image.jpg')
        cv2.imwrite(output_path, image)
        return output_path

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and object detection"""
    if 'file' not in request.files:
        return {'error': 'No file uploaded'}, 400

    file = request.files['file']

    if file.filename == '':
        return {'error': 'No selected file'}, 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Process image
        processed_image_path = process_image(filepath)

        if processed_image_path:
            return send_file(processed_image_path, mimetype='image/jpeg')
        else:
            return {'error': 'Image processing failed'}, 500

    return {'error': 'File type not allowed'}, 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)