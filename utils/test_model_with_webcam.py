import tensorflow as tf
import cv2
import numpy as np
import time
import os
import glob
from pathlib import Path

# Directory containing raw images
RAW_IMAGES_DIR = "./pictures"
OUTPUT_DIR = "detection_results"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the model
interpreter = tf.lite.Interpreter(model_path='swift_yolo_32.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Model expects input shape: {input_details[0]['shape']}")
print(f"Model expects input type: {input_details[0]['dtype']}")
print("Forcing INT8 input type to match ESP32 hardware behavior")

# Constants
MODEL_INPUT_SIZE = (192, 192)
DETECTION_THRESHOLD = 10  # Changed from 0.45 to 45 since scores are 0-100
FACE_CLASS_ID = 100
NMS_THRESHOLD = 0.3  # Non-maximum suppression threshold

# Global counter for debug output
frame_count = 0

def list_files(directory):
    """List all files in the directory"""
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def load_raw_image(file_path, width, height, format_type):
    """Load and decode a raw image file (RGB565 or RGB888)"""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    
    # Determine format based on user selection
    if format_type == 'rgb565':
        # RGB565 format (2 bytes per pixel)
        if len(raw_data) != width * height * 2:
            raise ValueError(f"Invalid RGB565 file size: expected {width * height * 2} bytes, got {len(raw_data)}")
        
        # Convert RGB565 to RGB888
        rgb888 = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(width * height):
            # Get RGB565 pixel (2 bytes)
            pixel = (raw_data[i * 2] << 8) | raw_data[i * 2 + 1]
            
            # Extract RGB components
            r = ((pixel >> 11) & 0x1F) << 3  # 5 bits to 8 bits
            g = ((pixel >> 5) & 0x3F) << 2   # 6 bits to 8 bits
            b = (pixel & 0x1F) << 3          # 5 bits to 8 bits
            
            # Store RGB888 pixel
            y = i // width
            x = i % width
            rgb888[y, x] = [r, g, b]
        
        return rgb888
    
    elif format_type == 'rgb888':
        # RGB888 format (3 bytes per pixel)
        if len(raw_data) != width * height * 3:
            raise ValueError(f"Invalid RGB888 file size: expected {width * height * 3} bytes, got {len(raw_data)}")
        
        # Reshape to RGB888 format
        rgb888 = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
        return rgb888
    
    else:
        raise ValueError(f"Unsupported file format: {format_type}")

def esp32_style_resize(frame, dest_w=192, dest_h=192):
    """
    Simple resize to match model input size
    """
    # Get source dimensions
    src_h, src_w = frame.shape[:2]
    output = np.zeros((dest_h, dest_w, 3), dtype=np.uint8)

    for y in range(dest_h):
        src_y = (y * src_h) // dest_h
        for x in range(dest_w):
            src_x = (x * src_w) // dest_w
            output[y, x] = frame[src_y, src_x]  # No channel flip

    return output

def preprocess_image(frame):
    """
    Process the image to match model input requirements
    """
    global frame_count
    
    # Resize and convert to RGB
    rgb = esp32_style_resize(frame, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1])
    
    # Convert to float and normalize to [0,1] range
    normalized = rgb.astype(np.float32) / 255.0
    
    # Check if the model is quantized (INT8)
    if input_details[0]['dtype'] == np.int8:
        # Get quantization parameters
        scale = input_details[0]['quantization'][0]
        zero_point = input_details[0]['quantization'][1]
        
        # Debug print first few values
        if frame_count % 30 == 0:  # Print every 30 frames
            print("\nPreprocessing debug:")
            print(f"First few normalized values: {normalized.flatten()[:5]}")
            print(f"Using quantization: scale={scale}, zero_point={zero_point}")
        
        # Quantize to INT8
        quantized = np.round(normalized / scale + zero_point).astype(np.int8)
        
        # Add batch dimension
        batched = np.expand_dims(quantized, axis=0)
        frame_count += 1
        return batched
    else:
        # For float model, just normalize
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        frame_count += 1
        return batched

def calculate_iou(box1, box2):
    """Calculate intersection over union between two boxes"""
    # Convert from center format to corner format
    box1_x1 = box1[0] - box1[2]/2
    box1_y1 = box1[1] - box1[3]/2
    box1_x2 = box1[0] + box1[2]/2
    box1_y2 = box1[1] + box1[3]/2
    
    box2_x1 = box2[0] - box2[2]/2
    box2_y1 = box2[1] - box2[3]/2
    box2_x2 = box2[0] + box2[2]/2
    box2_y2 = box2[1] + box2[3]/2
    
    # Calculate intersection area
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    # Calculate union area
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area
    
    # Calculate IoU
    iou = inter_area / union_area if union_area > 0 else 0
    
    return iou

def is_valid_face(x, y, w, h):
    """Additional filtering for face proportions"""
    # Check if box has reasonable aspect ratio for a face (not too wide or tall)
    aspect_ratio = w / h if h > 0 else 0
    valid_ratio = 0.3 < aspect_ratio < 2.0  # Relaxed values to match ESP32
    
    # Check if box is not too small
    min_size = MODEL_INPUT_SIZE[0] * 0.03  # Matching ESP32's MIN_SIZE_FRACTION
    valid_size = w > min_size and h > min_size
    
    # Check if box is not too large (over 90% of the image)
    max_area = 0.9 * MODEL_INPUT_SIZE[0] * MODEL_INPUT_SIZE[1]  # Matching ESP32's MAX_SIZE_FRACTION
    box_area = w * h
    valid_area = box_area < max_area
    
    # Print validation details if any check fails
    if not (valid_ratio and valid_size and valid_area):
        print(f"  Validation failed: ratio={'OK' if valid_ratio else 'BAD'} ({aspect_ratio:.2f}), "
              f"size={'OK' if valid_size else 'TOO_SMALL'} (w={w:.2f}, h={h:.2f}), "
              f"area={'OK' if valid_area else 'TOO_LARGE'} ({box_area:.2f})")
    
    return valid_ratio and valid_size and valid_area

def apply_nms(detections, nms_threshold):
    """Apply Non-Maximum Suppression to filter overlapping detections"""
    if len(detections) == 0:
        return []
    
    # Sort by confidence (highest first)
    sorted_detections = sorted(detections, key=lambda x: x[4], reverse=True)
    
    # Initialize list of kept detections
    kept_detections = []
    
    while len(sorted_detections) > 0:
        # Take the detection with highest confidence
        best_detection = sorted_detections.pop(0)
        kept_detections.append(best_detection)
        
        # Filter out detections with high IoU overlap with the best detection
        filtered_detections = []
        for detection in sorted_detections:
            iou = calculate_iou(best_detection[:4], detection[:4])
            if iou <= nms_threshold:
                filtered_detections.append(detection)
        
        sorted_detections = filtered_detections
    
    return kept_detections

def draw_detections(frame, detections):
    h, w = frame.shape[:2]
    scale_x, scale_y = w / MODEL_INPUT_SIZE[0], h / MODEL_INPUT_SIZE[1]
    
    for detection in detections:
        # Get bounding box coordinates
        x, y, width, height = detection[:4]
        
        # Convert coordinates to pixels in original frame
        x1 = int((x - width/2) * scale_x)
        y1 = int((y - height/2) * scale_y)
        x2 = int((x + width/2) * scale_x)
        y2 = int((y + height/2) * scale_y)
        
        # Ensure coordinates are within frame
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add confidence text
        confidence = detection[4]
        label = f"Face: {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def process_file(file_path, width, height):
    """Process a single file and save the result"""
    try:
        print(f"Processing: {file_path}")
        
        # Detect format from filename
        filename = os.path.basename(file_path).lower()
        if filename.startswith('rgb888_'):
            format_type = 'rgb888'
        elif filename.startswith('rgb565_'):
            format_type = 'rgb565'
        elif filename.endswith('.jpg') or filename.endswith('.jpeg'):
            format_type = 'jpeg'
        else:
            print(f"Could not determine format for {filename}, skipping...")
            return False
        
        # Load and process the image
        if format_type == 'jpeg':
            # Load JPEG directly with OpenCV
            frame = cv2.imread(file_path)
            if frame is None:
                raise ValueError(f"Failed to load JPEG file: {file_path}")
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = load_raw_image(file_path, width, height, format_type)
        
        input_data = preprocess_image(frame)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        start_time = time.time()
        interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Get results
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Filter detections with extra validation
        detections = []
        raw_detections = 0
        
        for i in range(output_data.shape[1]):
            x, y, w, h, score, cls = output_data[0, i, :]
            
            # Check if this is a potential face detection with confidence above threshold
            if score > DETECTION_THRESHOLD and round(cls) == FACE_CLASS_ID:
                raw_detections += 1
                
                # Check if face is valid
                is_valid = is_valid_face(x, y, w, h)
                
                # Only add valid faces
                if is_valid:
                    detections.append([x, y, w, h, score])
        
        # Apply NMS to filter redundant detections
        filtered_detections = apply_nms(detections, NMS_THRESHOLD)
        
        # Print detection stats
        print(f"Faces Detected: {raw_detections} raw, {len(detections)} valid, {len(filtered_detections)} after NMS")
        
        # Create a writable copy of the frame for drawing
        result_frame = frame.copy()
        
        # Draw detections on frame
        result_frame = draw_detections(result_frame, filtered_detections)
        
        # Add inference time and face count
        cv2.putText(result_frame, f"Inference: {inference_time:.1f}ms", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result_frame, f"Faces: {len(filtered_detections)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Save the result
        output_filename = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(file_path))[0]}_detected.jpg")
        cv2.imwrite(output_filename, cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))
        print(f"Saved result to: {output_filename}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to process all files"""
    # Get list of files
    files = list_files(RAW_IMAGES_DIR)
    if not files:
        print(f"No files found in {RAW_IMAGES_DIR}")
        return
    
    print(f"Found {len(files)} files to process")
    
    # Process each file
    success_count = 0
    for file_path in files:
        full_path = os.path.join(RAW_IMAGES_DIR, file_path)
        if process_file(full_path, 800, 600):  # Default to 800x600 resolution
            success_count += 1
    
    print(f"\nProcessing complete. Successfully processed {success_count} out of {len(files)} files.")
    print(f"Results saved in: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main() 