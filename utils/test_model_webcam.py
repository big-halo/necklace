import tensorflow as tf
import cv2
import numpy as np
import time

# Load the model
interpreter = tf.lite.Interpreter(model_path='swift_yolo.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Model expects input shape: {input_details[0]['shape']}")
print(f"Model expects input type: {input_details[0]['dtype']}")
print("Forcing INT8 input type to match ESP32 hardware behavior")

# Constants
MODEL_INPUT_SIZE = (192, 192)
DETECTION_THRESHOLD = 45  # Changed from 0.45 to 45 since scores are 0-100
FACE_CLASS_ID = 100
NMS_THRESHOLD = 0.3  # Non-maximum suppression threshold

# Print model details
print("Model details:")
print(f"Input details: {input_details}")
print(f"Output details: {output_details}")

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use default camera
if not cap.isOpened():
    print("Could not open webcam")
    exit()

# Set camera to SVGA resolution (800x600) to match ESP32 camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

def esp32_style_resize(frame, dest_w=192, dest_h=192):
    """
    Simple resize to match model input size
    """
    # Get source dimensions
    src_h, src_w = frame.shape[:2]
    
    # Create output buffer
    output = np.zeros((dest_h, dest_w, 3), dtype=np.uint8)
    
    # Simple nearest-neighbor resize
    for y in range(dest_h):
        src_y = (y * src_h) // dest_h
        for x in range(dest_w):
            src_x = (x * src_w) // dest_w
            # Get the pixel from source and store in RGB order
            b, g, r = frame[src_y, src_x]  # OpenCV uses BGR
            output[y, x, 0] = r
            output[y, x, 1] = g
            output[y, x, 2] = b
    
    return output

def preprocess_image(frame):
    """
    Process the image to match model input requirements
    """
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
        
        # Quantize to INT8 properly:
        # 1. Scale the normalized [0,1] values to the range expected by the model
        # 2. Add zero point
        # 3. Clip to int8 range (-128 to 127)
        # 4. Convert to int8 dtype
        quantized = np.clip(np.round(normalized / scale + zero_point), -128, 127).astype(np.int8)
        
        # Add batch dimension
        batched = np.expand_dims(quantized, axis=0)
        return batched
    else:
        # For float model, just normalize
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
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

# Create a slider window for threshold adjustment
cv2.namedWindow('Threshold Controls')
cv2.createTrackbar('Detection Threshold', 'Threshold Controls', DETECTION_THRESHOLD, 100, lambda x: None)

print("Starting webcam test. Press 'q' to quit.")
fps_time = time.time()
frame_count = 0

try:
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Get current threshold value from slider
        current_threshold = cv2.getTrackbarPos('Detection Threshold', 'Threshold Controls')
            
        # Process the image to match model input requirements
        input_data = preprocess_image(frame)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        start_time = time.time()
        interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Get results
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # If output is quantized, dequantize it
        if output_details[0]['dtype'] == np.int8:
            # Get quantization parameters for output
            output_scale = output_details[0]['quantization'][0]
            output_zero_point = output_details[0]['quantization'][1]
            
            # Dequantize the output
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
            
            # Debug output quantization
            if frame_count % 30 == 0:  # Print every 30 frames
                print(f"\nOutput quantization: scale={output_scale}, zero_point={output_zero_point}")
                print(f"First few output values after dequantization: {output_data[0,0,:]}")
        
        # Filter detections with extra validation
        detections = []
        raw_detections = 0
        
        for i in range(output_data.shape[1]):
            x, y, w, h, score, cls = output_data[0, i, :]
            
            # Check if this is a potential face detection with confidence above threshold
            if score > current_threshold and round(cls) == FACE_CLASS_ID:
                raw_detections += 1
                
                # Print raw detection details if enabled
                if raw_detections % 1 == 0:  # Print all raw detections
                    print(f"RAW Face #{raw_detections}: Score={score:.1f}, Box=({x:.2f}, {y:.2f}, {w:.2f}, {h:.2f})", end=", ")
                    
                # Check if face is valid
                is_valid = is_valid_face(x, y, w, h)
                print(f"Valid={'YES' if is_valid else 'NO'}")
                
                # Only add valid faces
                if is_valid:
                    detections.append([x, y, w, h, score])
        
        # Apply NMS to filter redundant detections
        filtered_detections = apply_nms(detections, NMS_THRESHOLD)
        
        # Print detection stats matching the ESP32 format
        print(f"Faces Detected: {raw_detections} raw, {len(detections)} valid, {len(filtered_detections)} after NMS")
        
        # Draw detections on frame
        result_frame = draw_detections(frame, filtered_detections)
        
        # Count all detections vs filtered detections
        total_detections = len(detections)
        kept_detections = len(filtered_detections)
        
        # Add FPS and inference time
        frame_count += 1
        if (time.time() - fps_time) > 1:
            fps = frame_count / (time.time() - fps_time)
            fps_time = time.time()
            frame_count = 0
        else:
            fps = frame_count / (time.time() - fps_time + 0.001)
        
        cv2.putText(result_frame, f"FPS: {fps:.1f}, Inference: {inference_time:.1f}ms", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result_frame, f"Threshold: {current_threshold}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result_frame, f"Faces: {kept_detections} (filtered from {total_detections})", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Swift YOLO Face Detection', result_frame)
        
        # Show model input (for debugging) - handle both float and quantized models
        if input_details[0]['dtype'] == np.int8:
            # Get quantization parameters
            scale = input_details[0]['quantization'][0]
            zero_point = input_details[0]['quantization'][1]
            
            # Convert back from int8 to float [0,1] then to uint8 [0,255]
            # Properly dequantize by subtracting zero_point first, then scaling
            input_preview = np.clip(((input_data[0].astype(np.float32) - zero_point) * scale * 255), 0, 255).astype(np.uint8)
        else:
            # For float model, just scale to [0,255]
            input_preview = (input_data[0] * 255).astype(np.uint8)
            
        input_preview = cv2.cvtColor(input_preview, cv2.COLOR_RGB2BGR)
        cv2.imshow('Model Input (192x192)', input_preview)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except Exception as e:
    print(f"Error: {e}")
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam test completed") 
