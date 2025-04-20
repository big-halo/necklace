import tensorflow as tf
import numpy as np

# Load the model
interpreter = tf.lite.Interpreter(model_path='swift_yolo.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to run inference and collect class data
def check_model_output(num_runs=20):
    all_classes = []
    
    for i in range(num_runs):
        # Generate different types of test patterns that might trigger different detections
        if i % 4 == 0:
            # Random noise
            test_input = np.random.random((1, 192, 192, 3)).astype(np.float32)
        elif i % 4 == 1:
            # Gradient pattern
            x = np.linspace(0, 1, 192)
            y = np.linspace(0, 1, 192)
            xx, yy = np.meshgrid(x, y)
            test_input = np.stack([xx, yy, xx*yy], axis=-1).reshape(1, 192, 192, 3).astype(np.float32)
        elif i % 4 == 2:
            # Circular pattern
            x = np.linspace(-1, 1, 192)
            y = np.linspace(-1, 1, 192)
            xx, yy = np.meshgrid(x, y)
            circle = np.sqrt(xx**2 + yy**2)
            test_input = np.stack([circle, 1-circle, circle*0.5], axis=-1).reshape(1, 192, 192, 3).astype(np.float32)
        else:
            # Rectangle pattern (might look like a person/face)
            test_input = np.zeros((1, 192, 192, 3), dtype=np.float32)
            # Add a rectangle in the middle
            test_input[0, 70:120, 80:110, :] = 0.8
            # Add a small circle on top (like a head)
            for y in range(55, 75):
                for x in range(85, 105):
                    if (x-95)**2 + (y-65)**2 < 100:
                        test_input[0, y, x, :] = 0.9
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Get high confidence detections
        confident_detections = output_data[0, output_data[0, :, 4] > 0.4]
        if len(confident_detections) > 0:
            for detection in confident_detections:
                cls = int(np.round(detection[5]))
                score = detection[4]
                all_classes.append((cls, score))
            print(f"Run {i}: Found {len(confident_detections)} confident detections")
    
    # Summarize findings
    unique_classes = {}
    for cls, score in all_classes:
        if cls not in unique_classes or score > unique_classes[cls]:
            unique_classes[cls] = score
    
    print("\nDetected classes and their highest confidence scores:")
    for cls, score in sorted(unique_classes.items()):
        print(f"Class {cls}: confidence {score:.4f}")
    
    return unique_classes

# Run the test
print("Testing model with various input patterns...")
detected_classes = check_model_output(30)

print("\nThe model appears to be trained to detect objects of class:", list(detected_classes.keys()))
print("This YOLO model most likely uses the COCO dataset class numbering, where:")
coco_classes = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird",
    15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon",
    45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut",
    55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven",
    70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
    75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush",
    100: "face"  # This is a custom class, not in original COCO
}

print("\nPossible object names:")
for cls in detected_classes.keys():
    if cls in coco_classes:
        print(f"Class {cls}: {coco_classes[cls]}")
    else:
        print(f"Class {cls}: Unknown - custom model class") 