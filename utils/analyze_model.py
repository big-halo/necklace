import tensorflow as tf
import numpy as np

# Load the model
interpreter = tf.lite.Interpreter(model_path='swift_yolo.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print detailed input information
print('INPUT DETAILS:')
print(f"Name: {input_details[0]['name']}")
print(f"Shape: {input_details[0]['shape']}")
print(f"Data Type: {input_details[0]['dtype']}")
print(f"Quantization: {input_details[0]['quantization']}")

# Print detailed output information
print('\nOUTPUT DETAILS:')
print(f"Name: {output_details[0]['name']}")
print(f"Shape: {output_details[0]['shape']}")
print(f"Data Type: {output_details[0]['dtype']}")
print(f"Quantization: {output_details[0]['quantization']}")

# Test inference with random data to confirm working
random_input = np.random.random((1, 192, 192, 3)).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], random_input)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Analyze output format
print('\nOUTPUT ANALYSIS:')
print(f"Output contains {output_details[0]['shape'][1]} detection boxes")
print(f"Each detection has {output_details[0]['shape'][2]} values")

# Check a few values from the first few detections
print('\nFirst detection values (x, y, w, h, confidence, class):')
print(output_data[0, 0, :])

# Count detections with confidence > 0.25
high_conf = np.sum(output_data[0, :, 4] > 0.25)
print(f"\nDetections with confidence > 0.25: {high_conf}")

# Get unique classes in the model output
classes = np.unique(np.round(output_data[0, :, 5]))
print(f"Possible classes in model output: {classes}") 