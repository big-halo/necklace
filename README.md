# ESP32 Face Detection with Bluetooth Transmission

This project uses an ESP32 camera module to detect faces using TensorFlow Lite, draw rectangles around them, and transmit the images via Bluetooth Low Energy (BLE) to a client application.

## Components

1. **ESP32 Camera Module Code (Arduino)**
   - Face detection using TensorFlow Lite
   - Rectangle drawing around detected faces
   - BLE transmission of images

2. **Python Client**
   - Connects to ESP32 via BLE
   - Receives and displays images with face detection rectangles
   - Saves images to disk

## Setup Instructions

### ESP32 Camera Module

1. Install the required libraries in Arduino IDE:
   - TensorFlow Lite for ESP32
   - ESP32 Camera library
   - BLE libraries

2. Upload the code to your ESP32 camera module:
   - `camera_capture.ino` - Main Arduino file
   - `model_utils.cpp/h` - TensorFlow and camera utilities
   - `ble_utils.cpp/h` - Bluetooth functionality

3. Ensure proper power supply for the ESP32 camera module.

### Python Client

1. Install required Python packages:
   ```
   pip install -r requirements_client.txt
   ```

2. Run the BLE client:
   ```
   python ble_image_receiver.py
   ```

3. The client will:
   - Scan for the "YOLO Face Detector" device
   - Connect and subscribe to notifications
   - Receive, display, and save images when faces are detected

## How It Works

1. The ESP32 captures images at 10 fps and runs face detection
2. When faces are detected, green rectangles are drawn around them
3. Images with faces are sent via BLE (limited to once every 3 seconds)

## BLE Protocol

The BLE protocol consists of several packet types:

1. **Metadata Packet**: Contains image dimensions and total packet count
2. **Data Packets**: Contains actual image data with sequence numbers
3. **End Marker Packet**: Signals the end of an image transmission

Each packet includes a checksum for data integrity validation.

## Troubleshooting

- If the ESP32 isn't detected, ensure Bluetooth is enabled on your computer
- If image reception is incomplete, move closer to the ESP32 or reduce interference
- If transmission is slow, the BLE packet rate can be adjusted in the ESP32 code 