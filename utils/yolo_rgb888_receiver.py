#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk
import asyncio
from bleak import BleakScanner, BleakClient
import numpy as np
import cv2
from PIL import Image, ImageTk
import time
import threading
import tensorflow as tf

# BLE parameters (matching ESP32)
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

# YOLO model parameters
MODEL_INPUT_SIZE = (192, 192)
DETECTION_THRESHOLD = 0.45
FACE_CLASS_ID = 100
NMS_THRESHOLD = 0.3

class YOLORGB888Receiver:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Face Detection with RGB888 Receiver")
        
        # Load the YOLO model
        self.interpreter = tf.lite.Interpreter(model_path='swift_yolo.tflite')
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"Model expects input shape: {self.input_details[0]['shape']}")
        print(f"Model expects input type: {self.input_details[0]['dtype']}")
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status label
        self.status_label = ttk.Label(self.main_frame, text="Status: Disconnected")
        self.status_label.grid(row=0, column=0, pady=5)
        
        # Connect button
        self.connect_button = ttk.Button(self.main_frame, text="Connect", command=self.start_scan)
        self.connect_button.grid(row=1, column=0, pady=5)
        
        # Image displays
        self.raw_image_label = ttk.Label(self.main_frame, text="Raw Image")
        self.raw_image_label.grid(row=2, column=0, pady=5)
        
        self.detection_label = ttk.Label(self.main_frame, text="Detections")
        self.detection_label.grid(row=2, column=1, pady=5)
        
        # Stats labels
        self.fps_label = ttk.Label(self.main_frame, text="FPS: 0")
        self.fps_label.grid(row=3, column=0, columnspan=2, pady=5)
        
        self.packet_label = ttk.Label(self.main_frame, text="Packets: 0")
        self.packet_label.grid(row=4, column=0, columnspan=2, pady=5)
        
        # Detection threshold slider
        self.threshold_frame = ttk.Frame(self.main_frame)
        self.threshold_frame.grid(row=5, column=0, columnspan=2, pady=5)
        
        self.threshold_label = ttk.Label(self.threshold_frame, text="Detection Threshold:")
        self.threshold_label.pack(side=tk.LEFT)
        
        self.threshold_scale = ttk.Scale(self.threshold_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                      command=self.update_threshold)
        self.threshold_scale.set(DETECTION_THRESHOLD * 100)
        self.threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Initialize variables
        self.client = None
        self.device = None
        self.is_connected = False
        self.image_buffer = None
        self.last_frame_time = 0
        self.frame_count = 0
        self.packet_count = 0
        self.expected_packet_seq = 0
        self.missing_packets = 0
        self.is_first_packet = True
        self.last_seq_num = 0
        self.sync_window = 100
        
        # Image state tracking
        self.packet_positions = {}
        self.packet_size = 0
        self.total_packets = 0
        
        # Create and start the asyncio event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Start the event loop in a separate thread
        self.loop_thread = threading.Thread(target=self.run_event_loop, daemon=True)
        self.loop_thread.start()
    
    def update_threshold(self, value):
        """Update detection threshold from slider"""
        global DETECTION_THRESHOLD
        DETECTION_THRESHOLD = float(value) / 100.0
    
    def calculate_iou(self, box1, box2):
        """Calculate intersection over union between two boxes"""
        box1_x1 = box1[0] - box1[2]/2
        box1_y1 = box1[1] - box1[3]/2
        box1_x2 = box1[0] + box1[2]/2
        box1_y2 = box1[1] + box1[3]/2
        
        box2_x1 = box2[0] - box2[2]/2
        box2_y1 = box2[1] - box2[3]/2
        box2_x2 = box2[0] + box2[2]/2
        box2_y2 = box2[1] + box2[3]/2
        
        xi1 = max(box1_x1, box2_x1)
        yi1 = max(box1_y1, box2_y1)
        xi2 = min(box1_x2, box2_x2)
        yi2 = min(box1_y2, box2_y2)
        
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / union_area if union_area > 0 else 0
        return iou
    
    def apply_nms(self, detections, nms_threshold):
        """Apply Non-Maximum Suppression to filter overlapping detections"""
        if len(detections) == 0:
            return []
        
        sorted_detections = sorted(detections, key=lambda x: x[4], reverse=True)
        kept_detections = []
        
        while len(sorted_detections) > 0:
            best_detection = sorted_detections.pop(0)
            kept_detections.append(best_detection)
            
            filtered_detections = []
            for detection in sorted_detections:
                iou = self.calculate_iou(best_detection[:4], detection[:4])
                if iou <= nms_threshold:
                    filtered_detections.append(detection)
            
            sorted_detections = filtered_detections
        
        return kept_detections
    
    def preprocess_image(self, frame):
        """Process the image for model input using exact ESP32 algorithm"""
        # Get source dimensions
        src_h, src_w = frame.shape[:2]
        dest_w, dest_h = MODEL_INPUT_SIZE
        
        # Create output buffer
        output = np.zeros((dest_h, dest_w, 3), dtype=np.uint8)
        
        # Implement exact same nearest-neighbor algorithm as ESP32
        for y in range(dest_h):
            src_y = (y * src_h) // dest_h  # Integer division, same as ESP32
            for x in range(dest_w):
                src_x = (x * src_w) // dest_w  # Integer division, same as ESP32
                
                # Get the pixel from source
                b, g, r = frame[src_y, src_x]  # OpenCV uses BGR
                
                # Convert to RGB565 format (simulate the hardware camera output)
                r5 = (r >> 3) & 0x1F  # Convert 8-bit to 5-bit
                g6 = (g >> 2) & 0x3F  # Convert 8-bit to 6-bit
                b5 = (b >> 3) & 0x1F  # Convert 8-bit to 5-bit
                
                # Pack into 16-bit RGB565 value (as hardware would)
                rgb565 = (r5 << 11) | (g6 << 5) | b5
                
                # Unpack back to 8-bit RGB (as done in ESP32 code)
                r8 = ((rgb565 >> 11) & 0x1F) << 3
                g8 = ((rgb565 >> 5) & 0x3F) << 2
                b8 = (rgb565 & 0x1F) << 3
                
                # Store in RGB order
                output[y, x, 0] = r8
                output[y, x, 1] = g8
                output[y, x, 2] = b8
        
        # Normalize to float32 [0,1] range
        normalized = output.astype(np.float32) / 255.0
        
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        return batched
    
    def run_event_loop(self):
        """Run the asyncio event loop in a separate thread"""
        try:
            self.loop.run_forever()
        except Exception as e:
            print(f"Event loop error: {e}")
    
    def calculate_checksum(self, data):
        """Calculate XOR checksum of data"""
        checksum = 0
        for byte in data:
            checksum ^= byte
        return checksum
    
    async def scan_for_device(self):
        """Scan for BLE device"""
        self.status_label.config(text="Status: Scanning...")
        self.connect_button.config(state='disabled')
        
        try:
            devices = await BleakScanner.discover()
            for device in devices:
                if SERVICE_UUID in [str(uuid) for uuid in device.metadata.get('uuids', [])]:
                    self.device = device
                    self.status_label.config(text=f"Status: Found device {device.name or device.address}")
                    await self.connect_to_device()
                    return
            
            self.status_label.config(text="Status: Device not found")
            
        except Exception as e:
            self.status_label.config(text=f"Status: Scan error - {str(e)}")
        
        finally:
            self.connect_button.config(state='normal')
    
    async def connect_to_device(self):
        """Connect to the BLE device"""
        try:
            self.client = BleakClient(self.device.address, timeout=20.0)
            await self.client.connect()
            self.is_connected = True
            self.status_label.config(text="Status: Connected")
            
            # Reset sync state
            self.is_first_packet = True
            self.packet_count = 0
            self.missing_packets = 0
            self.image_buffer = None
            self.expected_packet_seq = 0
            self.last_seq_num = 0
            
            # Subscribe to notifications
            await self.client.start_notify(CHARACTERISTIC_UUID, self.handle_notification)
            
            # Update UI
            self.connect_button.config(text="Disconnect", command=self.disconnect)
            
        except Exception as e:
            self.status_label.config(text=f"Status: Connection failed - {str(e)}")
            self.connect_button.config(state='normal')
    
    async def disconnect(self):
        """Disconnect from the BLE device"""
        if self.client and self.client.is_connected:
            try:
                await self.client.disconnect()
                self.is_connected = False
                self.status_label.config(text="Status: Disconnected")
                self.connect_button.config(text="Connect", command=self.start_scan)
            except Exception as e:
                print(f"Disconnect error: {e}")
    
    def start_scan(self):
        """Start scanning for devices"""
        asyncio.run_coroutine_threadsafe(self.scan_for_device(), self.loop)
    
    def handle_notification(self, sender, data):
        """Handle incoming BLE notifications"""
        try:
            if len(data) < 5:  # Minimum packet size
                return
            
            # Extract header
            seq_num = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3]
            checksum = data[4]
            actual_data = data[5:]
            
            # Verify checksum
            calculated_checksum = self.calculate_checksum(actual_data)
            if calculated_checksum != checksum:
                print(f"Checksum mismatch: got {checksum}, calculated {calculated_checksum}")
                return
            
            # Handle initial sync
            if self.is_first_packet:
                self.expected_packet_seq = seq_num + 1
                self.is_first_packet = False
                print(f"Initial sync: starting from sequence {seq_num}")
                return
            
            # Handle packet sequence
            if seq_num != self.expected_packet_seq:
                missed = seq_num - self.expected_packet_seq
                if missed > 0 and missed < 1000:
                    self.missing_packets += missed
                    print(f"Missed {missed} packets between {self.expected_packet_seq} and {seq_num}")
                    
                    if self.packet_count < self.sync_window:
                        print("Still in sync window, attempting to recover...")
                        self.expected_packet_seq = seq_num + 1
                        self.image_buffer = None
                        self.packet_positions.clear()
                    else:
                        print("Outside sync window, resetting buffer")
                        self.image_buffer = None
                        self.packet_positions.clear()
            
            self.expected_packet_seq = seq_num + 1
            self.packet_count += 1
            self.last_seq_num = seq_num
            
            # Update packet count display
            self.packet_label.config(text=f"Packets: {self.packet_count}, Missing: {self.missing_packets}")
            
            # Process the image data
            if self.image_buffer is None:
                self.image_buffer = bytearray(192 * 192 * 3)
                self.packet_size = len(actual_data)
                self.total_packets = (192 * 192 * 3 + self.packet_size - 1) // self.packet_size
                print(f"Starting new image buffer, expecting {self.total_packets} packets")
            
            # Calculate position in buffer
            packet_index = seq_num % self.total_packets
            buffer_pos = packet_index * self.packet_size
            
            if buffer_pos + len(actual_data) <= len(self.image_buffer):
                self.image_buffer[buffer_pos:buffer_pos + len(actual_data)] = actual_data
                self.packet_positions[packet_index] = True
                
                if len(self.packet_positions) >= self.total_packets * 0.5:
                    self.process_image(self.image_buffer)
                    
                    if len(self.packet_positions) >= self.total_packets:
                        self.image_buffer = None
                        self.packet_positions.clear()
                        self.packet_count = 0
                
        except Exception as e:
            print(f"Notification handling error: {e}")
            import traceback
            traceback.print_exc()
    
    def update_display(self, raw_image, detection_image):
        """Update both raw and detection image displays"""
        try:
            # Convert numpy arrays to PIL Images
            raw_pil = Image.fromarray(raw_image)
            detection_pil = Image.fromarray(detection_image)
            
            # Resize for display while maintaining aspect ratio
            display_size = (400, 400)
            raw_pil.thumbnail(display_size)
            detection_pil.thumbnail(display_size)
            
            # Convert to PhotoImage and keep references
            raw_photo = ImageTk.PhotoImage(image=raw_pil)
            detection_photo = ImageTk.PhotoImage(image=detection_pil)
            
            # Update labels
            self.raw_image_label.image = raw_photo
            self.raw_image_label.config(image=raw_photo)
            
            self.detection_label.image = detection_photo
            self.detection_label.config(image=detection_photo)
            
            # Update FPS counter
            current_time = time.time()
            if current_time - self.last_frame_time >= 1.0:
                fps = self.frame_count
                self.fps_label.config(text=f"FPS: {fps}")
                self.frame_count = 0
                self.last_frame_time = current_time
            self.frame_count += 1
            
        except Exception as e:
            print(f"Display update error: {e}")
            import traceback
            traceback.print_exc()
    
    def process_image(self, data):
        """Process received image data and run YOLO inference"""
        try:
            # Convert to numpy array with RGB888 format
            image = np.frombuffer(data, dtype=np.uint8)
            image = image.reshape((192, 192, 3))
            
            # Create a copy for detection display
            detection_image = image.copy()
            
            # Debug: Print some pixel values to verify color space
            print(f"Sample pixels: {image[0,0]}, {image[0,1]}, {image[1,0]}")
            
            # Preprocess for model - just normalize to float32 [0,1]
            normalized = image.astype(np.float32) / 255.0
            input_data = np.expand_dims(normalized, axis=0)
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            start_time = time.time()
            self.interpreter.invoke()
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # Get results
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Process detections
            detections = []
            for i in range(output_data.shape[1]):
                x, y, w, h, score, cls = output_data[0, i, :]
                
                if score > DETECTION_THRESHOLD and round(cls) == FACE_CLASS_ID:
                    detections.append([x, y, w, h, score])
            
            # Draw all detections
            for detection in detections:
                x, y, w, h, score = detection
                
                # Convert normalized coordinates to pixel coordinates
                x1 = int((x - w/2) * MODEL_INPUT_SIZE[0])
                y1 = int((y - h/2) * MODEL_INPUT_SIZE[1])
                x2 = int((x + w/2) * MODEL_INPUT_SIZE[0])
                y2 = int((y + h/2) * MODEL_INPUT_SIZE[1])
                
                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(MODEL_INPUT_SIZE[0]-1, x2), min(MODEL_INPUT_SIZE[1]-1, y2)
                
                # Draw bounding box
                cv2.rectangle(detection_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add confidence text
                label = f"Face: {score:.2f}"
                cv2.putText(detection_image, label, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add inference time and face count
            cv2.putText(detection_image, f"Inference: {inference_time:.1f}ms",
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(detection_image, f"Faces: {len(detections)}",
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Update display
            self.update_display(image, detection_image)
            
        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()

def main():
    root = tk.Tk()
    app = YOLORGB888Receiver(root)
    root.mainloop()

if __name__ == "__main__":
    main() 