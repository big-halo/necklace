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

# BLE parameters (matching ESP32)
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

class RGB888Receiver:
    def __init__(self, root):
        self.root = root
        self.root.title("RGB888 Image Receiver")
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status label
        self.status_label = ttk.Label(self.main_frame, text="Status: Disconnected")
        self.status_label.grid(row=0, column=0, pady=5)
        
        # Connect button
        self.connect_button = ttk.Button(self.main_frame, text="Connect", command=self.start_scan)
        self.connect_button.grid(row=1, column=0, pady=5)
        
        # Image display
        self.image_label = ttk.Label(self.main_frame)
        self.image_label.grid(row=2, column=0, pady=5)
        
        # Stats labels
        self.fps_label = ttk.Label(self.main_frame, text="FPS: 0")
        self.fps_label.grid(row=3, column=0, pady=5)
        
        self.packet_label = ttk.Label(self.main_frame, text="Packets: 0")
        self.packet_label.grid(row=4, column=0, pady=5)
        
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
        self.sync_window = 100  # Number of packets to wait before considering sync
        
        # Image state tracking
        self.packet_positions = {}  # Track which packets we've received
        self.packet_size = 0  # Size of each packet's data
        self.total_packets = 0  # Total number of packets expected
        
        # Create and start the asyncio event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Start the event loop in a separate thread
        self.loop_thread = threading.Thread(target=self.run_event_loop, daemon=True)
        self.loop_thread.start()
    
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
                    
                    # If we're still in sync window, try to recover
                    if self.packet_count < self.sync_window:
                        print("Still in sync window, attempting to recover...")
                        self.expected_packet_seq = seq_num + 1
                        self.image_buffer = None
                        self.packet_positions.clear()
                    else:
                        # Outside sync window, reset buffer and wait for next image
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
                # First packet - initialize buffer for 192x192 RGB888
                self.image_buffer = bytearray(192 * 192 * 3)  # Pre-allocate full size for RGB888
                self.packet_size = len(actual_data)
                self.total_packets = (192 * 192 * 3 + self.packet_size - 1) // self.packet_size
                print(f"Starting new image buffer, expecting {self.total_packets} packets")
            
            # Calculate position in buffer - use sequence number directly
            packet_index = seq_num % self.total_packets
            buffer_pos = packet_index * self.packet_size
            
            print(f"seq={seq_num}, packet_index={packet_index}, buffer_pos={buffer_pos}, chunk={len(actual_data)}")

            if buffer_pos + len(actual_data) <= len(self.image_buffer):
                # Store packet data
                self.image_buffer[buffer_pos:buffer_pos + len(actual_data)] = actual_data
                self.packet_positions[packet_index] = True
                
                # Check if we have enough packets to render
                if len(self.packet_positions) >= self.total_packets * 0.5:  # Render if we have at least 50% of packets
                    print(f"Rendering partial image with {len(self.packet_positions)}/{self.total_packets} packets")
                    self.process_image(self.image_buffer)
                    
                    # Reset for next image if we have all packets
                    if len(self.packet_positions) >= self.total_packets:
                        print("Image complete, resetting for next image")
                        self.image_buffer = None
                        self.packet_positions.clear()
                        self.packet_count = 0  # Reset packet count for next image
                
        except Exception as e:
            print(f"Notification handling error: {e}")
            import traceback
            traceback.print_exc()
    
    def update_display(self, image):
        """Update the image display"""
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            
            # Resize for display while maintaining aspect ratio
            display_size = (400, 400)  # Square display for 192x192 image
            pil_image.thumbnail(display_size)
            
            # Convert to PhotoImage and keep reference
            photo = ImageTk.PhotoImage(image=pil_image)
            
            # Update label
            self.image_label.image = photo
            self.image_label.config(image=photo)
            
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
            traceback.print_exc()
    
    def process_image(self, data):
        """Process received image data and update display"""
        try:
            # Convert to numpy array with RGB888 format
            image = np.frombuffer(data, dtype=np.uint8)
            
            # Reshape to (height, width, 3) for RGB888
            image = image.reshape((192, 192, 3))
            
            # Flip left-right for correct camera orientation
            image = np.fliplr(image)
            
            # Convert to RGB for display
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Update display
            self.update_display(image)
            
        except Exception as e:
            print(f"Error processing image: {e}")
            traceback.print_exc()

def main():
    root = tk.Tk()
    app = RGB888Receiver(root)
    root.mainloop()

if __name__ == "__main__":
    main() 