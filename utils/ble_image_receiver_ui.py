#!/usr/bin/env python3
import asyncio
import sys
import os
import io
import time
import struct
from PIL import Image, ImageTk, ImageFile
import tkinter as tk
from tkinter import Label, Button, Frame, Canvas, Scrollbar, ttk
import matplotlib.pyplot as plt
from bleak import BleakClient, BleakScanner
import numpy as np
import cv2

# Make PIL more tolerant of incomplete JPEGs
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Constants from the ESP32 code
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
DEVICE_NAME = "Caine's Halo"  # Match the actual device name seen in scanning

# Global variables
image_buffer = bytearray()
jpeg_start_marker = bytes([0xFF, 0xD8])
jpeg_end_marker = bytes([0xFF, 0xD9])
received_packets = 0
last_packet_time = 0
is_receiving_image = False

# Packet tracking variables
expected_packet_seq = 0
missing_packets = 0
corrupted_packets = 0
total_bytes_received = 0
missing_packet_ranges = []  # List to store ranges of missing packet sequence numbers

# Packet size tracking
max_packet_size_seen = 0
packet_size_distribution = {}

# Image transmission timing
image_start_time = 0  # Time when we first see a JPEG start marker

# Auto-save setting
AUTO_SAVE_IMAGES = True

# Calculate checksum (same algorithm as in ESP32)
def calculate_checksum(data):
    checksum = 0
    for byte in data:
        checksum ^= byte  # XOR all bytes
    return checksum

class BluetoothCameraViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Bluetooth Camera Viewer")
        self.root.geometry("1200x600")  # Increased window size
        
        # Main container
        self.main_container = Frame(root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create a PanedWindow to hold all panels
        self.paned_window = ttk.PanedWindow(self.main_container, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Set up the main UI frame
        self.frame = Frame(self.paned_window)
        self.paned_window.add(self.frame, weight=3)  # Main frame gets more space
        
        # Image display - make it expand to fill available space
        self.image_container = Frame(self.frame, bg="black", width=640, height=480)
        self.image_container.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Force the container to keep its size even if empty
        self.image_container.pack_propagate(False)
        
        # Image label with proper settings for centered display
        self.image_label = Label(self.image_container, text="Waiting for image data...", 
                              bg="black", fg="white", anchor="center")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Current image reference
        self.current_image = None
        
        # Status label
        self.status_label = Label(self.frame, text="Not connected")
        self.status_label.pack(pady=5)
        
        # Buffer status
        self.buffer_label = Label(self.frame, text="Buffer: 0 bytes")
        self.buffer_label.pack(pady=5)
        
        # Packets status
        self.packets_label = Label(self.frame, text="Packets: 0")
        self.packets_label.pack(pady=5)
        
        # Image Counter
        self.counter_label = Label(self.frame, text="Images received: 0")
        self.counter_label.pack(pady=5)
        
        # Packet loss tracking
        self.packet_loss_label = Label(self.frame, text="Packet stats: N/A")
        self.packet_loss_label.pack(pady=5)
        
        # MTU size display
        self.mtu_label = Label(self.frame, text="MTU size: Not negotiated")
        self.mtu_label.pack(pady=5)
        
        # Packet size information
        self.packet_size_label = Label(self.frame, text="Max packet: 0 bytes")
        self.packet_size_label.pack(pady=5)
        
        # Image transmission time
        self.transmission_label = Label(self.frame, text="Image transmission time: N/A")
        self.transmission_label.pack(pady=5)
        
        # Images counter
        self.images_received = 0
        
        # Button frame
        self.button_frame = Frame(self.frame)
        self.button_frame.pack(pady=5)
        
        # Connect button
        self.connect_button = Button(self.button_frame, text="Connect", command=self.start_bluetooth)
        self.connect_button.pack(side=tk.LEFT, padx=5)
        
        # Save button
        self.save_button = Button(self.button_frame, text="Save Current Image", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        # Clear button
        self.clear_button = Button(self.button_frame, text="Clear Buffer", command=self.clear_buffer, state=tk.DISABLED)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # Reset Stats button
        self.reset_stats_button = Button(self.button_frame, text="Reset Packet Stats", command=self.reset_packet_stats)
        self.reset_stats_button.pack(side=tk.LEFT, padx=5)
        
        # Auto-save toggle button
        self.auto_save_var = tk.BooleanVar(value=AUTO_SAVE_IMAGES)
        self.auto_save_button = Button(self.button_frame, text="Toggle Auto-save: ON" if AUTO_SAVE_IMAGES else "Toggle Auto-save: OFF", 
                                 command=self.toggle_auto_save)
        self.auto_save_button.pack(side=tk.LEFT, padx=5)
        
        # Create the side panel for frame history
        self.side_panel = Frame(self.paned_window, bg="#f0f0f0")
        self.paned_window.add(self.side_panel, weight=1)  # Side panel gets less space
        
        # Title for side panel
        self.side_panel_title = Label(self.side_panel, text="Frame History", font=("Arial", 12, "bold"), bg="#f0f0f0")
        self.side_panel_title.pack(pady=5, fill=tk.X)
        
        # Create scrollable canvas for the frame history
        self.canvas_frame = Frame(self.side_panel, bg="#f0f0f0")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbar
        self.scrollbar = Scrollbar(self.canvas_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create canvas
        self.canvas = Canvas(self.canvas_frame, bg="#f0f0f0", yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure scrollbar
        self.scrollbar.config(command=self.canvas.yview)
        
        # Create frame inside canvas to hold frame entries
        self.frame_list = Frame(self.canvas, bg="#f0f0f0")
        self.canvas.create_window((0, 0), window=self.frame_list, anchor='nw')
        
        # Configure canvas scrolling
        self.frame_list.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
        # Store frame history data
        self.frame_history = []  # List to store frame metadata and image references
        
        # Add clear history button
        self.clear_history_button = Button(self.side_panel, text="Clear History", command=self.clear_frame_history)
        self.clear_history_button.pack(pady=5, padx=5, fill=tk.X)
        
        # Flag to control the asyncio loop
        self.running = False
        self.client = None
        
        # Working directory
        self.working_dir = os.getcwd()
        print(f"Working directory: {self.working_dir}")
        
    def update_status(self, message):
        self.status_label.config(text=message)
        self.root.update()

    def update_buffer_status(self):
        global image_buffer, received_packets
        self.buffer_label.config(text=f"Buffer: {len(image_buffer)} bytes")
        self.packets_label.config(text=f"Packets: {received_packets}")
        self.counter_label.config(text=f"Images received: {self.images_received}")
        self.root.update()
        
    def update_packet_stats(self):
        """Update the packet loss statistics display"""
        global missing_packets, corrupted_packets, received_packets, total_bytes_received
        
        if received_packets > 0:
            loss_percent = (missing_packets / (received_packets + missing_packets)) * 100 if missing_packets > 0 else 0
            corruption_percent = (corrupted_packets / received_packets) * 100 if corrupted_packets > 0 else 0
            
            self.packet_loss_label.config(
                text=f"Packets: {received_packets} recv, {missing_packets} miss ({loss_percent:.2f}%), {corrupted_packets} corrupt ({corruption_percent:.2f}%)"
            )
        else:
            self.packet_loss_label.config(text="Packet stats: No packets received yet")
        
        self.root.update()
        
    def reset_packet_stats(self):
        """Reset packet tracking statistics"""
        global expected_packet_seq, missing_packets, corrupted_packets, received_packets, total_bytes_received
        global packet_size_distribution
        
        expected_packet_seq = 0
        missing_packets = 0
        corrupted_packets = 0
        received_packets = 0
        total_bytes_received = 0
        packet_size_distribution = {}
        
        self.update_packet_stats()
        self.update_status("Packet statistics reset")
        
    def clear_buffer(self):
        global image_buffer, received_packets
        
        image_buffer = bytearray()
        received_packets = 0
        
        self.update_buffer_status()
        self.image_label.config(image="", text="Waiting for image data...")
        self.save_button.config(state=tk.DISABLED)
        self.clear_button.config(state=tk.DISABLED)
        
    def save_image(self):
        """Save the current image if available"""
        if self.current_image is None:
            self.update_status("No image to save")
            return
            
        # Generate a timestamp filename
        filename = f"manual_image_{int(time.time())}.jpg"
        full_path = os.path.join(self.working_dir, filename)
        
        try:
            # Save the PIL image
            self.current_image.save(full_path, "JPEG")
            self.update_status(f"Image saved as {filename}")
            print(f"Image saved as {full_path}")
        except Exception as e:
            self.update_status(f"Error saving image: {e}")
            print(f"Error saving image: {e}")
            
    def toggle_auto_save(self):
        """Toggle auto-save on or off"""
        global AUTO_SAVE_IMAGES
        
        AUTO_SAVE_IMAGES = not AUTO_SAVE_IMAGES
        self.auto_save_button.config(text=f"Toggle Auto-save: {'ON' if AUTO_SAVE_IMAGES else 'OFF'}")
        self.update_status(f"Auto-save toggled to: {'ON' if AUTO_SAVE_IMAGES else 'OFF'}")
            
    def auto_save_image(self, img, prefix="auto"):
        """Automatically save an image with timestamp"""
        global AUTO_SAVE_IMAGES
        
        if not AUTO_SAVE_IMAGES:
            return False, f"unsaved_{int(time.time())}"
            
        if img is None:
            print("Cannot save: Image is None")
            return False, None
            
        # Generate a timestamp filename
        filename = f"{prefix}_image_{int(time.time())}.jpg"
        full_path = os.path.join(self.working_dir, filename)
        
        try:
            # Save the image
            img.save(full_path, "JPEG")
            self.update_status(f"Auto-saved as {filename}")
            return True, filename
        except Exception as e:
            print(f"Error auto-saving image: {e}")
            return False, f"unsaved_{int(time.time())}"
            
    def display_image(self, img, is_preview=False):
        """Display a PIL Image in the UI with proper scaling"""
        if img is None:
            return False
            
        # Save reference to original image
        self.current_image = img
        
        # Get current container dimensions
        container_width = self.image_container.winfo_width()
        container_height = self.image_container.winfo_height()
        
        # If container isn't properly sized yet, use default size
        if container_width < 50 or container_height < 50:
            container_width = 640
            container_height = 480
        
        # Calculate proper scaling
        img_width, img_height = img.size
        width_ratio = container_width / img_width
        height_ratio = container_height / img_height
        scale_factor = min(width_ratio, height_ratio)
        
        # Calculate new dimensions to maintain aspect ratio
        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)
        
        # Resize the image using a high-quality method
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to Tkinter PhotoImage and display
        photo = ImageTk.PhotoImage(resized_img)
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo  # Keep a reference!
        
        # Enable save button
        self.save_button.config(state=tk.NORMAL)
        self.clear_button.config(state=tk.NORMAL)
        
        return True
        
    def clear_frame_history(self):
        """Clear all frames from the history panel"""
        # Clear the frame history data
        self.frame_history = []
        
        # Destroy all widgets in the frame list
        for widget in self.frame_list.winfo_children():
            widget.destroy()
            
        # Update the canvas
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        # Reset the current image display if needed
        if self.image_label.cget("text") == "":  # If we're displaying an image
            self.image_label.config(image="", text="Waiting for image data...")
            self.save_button.config(state=tk.DISABLED)
        
        self.update_status("Frame history cleared")
        
    def add_frame_to_history(self, img, filename, size, timestamp, packet_stats, transmission_time=None, missing_seq_numbers=None):
        """Add a frame to the history panel with its metadata"""
        # Create a container frame for this entry
        frame_entry = Frame(self.frame_list, bg="#e0e0e0", bd=1, relief=tk.RAISED)
        frame_entry.pack(fill=tk.X, padx=5, pady=5, ipadx=3, ipady=3)
        
        # Create a horizontal layout
        img_frame = Frame(frame_entry, bg="#e0e0e0")
        img_frame.pack(side=tk.LEFT, padx=5, pady=5)
        
        info_frame = Frame(frame_entry, bg="#e0e0e0")
        info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        # Create a small thumbnail
        thumb_size = (80, 60)
        thumbnail = img.copy()
        thumbnail.thumbnail(thumb_size)
        photo = ImageTk.PhotoImage(thumbnail)
        
        # Image thumbnail
        img_label = Label(img_frame, image=photo, bg="#e0e0e0")
        img_label.image = photo  # Keep a reference
        img_label.pack(side=tk.TOP)
        
        # Make the thumbnail clickable
        img_label.bind("<Button-1>", lambda e, img=img: self.display_selected_image(img))
        
        # Frame info
        frame_num = Label(info_frame, text=f"Frame #{len(self.frame_history) + 1}", font=("Arial", 9, "bold"), bg="#e0e0e0")
        frame_num.pack(anchor=tk.W)
        
        # File info
        file_label = Label(info_frame, text=f"File: {filename}", bg="#e0e0e0", font=("Arial", 8))
        file_label.pack(anchor=tk.W)
        
        # Size info
        size_label = Label(info_frame, text=f"Size: {size} bytes", bg="#e0e0e0", font=("Arial", 8))
        size_label.pack(anchor=tk.W)
        
        # Time info
        time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
        time_label = Label(info_frame, text=f"Time: {time_str}", bg="#e0e0e0", font=("Arial", 8))
        time_label.pack(anchor=tk.W)
        
        # Packet stats
        stats_label = Label(info_frame, text=f"Packets: {packet_stats}", bg="#e0e0e0", font=("Arial", 8))
        stats_label.pack(anchor=tk.W)
        
        # Transmission time if available
        if transmission_time is not None:
            trans_label = Label(info_frame, text=f"Transfer time: {transmission_time:.2f} sec", bg="#e0e0e0", font=("Arial", 8))
            trans_label.pack(anchor=tk.W)
            
        # Add buttons for additional actions
        button_frame = Frame(info_frame, bg="#e0e0e0")
        button_frame.pack(anchor=tk.W, pady=3)
        
        # View button
        view_button = Button(button_frame, text="View", font=("Arial", 7), 
                           command=lambda img=img: self.display_selected_image(img))
        view_button.pack(side=tk.LEFT, padx=2)
        
        # Save button
        save_button = Button(button_frame, text="Save", font=("Arial", 7), 
                           command=lambda img=img: self.save_specific_image(img, f"manual_{len(self.frame_history) + 1}"))
        save_button.pack(side=tk.LEFT, padx=2)
        
        # Store the frame data
        frame_data = {
            'image': img,
            'filename': filename,
            'size': size,
            'timestamp': timestamp,
            'packet_stats': packet_stats,
            'resolution': img.size if hasattr(img, 'size') else None,
            'transmission_time': transmission_time,
            'missing_seq_numbers': missing_seq_numbers,
            'frame_entry': frame_entry
        }
        
        # Add to history
        self.frame_history.append(frame_data)
        
        # Update the canvas scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        # Scroll to show the newest entry
        self.canvas.yview_moveto(1.0)
        
    def display_selected_image(self, img):
        """Display an image from history in the main view"""
        self.current_image = img
        self.display_image(img, is_preview=False)
        self.update_status("Displaying selected image from history")
        
    def save_specific_image(self, img, prefix="manual"):
        """Save a specific image from the history"""
        if img is None:
            return
            
        # Generate a timestamp filename
        filename = f"{prefix}_image_{int(time.time())}.jpg"
        full_path = os.path.join(self.working_dir, filename)
        
        try:
            # Save the image
            img.save(full_path, "JPEG")
            self.update_status(f"Saved as {filename}")
            print(f"Saved image as {full_path}")
        except Exception as e:
            self.update_status(f"Error saving image: {e}")
            print(f"Error saving image: {e}")
    
    def start_bluetooth(self):
        self.update_status("Starting Bluetooth...")
        self.connect_button.config(state=tk.DISABLED)
        
        # Start asyncio loop in another thread
        self.running = True
        asyncio.run(self.run_bluetooth())

    def handle_notification(self, sender, data):
        """Handle BLE notifications with incoming data packets"""
        global image_buffer, received_packets, last_packet_time, is_receiving_image
        global expected_packet_seq, missing_packets, corrupted_packets
        global total_bytes_received, missing_packet_ranges
        global max_packet_size_seen, packet_size_distribution, image_start_time
        
        # Update last packet time
        last_packet_time = time.time()
        is_receiving_image = True
        
        # Track packet size statistics
        packet_size = len(data)
        max_packet_size_seen = max(max_packet_size_seen, packet_size)
        
        # Update UI occasionally to avoid excessive updates
        if received_packets % 10 == 0:
            self.packet_size_label.config(text=f"Max packet: {max_packet_size_seen} bytes")
        
        # Track distribution of packet sizes
        if packet_size in packet_size_distribution:
            packet_size_distribution[packet_size] += 1
        else:
            packet_size_distribution[packet_size] = 1
        
        # Check if data packet has header (5+ bytes)
        if len(data) >= 5:
            # Extract header: 4 bytes sequence + 1 byte checksum
            # Fixed to correctly parse big-endian format (MSB first)
            seq_num = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3]
            
            # Debug check the first few packets
            if seq_num <= 10 or seq_num % 100 == 0:
                print(f"Packet seq: {seq_num}")
                print(f"Raw bytes: {data[0]:02x} {data[1]:02x} {data[2]:02x} {data[3]:02x}")
            
            checksum = data[4]
            actual_data = data[5:]
            
            # Check for lost packets
            if seq_num != expected_packet_seq:
                missed = seq_num - expected_packet_seq
                if missed > 0 and missed < 1000:  # Only count reasonable gaps
                    for missing_seq in range(expected_packet_seq, seq_num):
                        missing_packet_ranges.append(missing_seq)
                    
                    missing_packets += missed
                    print(f"MISSING PACKETS: Expected {expected_packet_seq}, got {seq_num} (gap of {missed})")
                else:
                    # Out-of-order packet or sequence reset
                    print(f"Sequence reset or out-of-order packet: Expected {expected_packet_seq}, got {seq_num}")
                    # Handle sequence reset - this is normal at the start of a new image
                    if abs(seq_num - expected_packet_seq) > 1000:
                        print(f"Large sequence gap detected - resetting expected sequence to {seq_num+1}")
                        expected_packet_seq = seq_num
            
            # Update next expected sequence number
            expected_packet_seq = seq_num + 1
            
            # Verify checksum
            calculated_checksum = calculate_checksum(actual_data)
            if calculated_checksum != checksum:
                corrupted_packets += 1
                print(f"CORRUPTED PACKET #{seq_num}: Checksum mismatch! Expected {checksum}, calculated {calculated_checksum}")
                return
            
            # Add only the actual data to buffer
            image_buffer.extend(actual_data)
            received_packets += 1
            total_bytes_received += len(actual_data)
            
            # Periodically update packet stats
            if received_packets % 20 == 0:
                self.update_packet_stats()
                print(f"PACKET STATS: Received {received_packets}, Missing {missing_packets}, Corrupted {corrupted_packets}")
        else:
            # Invalid packet (too small for header)
            corrupted_packets += 1
            print(f"INVALID PACKET: Too small for header ({len(data)} bytes)")
            return
        
        # Update UI periodically
        if received_packets % 10 == 0:
            self.update_buffer_status()
        
        # Process the image if we see a complete JPEG
        try:
            if len(image_buffer) > 4 and image_buffer[:2] == jpeg_start_marker:
                # Now check for end marker
                for i in range(len(image_buffer) - 2):
                    if image_buffer[i:i+2] == jpeg_end_marker and i > 100:  # Must be reasonable size
                        # We have a complete JPEG
                        jpeg_data = image_buffer[:i+2]
                        
                        # Calculate transmission time
                        if image_start_time == 0:
                            image_start_time = time.time() - 0.1  # Approximate if we didn't catch the start
                        
                        transmission_time = time.time() - image_start_time
                        self.transmission_label.config(text=f"Image transmission time: {transmission_time:.2f} sec")
                        
                        # Get a copy of the missing packet ranges for this frame
                        current_missing_ranges = missing_packet_ranges.copy() if missing_packet_ranges else []
                        
                        # Try to load the image
                        try:
                            img = Image.open(io.BytesIO(jpeg_data))
                            img.load()
                            
                            # Auto-save and add to history
                            self.images_received += 1
                            saved, filename = self.auto_save_image(img)
                            self.display_image(img, is_preview=False)
                            
                            # Add to frame history
                            packet_stats = f"{received_packets} recv, {missing_packets} miss, {corrupted_packets} corrupt"
                            self.add_frame_to_history(img, filename, len(jpeg_data), time.time(), packet_stats, 
                                                  transmission_time, current_missing_ranges)
                            
                            # Remove the processed data
                            image_buffer = image_buffer[i+2:]
                            
                            # Reset counters for next image
                            received_packets = 0
                            missing_packets = 0
                            corrupted_packets = 0
                            missing_packet_ranges = []
                            
                            # Reset image start time for next image
                            image_start_time = 0
                            
                            print(f"Complete JPEG image received and processed: {img.width}x{img.height} pixels")
                            break
                        
                        except Exception as e:
                            print(f"Error processing JPEG: {e}")
                
                # If we don't yet have a complete JPEG but found start marker, track the start time
                if image_start_time == 0:
                    image_start_time = time.time()
                    print("JPEG start marker detected - starting to capture image")
            
        except Exception as e:
            print(f"Error checking for JPEG: {e}")

    async def run_bluetooth(self):
        global image_buffer, received_packets, expected_packet_seq, missing_packets, corrupted_packets
        
        # Reset packet stats when starting a new connection
        expected_packet_seq = 0
        missing_packets = 0
        corrupted_packets = 0
        received_packets = 0
        total_bytes_received = 0
        missing_packet_ranges = []
        
        self.update_status("Scanning for devices...")
        
        try:
            # First scan for the device
            device = None
            devices = await BleakScanner.discover()
            for d in devices:
                print(f"Found device: {d.name} ({d.address})")
                if d.name and DEVICE_NAME in d.name:
                    device = d
                    break
            
            if not device:
                self.update_status(f"Device '{DEVICE_NAME}' not found!")
                self.connect_button.config(state=tk.NORMAL)
                return
                
            self.update_status(f"Connecting to {device.name}...")
            
            # Connect to the device
            async with BleakClient(device.address) as client:
                self.client = client
                self.update_status(f"Connected to {device.name}")
                
                # Request the maximum MTU size
                try:
                    mtu = await client.exchange_mtu(517)  # Request max MTU (517)
                    print(f"MTU size negotiated: {mtu} bytes")
                    self.update_status(f"Connected to {device.name} (MTU: {mtu} bytes)")
                    self.mtu_label.config(text=f"MTU size: {mtu} bytes")
                except Exception as e:
                    print(f"Failed to negotiate MTU: {e}")
                    print(f"Will use default MTU size")
                
                # Check if service exists
                services = client.services
                target_service = services.get_service(SERVICE_UUID)
                if not target_service:
                    self.update_status(f"Service {SERVICE_UUID} not found!")
                    return
                
                # Enable notifications with the class method as handler
                await client.start_notify(CHARACTERISTIC_UUID, self.handle_notification)
                self.update_status("Receiving data...")
                
                # Loop to keep connection alive and update UI
                while self.running:
                    # Update buffer status
                    self.update_buffer_status()
                    self.update_packet_stats()
                    
                    # Wait a bit
                    await asyncio.sleep(0.1)
                
                # Clean up
                await client.stop_notify(CHARACTERISTIC_UUID)
                
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            print(f"Exception: {e}")
        finally:
            self.connect_button.config(state=tk.NORMAL)
            self.client = None

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = BluetoothCameraViewer(root)
    
    # Set up proper shutdown
    def on_closing():
        app.running = False
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop() 