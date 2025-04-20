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

# Make PIL more tolerant of incomplete JPEGs
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Constants from the ESP32 code
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
CONTROL_CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a9"  # New control characteristic for ACKs
DEVICE_NAME = "Anhthin's Halo"  # Change to match your device name

# Global variables
image_buffer = bytearray()
jpeg_start_marker = bytes([0xFF, 0xD8])
jpeg_end_marker = bytes([0xFF, 0xD9])
received_packets = 0
expected_image_size = 0
last_packet_time = 0
is_receiving_image = False
last_image_start_pos = 0  # Track the position of the last JPEG start marker

# Feature flags
AUTO_SAVE_IMAGES = False  # Auto-save is turned off by default

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
        self.root.geometry("1200x600")  # Increased window size to accommodate all panels
        
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
        
        # Timing status
        self.timing_label = Label(self.frame, text="JPEG timing: N/A")
        self.timing_label.pack(pady=5)
        
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
        
        # Timing tracking
        self.last_jpeg_time = 0
        self.min_jpeg_interval = float('inf')
        self.max_jpeg_interval = 0
        self.current_jpeg_interval = 0
        
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
        
        # Reset Max Packet button
        self.reset_max_button = Button(self.button_frame, text="Reset Max Packet", command=self.reset_max_packet_size)
        self.reset_max_button.pack(side=tk.LEFT, padx=5)
        
        # Auto-save toggle button
        self.auto_save_var = tk.BooleanVar(value=AUTO_SAVE_IMAGES)
        self.auto_save_button = Button(self.button_frame, text="Toggle Auto-save: OFF", 
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
        
        # Create the video playback panel on the right
        self.video_panel = Frame(self.paned_window, bg="#f0f0f0")
        self.paned_window.add(self.video_panel, weight=3)  # Video panel gets equal space to main panel
        
        # Title for video panel
        self.video_panel_title = Label(self.video_panel, text="Video Playback", font=("Arial", 12, "bold"), bg="#f0f0f0")
        self.video_panel_title.pack(pady=5, fill=tk.X)
        
        # Video display container
        self.video_container = Frame(self.video_panel, bg="black", width=640, height=480)
        self.video_container.pack(pady=10, fill=tk.BOTH, expand=True)
        self.video_container.pack_propagate(False)
        
        # Video frame display
        self.video_label = Label(self.video_container, text="Video playback will appear here", 
                              bg="black", fg="white", anchor="center")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Video controls frame
        self.video_controls = Frame(self.video_panel, bg="#f0f0f0")
        self.video_controls.pack(pady=5, fill=tk.X)
        
        # Play/Pause button
        self.is_playing = False
        self.play_button = Button(self.video_controls, text="▶ Play", command=self.toggle_playback)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        # FPS control
        self.fps_label = Label(self.video_controls, text="FPS:", bg="#f0f0f0")
        self.fps_label.pack(side=tk.LEFT, padx=5)
        
        self.fps_var = tk.StringVar(value="3")
        fps_values = ["1", "2", "3", "5", "10", "15", "30"]
        self.fps_dropdown = ttk.Combobox(self.video_controls, textvariable=self.fps_var, 
                                      values=fps_values, width=5)
        self.fps_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Status label for video
        self.video_status = Label(self.video_panel, text="Ready", bg="#f0f0f0")
        self.video_status.pack(pady=5)
        
        # Video playback variables
        self.current_frame_index = 0
        self.playback_after_id = None
        
        # Flag to control the asyncio loop
        self.running = False
        self.client = None
        
        # Working directory
        self.working_dir = os.getcwd()
        print(f"Working directory: {self.working_dir}")

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
        
        # Missing packet sequences if available
        if missing_seq_numbers and len(missing_seq_numbers) > 0:
            # Format the missing packet ranges in a readable way
            if len(missing_seq_numbers) > 20:
                # If too many missing packets, just show count and first/last few
                miss_text = f"Missing seq: {len(missing_seq_numbers)} pkts - first: {missing_seq_numbers[:5]}, last: {missing_seq_numbers[-5:]}"
            else:
                miss_text = f"Missing seq: {missing_seq_numbers}"
                
            miss_label = Label(info_frame, text=miss_text, bg="#e0e0e0", font=("Arial", 8))
            miss_label.pack(anchor=tk.W)
        
        # Resolution info if available
        try:
            width, height = img.size
            resolution_label = Label(info_frame, text=f"Resolution: {width}x{height}", bg="#e0e0e0", font=("Arial", 8))
            resolution_label.pack(anchor=tk.W)
        except:
            pass
        
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
        
        # If video not playing, update the current frame index to the last frame
        if not self.is_playing:
            self.current_frame_index = len(self.frame_history) - 1
            # Display the latest frame in the video panel
            self.display_video_frame()
            self.video_status.config(text=f"Latest frame: {self.current_frame_index+1}")

    def display_selected_image(self, img):
        """Display an image from history in the main view"""
        self.current_image = img
        self.display_image(img, is_preview=False)
        self.update_status("Displaying selected image from history")

    def reset_packet_stats(self):
        """Reset packet tracking statistics"""
        global expected_packet_seq, missing_packets, corrupted_packets, received_packets, total_bytes_received
        global packet_size_distribution
        expected_packet_seq = 0
        missing_packets = 0
        corrupted_packets = 0
        received_packets = 0
        total_bytes_received = 0
        packet_size_distribution = {}  # Reset distribution but keep max_packet_size_seen for reference
        self.update_packet_stats()
        self.update_status("Packet statistics reset")

    def update_status(self, message):
        self.status_label.config(text=message)
        self.root.update()

    def update_buffer_status(self):
        global image_buffer, received_packets
        self.buffer_label.config(text=f"Buffer: {len(image_buffer)} bytes")
        self.packets_label.config(text=f"Packets: {received_packets}")
        self.counter_label.config(text=f"Images received: {self.images_received}")
        
        # Update timing label
        if self.current_jpeg_interval > 0:
            self.timing_label.config(text=f"JPEG interval: {self.current_jpeg_interval:.2f}s (min: {self.min_jpeg_interval:.2f}s, max: {self.max_jpeg_interval:.2f}s)")
        
        self.root.update()
        
    def update_packet_stats(self):
        """Update the packet loss statistics display"""
        global missing_packets, corrupted_packets, received_packets, total_bytes_received
        
        if received_packets > 0:
            loss_percent = (missing_packets / (received_packets + missing_packets)) * 100
            corruption_percent = (corrupted_packets / received_packets) * 100
            
            self.packet_loss_label.config(
                text=f"Packets: {received_packets} received, {missing_packets} missing ({loss_percent:.2f}%), {corrupted_packets} corrupted ({corruption_percent:.2f}%)"
            )
        else:
            self.packet_loss_label.config(text="Packet stats: No packets received yet")
        
        self.root.update()
        
    def update_jpeg_timing(self):
        """Update timing statistics when a new complete JPEG is received"""
        current_time = time.time()
        
        # Only update if we've received at least one JPEG before
        if self.last_jpeg_time > 0:
            self.current_jpeg_interval = current_time - self.last_jpeg_time
            
            # Update min/max - ensure they're initialized properly first
            if self.min_jpeg_interval == float('inf') or self.current_jpeg_interval < self.min_jpeg_interval:
                self.min_jpeg_interval = self.current_jpeg_interval
                print(f"New minimum JPEG interval: {self.min_jpeg_interval:.2f}s")
                
            if self.max_jpeg_interval == 0 or self.current_jpeg_interval > self.max_jpeg_interval:
                self.max_jpeg_interval = self.current_jpeg_interval
                print(f"New maximum JPEG interval: {self.max_jpeg_interval:.2f}s")
            
            # Update UI with the latest values
            self.timing_label.config(text=f"JPEG interval: {self.current_jpeg_interval:.2f}s (min: {self.min_jpeg_interval:.2f}s, max: {self.max_jpeg_interval:.2f}s)")
            
            print(f"JPEG interval: {self.current_jpeg_interval:.2f}s (min: {self.min_jpeg_interval:.2f}s, max: {self.max_jpeg_interval:.2f}s)")
        
        # Update last JPEG time
        self.last_jpeg_time = current_time

    def auto_save_image(self, img, prefix="auto"):
        """Automatically save an image with timestamp"""
        global AUTO_SAVE_IMAGES
        
        if not AUTO_SAVE_IMAGES:
            # Auto-save is disabled, return dummy filename but don't save
            return False, f"unsaved_{int(time.time())}"
            
        if img is None:
            print("⚠️ Cannot save: Image is None")
            return False, None
            
        try:
            # Debug output for image
            width, height = img.size
            print(f"📷 Image to save: {width}x{height} pixels, format: {img.format}")
        except Exception as e:
            print(f"⚠️ Error checking image properties: {e}")
        
        # Generate a timestamp filename
        filename = f"{prefix}_image_{int(time.time())}.jpg"
        full_path = os.path.join(self.working_dir, filename)
        
        print(f"📝 Attempting to save to: {full_path}")
        
        try:
            # Save the image
            img.save(full_path, "JPEG")
            self.update_status(f"Auto-saved as {filename}")
            print(f"✅ Auto-saved image as {full_path}")
            return True, filename
        except Exception as e:
            print(f"❌ Error auto-saving image: {e}")
            # Try saving to a different location as fallback
            try:
                alt_path = os.path.join(os.path.expanduser("~"), filename)
                print(f"🔄 Trying alternative path: {alt_path}")
                img.save(alt_path, "JPEG")
                self.update_status(f"Auto-saved to home dir as {filename}")
                print(f"✅ Auto-saved image to alternative path: {alt_path}")
                return True, filename
            except Exception as e2:
                print(f"❌ Alternative save also failed: {e2}")
                return False, f"unsaved_{int(time.time())}"  # Return a dummy filename so we can still add to history

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

    def process_image_data(self, start_pos, end_pos, is_preview=False):
        """Process image data between the specified positions"""
        global image_buffer, missing_packets, corrupted_packets, received_packets
        global image_start_time, expected_packet_seq, last_process_time, missing_packet_ranges
        
        if end_pos <= start_pos or start_pos < 0 or end_pos > len(image_buffer):
            print(f"Invalid image data range: {start_pos} to {end_pos}")
            return False, None
            
        img_data = image_buffer[start_pos:end_pos]
        
        if len(img_data) < 100:
            print(f"Image data too small: {len(img_data)} bytes")
            return False, None
            
        try:
            # Try to create and display the image
            img = Image.open(io.BytesIO(img_data))
            img.load()  # Force load to verify it's valid
            
            if self.display_image(img, is_preview):
                print(f"SUCCESS: Displayed image ({len(img_data)} bytes)")
                
                # Only save and add to history if this is not a preview
                if not is_preview:
                    # Calculate transmission time
                    transmission_time = None
                    if image_start_time > 0:
                        transmission_time = time.time() - image_start_time
                        self.transmission_label.config(text=f"Image transmission time: {transmission_time:.2f} sec")
                        print(f"📊 Timeout image transmission time: {transmission_time:.2f} seconds")
                    
                    # Get a copy of the missing packet ranges for this frame
                    current_missing_ranges = missing_packet_ranges.copy() if missing_packet_ranges else []
                        
                    # Increment image counter and update timing
                    self.images_received += 1
                    self.update_jpeg_timing()
                    self.update_buffer_status()
                    
                    # Auto-save the image
                    saved, filename = self.auto_save_image(img)
                    
                    # Add to frame history with packet stats - ALWAYS add to history even if save fails
                    packet_stats = f"{received_packets} recv, {missing_packets} miss, {corrupted_packets} corrupt"
                    self.add_frame_to_history(img.copy(), filename, len(img_data), time.time(), packet_stats, 
                                           transmission_time, current_missing_ranges)
                    print(f"📋 Added image to history tab with stats: {packet_stats}")
                    if current_missing_ranges:
                        print(f"📋 Missing packets for this frame: {len(current_missing_ranges)} packets")
                        if len(current_missing_ranges) < 20:
                            print(f"📋 Missing sequence numbers: {current_missing_ranges}")
                    
                    # Update processing time tracker
                    last_process_time = time.time()
                    
                    # Reset counters
                    received_packets = 0
                    missing_packets = 0
                    corrupted_packets = 0
                    expected_packet_seq = 0
                    missing_packet_ranges = []  # Clear missing packet ranges
                    
                    # Reset image start time for next image
                    image_start_time = 0
                
                return True, img  # Return both success and the image object
        except Exception as e:
            print(f"Failed to process image: {e}")
            
            # Try adding an end marker if it's missing
            try:
                modified_data = bytearray(img_data)
                if jpeg_end_marker not in modified_data:
                    print("Adding missing JPEG end marker...")
                    modified_data.extend(jpeg_end_marker)
                    img = Image.open(io.BytesIO(modified_data))
                    img.load()
                    
                    if self.display_image(img, is_preview):
                        # Only save and add to history if this is not a preview
                        if not is_preview:
                            # Calculate transmission time
                            transmission_time = None
                            if image_start_time > 0:
                                transmission_time = time.time() - image_start_time
                            
                            # Get a copy of the missing packet ranges for this frame
                            current_missing_ranges = missing_packet_ranges.copy() if missing_packet_ranges else []
                            
                            # Increment image counter and update timing
                            self.images_received += 1
                            self.update_jpeg_timing()
                            self.update_buffer_status()
                            
                            # Auto-save the image
                            saved, filename = self.auto_save_image(img)
                            
                            # Add to frame history with packet stats - ALWAYS add to history
                            packet_stats = f"{received_packets} recv, {missing_packets} miss, {corrupted_packets} corrupt"
                            self.add_frame_to_history(img.copy(), filename, len(modified_data), time.time(), packet_stats,
                                                   transmission_time, current_missing_ranges)
                            print(f"📋 Added image to history tab with stats: {packet_stats}")
                            if current_missing_ranges:
                                print(f"📋 Missing packets for this frame: {len(current_missing_ranges)} packets")
                                if len(current_missing_ranges) < 20:
                                    print(f"📋 Missing sequence numbers: {current_missing_ranges}")
                            
                            # Update processing time tracker
                            last_process_time = time.time()
                            
                            print(f"✅ COMPLETE JPEG #{self.images_received} with added end marker - Size: {len(modified_data)} bytes")
                        else:
                            print(f"❌ Invalid JPEG between markers: {e}")
                    else:
                        print(f"⚠️ JPEG too small between markers: {len(modified_data)} bytes")
            except Exception as e2:
                print(f"Failed after adding end marker: {e2}")
                
        return False, None  # Return failure and no image

    def clear_buffer(self):
        global image_buffer, received_packets, last_image_start_pos
        image_buffer = bytearray()
        received_packets = 0
        last_image_start_pos = 0
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
            # Save the current PIL image directly
            self.current_image.save(full_path, "JPEG")
            self.update_status(f"Image saved as {filename}")
            print(f"Image saved as {full_path}")
        except Exception as e:
            self.update_status(f"Error saving image: {e}")
            print(f"Error saving image: {e}")

    def start_bluetooth(self):
        self.update_status("Starting Bluetooth...")
        self.connect_button.config(state=tk.DISABLED)
        
        # Start asyncio loop in another thread
        self.running = True
        asyncio.run(self.run_bluetooth())

    async def run_bluetooth(self):
        global image_buffer, received_packets, last_packet_time, is_receiving_image
        global last_image_start_pos, expected_packet_seq, missing_packets, corrupted_packets
        global total_bytes_received, last_process_time, image_start_time, missing_packet_ranges
        
        # Reset packet stats when starting a new connection
        expected_packet_seq = 0
        missing_packets = 0
        corrupted_packets = 0
        received_packets = 0
        total_bytes_received = 0
        missing_packet_ranges = []
        
        # Track when we last processed an image
        last_process_time = 0
        
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
                
                # Request the maximum MTU size (517 bytes)
                try:
                    # First get the maximum MTU supported by the client
                    max_supported_mtu = await client.get_mtu_size()
                    print(f"Initial MTU size: {max_supported_mtu}")
                    
                    # Request the maximum MTU size (517 bytes)
                    mtu = await client.exchange_mtu(517)
                    print(f"✅ MTU size negotiated: {mtu} bytes")
                    self.update_status(f"Connected to {device.name} (MTU: {mtu} bytes)")
                    self.mtu_label.config(text=f"MTU size: {mtu} bytes")
                    
                    # Verify if we got the size we requested
                    if mtu < 517:
                        print(f"⚠️ Warning: Negotiated MTU ({mtu}) is less than requested (517)")
                        print(f"This may limit packet sizes and affect performance")
                    else:
                        print(f"✅ Successfully negotiated full 517-byte MTU")
                except Exception as e:
                    print(f"⚠️ Failed to negotiate MTU: {e}")
                    print(f"Will use default MTU size, which may cause packet loss")
                
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
                    
                    current_time = time.time()
                    
                    # Check if we should try to process the current buffer as a complete image
                    # Do this if we've accumulated data and haven't processed an image recently
                    if (len(image_buffer) > 5000 and 
                        last_image_start_pos >= 0 and 
                        (current_time - last_process_time) > 1.0):
                        
                        # Try to find an end marker first
                        for i in range(last_image_start_pos + 2, len(image_buffer) - 1):
                            if image_buffer[i] == 0xFF and image_buffer[i+1] == 0xD9:
                                print(f"🔍 Found JPEG end marker at position {i}, processing complete JPEG")
                                # This is a complete JPEG (from start marker to end marker + 2 bytes)
                                jpeg_data = image_buffer[last_image_start_pos:i+2]
                                
                                # Process the complete JPEG
                                try:
                                    # Calculate transmission time
                                    transmission_time = None
                                    if image_start_time > 0:
                                        transmission_time = current_time - image_start_time
                                        self.transmission_label.config(text=f"Image transmission time: {transmission_time:.2f} sec")
                                        print(f"📊 Image transmission time: {transmission_time:.2f} seconds")
                                    
                                    # Get a copy of the missing packet ranges for this frame
                                    current_missing_ranges = missing_packet_ranges.copy() if missing_packet_ranges else []
                                        
                                    # Update timing and counters
                                    self.update_jpeg_timing()
                                    self.images_received += 1
                                    self.update_buffer_status()
                                    
                                    # Try to load the image
                                    img = Image.open(io.BytesIO(jpeg_data))
                                    img.load()
                                    
                                    # Auto-save and add to history
                                    saved, filename = self.auto_save_image(img)
                                    self.display_image(img, is_preview=False)
                                    
                                    # Add to frame history
                                    packet_stats = f"{received_packets} recv, {missing_packets} miss, {corrupted_packets} corrupt"
                                    self.add_frame_to_history(img.copy(), filename, len(jpeg_data), time.time(), packet_stats, 
                                                           transmission_time, current_missing_ranges)
                                    print(f"📋 Added end-marker-found image to history (frame #{self.images_received})")
                                    if current_missing_ranges:
                                        print(f"📋 Missing packets for this frame: {len(current_missing_ranges)} packets")
                                        if len(current_missing_ranges) < 20:
                                            print(f"📋 Missing sequence numbers: {current_missing_ranges}")
                                    
                                    # Update tracking
                                    last_process_time = current_time
                                    
                                    # Remove the processed data up to end marker
                                    image_buffer = image_buffer[i+2:]
                                    last_image_start_pos = -1  # Reset to look for a new start marker
                                    
                                    # Reset packet counters since we've processed the image
                                    received_packets = 0
                                    missing_packets = 0
                                    corrupted_packets = 0
                                    expected_packet_seq = 0
                                    missing_packet_ranges = []  # Clear missing packet ranges
                                    print("🔄 End marker detection: packet counters reset to 0")
                                    
                                    # Reset image start time for next image
                                    image_start_time = 0
                                    
                                    break
                                    
                                except Exception as e:
                                    print(f"❌ Error processing complete JPEG: {e}")
                        
                    # Check if we're still receiving data (timeout case)
                    if is_receiving_image and current_time - last_packet_time > 2.0:
                        # No data received for 2 seconds, process whatever we have
                        if last_image_start_pos >= 0 and len(image_buffer) > last_image_start_pos + 100:
                            print("No data received for 2 seconds, processing current image")
                            success, img = self.process_image_data(last_image_start_pos, len(image_buffer), is_preview=False)
                            if success:
                                print(f"✅ Successfully processed timeout image")
                                last_process_time = current_time
                            else:
                                print(f"❌ Failed to process timeout image")
                            is_receiving_image = False
                    
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
            
    def handle_notification(self, sender, data):
        """Handle BLE notifications with incoming data packets"""
        global image_buffer, received_packets, last_packet_time, is_receiving_image
        global last_image_start_pos, expected_packet_seq, missing_packets, corrupted_packets
        global total_bytes_received, last_process_time, image_start_time, missing_packet_ranges
        global max_packet_size_seen, packet_size_distribution
        
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
        
        # Log packet size periodically
        if received_packets % 50 == 0:
            print(f"📦 Received packet size: {packet_size} bytes")
            print(f"📊 Maximum packet size seen: {max_packet_size_seen} bytes")
            # Show top 5 most common packet sizes
            sizes = sorted(packet_size_distribution.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"📊 Most common packet sizes: {sizes}")
        
        # Check if data packet has header (5+ bytes)
        if len(data) >= 5:
            # Extract header: 4 bytes sequence + 1 byte checksum
            seq_num = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3]
            checksum = data[4]
            actual_data = data[5:]
            
            # Check for lost packets
            if seq_num != expected_packet_seq:
                missed = seq_num - expected_packet_seq
                if missed > 0:
                    # Record each missing sequence number - only add if it's a reasonable gap
                    # (sometimes sequence might reset or wrap around)
                    if missed < 1000:  # Only record if gap is reasonable
                        for missing_seq in range(expected_packet_seq, seq_num):
                            missing_packet_ranges.append(missing_seq)
                        
                        missing_packets += missed
                        print(f"⚠️ MISSING PACKETS: Expected {expected_packet_seq}, got {seq_num} (gap of {missed})")
                else:
                    # Out-of-order packet or sequence reset
                    print(f"🔄 Sequence reset or out-of-order packet: Expected {expected_packet_seq}, got {seq_num}")
                    # If it's a reset (sequence number much lower than expected), reset expected_packet_seq
                    if seq_num < expected_packet_seq - 1000:
                        expected_packet_seq = seq_num
            
            # Update next expected sequence number
            expected_packet_seq = seq_num + 1
            
            # Verify checksum
            calculated_checksum = calculate_checksum(actual_data)
            if calculated_checksum != checksum:
                corrupted_packets += 1
                print(f"⚠️ CORRUPTED PACKET #{seq_num}: Checksum mismatch! Expected {checksum}, calculated {calculated_checksum}")
            
            # Add only the actual data to buffer
            image_buffer.extend(actual_data)
            received_packets += 1
            total_bytes_received += len(actual_data)
            
            # Periodically update packet stats (every 50 packets)
            if received_packets % 50 == 0:
                self.update_packet_stats()
                print(f"📊 PACKET STATS: Received {received_packets}, Missing {missing_packets}, Corrupted {corrupted_packets}")
                if missing_packet_ranges and len(missing_packet_ranges) < 50:
                    print(f"📊 MISSING SEQUENCE NUMBERS: {missing_packet_ranges}")
                else:
                    print(f"📊 MISSING PACKETS COUNT: {len(missing_packet_ranges)}")
        else:
            # Invalid packet (too small for header)
            corrupted_packets += 1
            print(f"⚠️ INVALID PACKET: Too small for header ({len(data)} bytes)")
            
            # Still add data to buffer as a fallback
            image_buffer.extend(data)
            received_packets += 1
            total_bytes_received += len(data)
        
        # Print debug info periodically
        if received_packets % 50 == 0:
            print(f"Received {received_packets} packets, buffer size: {len(image_buffer)} bytes")
        
        # Update UI periodically
        if received_packets % 10 == 0:
            self.update_buffer_status()
            # Try to update the display with a preview of current data
            self.display_preview()
        
        # Check for JPEG start marker in this newly added data
        # We need to look for it in overlapping data to ensure we don't miss markers
        # that span two packets
        overlap_check_start = max(0, len(image_buffer) - len(data) - 1)
        
        # Look for JPEG start markers in the overlapped region
        for i in range(overlap_check_start, len(image_buffer) - 1):
            if image_buffer[i] == 0xFF and image_buffer[i+1] == 0xD8:
                # Found a new JPEG start marker
                new_marker_pos = i

                # If we already have a previous start marker, process the data between them
                if last_image_start_pos >= 0 and last_image_start_pos < new_marker_pos:
                    print(f"⭐️ New JPEG start marker found at {new_marker_pos}, processing previous complete JPEG")
                    print(f"📊 DEBUG INFO: last_start={last_image_start_pos}, new_start={new_marker_pos}, buffer_size={len(image_buffer)}")
                    
                    # Calculate transmission time for this image
                    transmission_time = None
                    if image_start_time > 0:
                        transmission_time = time.time() - image_start_time
                        self.transmission_label.config(text=f"Image transmission time: {transmission_time:.2f} sec")
                        print(f"📊 Image transmission time: {transmission_time:.2f} seconds")
                    
                    # Get a copy of the missing packet ranges for this frame
                    current_missing_ranges = missing_packet_ranges.copy() if missing_packet_ranges else []
                    current_missing_count = missing_packets
                    current_corrupted_count = corrupted_packets
                    current_received_count = received_packets
                    
                    # This is a complete JPEG (from one start marker to the next)
                    # Process data from last marker to just before this new marker 
                    jpeg_data = image_buffer[last_image_start_pos:new_marker_pos]
                    
                    # Only count and save if we have a reasonable amount of data
                    if len(jpeg_data) > 500:  # Lowered minimum size for debugging
                        # First update the timing stats and increment counter BEFORE processing
                        # This ensures accurate counting regardless of display success
                        self.update_jpeg_timing()
                        self.images_received += 1
                        self.update_buffer_status()
                        
                        print(f"✅ COMPLETE JPEG #{self.images_received} - Size: {len(jpeg_data)} bytes")
                        
                        try:
                            # Try to load the image to verify it's valid
                            img = Image.open(io.BytesIO(jpeg_data))
                            img.load()  # Force load to verify it's valid
                            
                            # Debug output
                            print(f"🖼️ Valid image loaded: {img.size[0]}x{img.size[1]}, mode: {img.mode}")
                            
                            # Auto-save the image and get filename
                            saved, filename = self.auto_save_image(img)
                            
                            # Now display it (don't increment counter again)
                            self.display_image(img, is_preview=False)
                            
                            # Add to frame history with packet stats - ALWAYS add to history even if save fails
                            packet_stats = f"{current_received_count} recv, {current_missing_count} miss, {current_corrupted_count} corrupt"
                            self.add_frame_to_history(img.copy(), filename, len(jpeg_data), time.time(), packet_stats, 
                                                    transmission_time, current_missing_ranges)
                            print(f"📋 Added image to history tab with stats: {packet_stats}")
                            if current_missing_ranges:
                                print(f"📋 Missing packets for this frame: {len(current_missing_ranges)} packets")
                                if len(current_missing_ranges) < 20:
                                    print(f"📋 Missing sequence numbers: {current_missing_ranges}")
                            
                            # Update processing time tracker
                            last_process_time = time.time()
                        except Exception as e:
                            print(f"❌ ERROR processing image: {e}")
                            # Try with added end marker if needed
                            try:
                                if jpeg_end_marker not in jpeg_data:
                                    print("🔄 Trying to fix incomplete JPEG by adding end marker")
                                    modified_data = bytearray(jpeg_data)
                                    modified_data.extend(jpeg_end_marker)
                                    img = Image.open(io.BytesIO(modified_data))
                                    img.load()
                                    
                                    print(f"🖼️ Fixed image loaded: {img.size[0]}x{img.size[1]}, mode: {img.mode}")
                                    
                                    # Auto-save the image and get filename
                                    saved, filename = self.auto_save_image(img)
                                    
                                    # Now display it (don't increment counter again)
                                    self.display_image(img, is_preview=False)
                                    
                                    # Add to frame history with packet stats - ALWAYS add to history
                                    packet_stats = f"{current_received_count} recv, {current_missing_count} miss, {current_corrupted_count} corrupt"
                                    self.add_frame_to_history(img.copy(), filename, len(modified_data), time.time(), packet_stats, 
                                                          transmission_time, current_missing_ranges)
                                    print(f"📋 Added fixed image to history tab with stats: {packet_stats}")
                                    
                                    # Update processing time tracker
                                    last_process_time = time.time()
                                    
                                    print(f"✅ COMPLETE JPEG #{self.images_received} with added end marker - Size: {len(modified_data)} bytes")
                                else:
                                    print(f"❌ Invalid JPEG between markers (has end marker but couldn't load): {e}")
                            except Exception as e2:
                                print(f"❌ Invalid JPEG between markers: {e}, {e2}")
                    else:
                        print(f"⚠️ JPEG too small between markers: {len(jpeg_data)} bytes")
                else:
                    print(f"⭐️ First JPEG start marker found at position {new_marker_pos}")
                    # Set the image start time when we first see a JPEG marker
                    image_start_time = time.time()
                    
                # Update the start position to this new marker
                last_image_start_pos = new_marker_pos
                
                # Remove data before this marker to save memory
                if new_marker_pos > 0:
                    image_buffer = image_buffer[new_marker_pos:]
                    last_image_start_pos = 0  # Reset to 0 since we trimmed the buffer
                    
                    # Reset packet counts when trimming buffer - AFTER we've stored the current stats
                    received_packets = 0
                    missing_packets = 0  # Reset missing packets counter too
                    corrupted_packets = 0  # Reset corrupted packets counter
                    expected_packet_seq = 0  # Reset expected sequence
                    missing_packet_ranges = []  # Clear missing packet ranges
                    print("🔄 Buffer trimmed, packet counters reset to 0")
                    
                break

    def display_preview(self):
        """Display a preview of the current image in progress"""
        global image_buffer, last_image_start_pos
        
        # Only try to display a preview if we have a start marker and enough data
        if last_image_start_pos >= 0 and len(image_buffer) > last_image_start_pos + 100:
            # print(f"Generating preview from bytes {last_image_start_pos} to {len(image_buffer)}")
            return self.process_image_data(last_image_start_pos, len(image_buffer), is_preview=True)
        
        return False

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
        
        # Reset video player
        self.stop_playback()
        self.is_playing = False
        self.current_frame_index = 0
        self.play_button.config(text="▶ Play")
        self.video_label.config(image="", text="Video playback will appear here")
        self.video_status.config(text="No frames available")
            
        self.update_status("Frame history cleared")

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

    def reset_max_packet_size(self):
        """Reset the max packet size seen"""
        global max_packet_size_seen
        max_packet_size_seen = 0
        self.packet_size_label.config(text=f"Max packet: {max_packet_size_seen} bytes")
        self.update_status("Max packet size reset")

    def toggle_auto_save(self):
        """Toggle auto-save on or off"""
        global AUTO_SAVE_IMAGES
        AUTO_SAVE_IMAGES = not AUTO_SAVE_IMAGES
        self.auto_save_button.config(text="Toggle Auto-save: ON" if AUTO_SAVE_IMAGES else "Toggle Auto-save: OFF")
        self.update_status(f"Auto-save toggled to: {'ON' if AUTO_SAVE_IMAGES else 'OFF'}")

    def toggle_playback(self):
        """Toggle video playback"""
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.play_button.config(text="⏸ Pause")
            self.video_status.config(text="Playing...")
            # Start playback
            self.start_playback()
        else:
            self.play_button.config(text="▶ Play")
            self.video_status.config(text="Paused")
            # Stop playback
            self.stop_playback()
            
    def start_playback(self):
        """Start playing frames as a video"""
        if not self.frame_history:
            self.video_status.config(text="No frames to play")
            self.is_playing = False
            self.play_button.config(text="▶ Play")
            return
        
        # Calculate frame delay based on selected FPS
        try:
            fps = float(self.fps_var.get())
            frame_delay = int(1000 / fps)  # Convert to milliseconds
        except ValueError:
            fps = 3.0  # Default to 3 FPS
            frame_delay = int(1000 / fps)
            
        # Reset to start if we're at the end
        if self.current_frame_index >= len(self.frame_history):
            self.current_frame_index = 0
            
        # Display the current frame
        self.display_video_frame()
        
        # Schedule the next frame
        self.playback_after_id = self.root.after(frame_delay, self.next_frame)
        
    def stop_playback(self):
        """Stop video playback"""
        if self.playback_after_id:
            self.root.after_cancel(self.playback_after_id)
            self.playback_after_id = None
            
    def next_frame(self):
        """Display the next frame in the sequence"""
        # Increment frame index
        self.current_frame_index += 1
        
        # If we've reached the end, loop back to the beginning
        if self.current_frame_index >= len(self.frame_history):
            self.current_frame_index = 0
            self.video_status.config(text="Restarting playback...")
        else:
            self.video_status.config(text=f"Playing frame {self.current_frame_index+1}/{len(self.frame_history)}")
            
        # Display the current frame
        self.display_video_frame()
        
        # Schedule the next frame if still playing
        if self.is_playing:
            try:
                fps = float(self.fps_var.get())
                frame_delay = int(1000 / fps)
            except ValueError:
                frame_delay = 333  # Default to ~3 FPS
                
            self.playback_after_id = self.root.after(frame_delay, self.next_frame)
            
    def display_video_frame(self):
        """Display the current frame in the video panel"""
        if not self.frame_history or self.current_frame_index >= len(self.frame_history):
            return
            
        # Get the image from frame history
        frame_data = self.frame_history[self.current_frame_index]
        img = frame_data['image']
        
        # Get container dimensions
        container_width = self.video_container.winfo_width()
        container_height = self.video_container.winfo_height()
        
        # If container isn't properly sized yet, use default size
        if container_width < 50 or container_height < 50:
            container_width = 640
            container_height = 480
        
        # Scale the image to fit
        img_width, img_height = img.size
        width_ratio = container_width / img_width
        height_ratio = container_height / img_height
        scale_factor = min(width_ratio, height_ratio)
        
        # Calculate new dimensions
        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)
        
        # Resize the image
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to Tkinter PhotoImage and display
        photo = ImageTk.PhotoImage(resized_img)
        self.video_label.config(image=photo, text="")
        self.video_label.image = photo  # Keep a reference!

    async def connect_ble():
        print(f"Scanning for device: {DEVICE_NAME}")
        device = await BleakScanner.find_device_by_name(DEVICE_NAME)
        
        if device is None:
            print(f"Could not find device with name: {DEVICE_NAME}")
            return None

        print(f"Found device: {device.name} ({device.address})")
        print(f"Looking for service: {SERVICE_UUID}")
        print(f"Looking for characteristic: {CHARACTERISTIC_UUID}")
        
        client = BleakClient(device)
        
        try:
            await client.connect()
            print("Connected!")
            
            # Debug: Print all services and characteristics
            print("\nAvailable Services:")
            for service in client.services:
                print(f"Service: {service.uuid}")
                for char in service.characteristics:
                    print(f"  Characteristic: {char.uuid}")
                    print(f"    Properties: {char.properties}")
                    print(f"    Handle: {char.handle}")
            
            return client
        except Exception as e:
            print(f"Connection failed: {str(e)}")
            return None

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = BluetoothCameraViewer(root)
    
    # Set up proper shutdown
    def on_closing():
        # Stop video playback
        if app.is_playing:
            app.stop_playback()
        # Indicate the app is shutting down
        app.running = False
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop() 