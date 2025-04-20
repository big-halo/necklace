import serial
import cv2
import numpy as np
import time

# Adjust this to match your ESP32 port and baud rate
PORT = "/dev/cu.usbmodem101"   # Update to your Mac port
BAUD = 460800  # Increased from 115200 to 460800

# BMP file size for 192x192 RGB image with padding
BMP_HEADER_SIZE = 54
IMG_WIDTH = 192
IMG_HEIGHT = 192
ROW_BYTES = IMG_WIDTH * 3
PADDED_ROW_BYTES = (ROW_BYTES + 3) & ~3
BMP_SIZE = BMP_HEADER_SIZE + (PADDED_ROW_BYTES * IMG_HEIGHT)

print(f"Opening serial port {PORT} at {BAUD} baud...")
# Open serial port
ser = serial.Serial(PORT, BAUD, timeout=2)
print("Serial port opened successfully")

# Create a blank image to start with
img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
last_update_time = 0

def read_and_update():
    global img, last_update_time
    buffer = bytearray()
    partial_update = False
    
    # Read whatever data is available
    try:
        data = ser.read_all()
        if data:
            buffer.extend(data)
            print(f"Received {len(data)} bytes")
            
            # If we have enough data for the header, process what we have
            if len(buffer) >= BMP_HEADER_SIZE + 1000:  # At least some pixels after header
                partial_update = True
                
                # Skip header
                raw = buffer[BMP_HEADER_SIZE:]
                
                # Figure out how many complete rows we have
                complete_rows = min(len(raw) // PADDED_ROW_BYTES, IMG_HEIGHT)
                
                # Update the image for the rows we have
                for y in range(complete_rows):
                    row_start = y * PADDED_ROW_BYTES
                    for x in range(IMG_WIDTH):
                        if row_start + x * 3 + 2 < len(raw):
                            i = row_start + x * 3
                            b = raw[i]
                            g = raw[i+1]
                            r = raw[i+2]
                            img[IMG_HEIGHT - 1 - y, x] = [b, g, r]  # Flip vertically
                
                # Show partial image if enough time has passed
                current_time = time.time()
                if current_time - last_update_time > 0.2:  # Update at most 5 times per second
                    cv2.imshow("ESP32 View (Progressive)", img)
                    cv2.waitKey(1)
                    last_update_time = current_time
                    
                # Clear the buffer if it's getting too large
                if len(buffer) > BMP_SIZE * 2:
                    buffer = bytearray()
                    print("Buffer cleared - resetting")
    except Exception as e:
        print(f"Error reading data: {e}")
    
    return partial_update

while True:
    try:
        print("Waiting for image data...")
        while True:
            if read_and_update():
                pass  # We got some data and updated the image
            else:
                time.sleep(0.1)  # No data yet, wait a bit
                
            # Check for key press to exit
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):  # ESC or q key
                break
            
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print("Error:", e)
        print("Error type:", type(e))
        import traceback
        traceback.print_exc()
        time.sleep(1)
