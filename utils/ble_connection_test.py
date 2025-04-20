#!/usr/bin/env python3
import asyncio
import sys
import time
from bleak import BleakClient, BleakScanner
from bleak.exc import BleakError

# Define the Halo camera service and characteristic UUIDs
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

# Target device address (update with the address of your device)
TARGET_DEVICE_ADDRESS = "9D6F8898-3CC6-4320-9153-D59CA9F0032D"
TARGET_DEVICE_NAME = "Anhthin's Halo"

# Stats for tracking data reception
received_packets = 0
total_bytes = 0
start_time = None

# Notification callback
def handle_notification(sender: int, data: bytearray):
    global received_packets, total_bytes, start_time
    
    if start_time is None:
        start_time = time.time()
        print(f"➡️ First packet received at {time.strftime('%H:%M:%S')}")
    
    received_packets += 1
    total_bytes += len(data)
    
    # Print stats every packet for the first 5, then every 10
    if received_packets <= 5 or received_packets % 10 == 0:
        elapsed = time.time() - start_time
        rate = total_bytes / elapsed / 1024 if elapsed > 0 else 0
        
        print(f"📦 Received packet #{received_packets}, size: {len(data)} bytes, total: {total_bytes} bytes ({rate:.2f} KB/s)")
        
        # Print the first 20 bytes of the most recent packet (to check header format)
        print(f"  Header: {data[:20].hex(' ')}")
        
        # If we have BLE packet headers (5 bytes), analyze them
        if len(data) > 5:
            # Extract sequence number (first 4 bytes)
            seq_num = int.from_bytes(data[0:4], byteorder='big')
            checksum = data[4]
            print(f"  Sequence: {seq_num}, Checksum: {checksum:02x}")
            
            # Check for JPEG markers
            jpeg_start = False
            jpeg_end = False
            start_pos = -1
            end_pos = -1
            
            for i in range(5, len(data)-1):
                if data[i] == 0xFF and data[i+1] == 0xD8:
                    jpeg_start = True
                    start_pos = i
                if data[i] == 0xFF and data[i+1] == 0xD9:
                    jpeg_end = True
                    end_pos = i
            
            if jpeg_start:
                print(f"  📸 Found JPEG SOI marker at offset {start_pos}")
            if jpeg_end:
                print(f"  📸 Found JPEG EOI marker at offset {end_pos}")

async def connect_and_wait_for_data():
    print(f"Connecting directly to device: {TARGET_DEVICE_ADDRESS}")
    
    try:
        # Use a long timeout for the initial connection
        async with BleakClient(TARGET_DEVICE_ADDRESS, timeout=20.0) as client:
            print(f"Connected: {client.is_connected}")
            
            # Get services
            print("Discovering services...")
            services = await client.get_services()
            
            # Find our characteristic
            target_char = None
            for service in services:
                print(f"Service: {service.uuid}")
                for char in service.characteristics:
                    props = []
                    if char.properties:
                        for prop in ['broadcast', 'read', 'write-without-response', 'write', 'notify', 'indicate', 'authenticated-signed-writes', 'reliable-write', 'writable-auxiliaries']:
                            if hasattr(char.properties, prop.replace('-', '_')):
                                if getattr(char.properties, prop.replace('-', '_')):
                                    props.append(prop)
                    
                    print(f"  Characteristic: {char.uuid}, Properties: {props}")
                    
                    if str(char.uuid).lower() == CHARACTERISTIC_UUID.lower():
                        target_char = char
                        print(f"  ✅ Found target characteristic: {char.uuid}")
                        
                        # Check if the characteristic has the notify property
                        if 'notify' in props:
                            print(f"  ✅ Characteristic supports notifications")
                        else:
                            print(f"  ❌ Characteristic does NOT support notifications")
            
            if not target_char:
                print(f"Characteristic {CHARACTERISTIC_UUID} not found")
                return
            
            # Start notifications
            print("Enabling notifications...")
            try:
                await client.start_notify(CHARACTERISTIC_UUID, handle_notification)
                print("✅ Notifications enabled successfully")
            except Exception as e:
                print(f"❌ Error enabling notifications: {e}")
                return
            
            # Make an initial read to help trigger the connection on the ESP32 side
            try:
                initial_value = await client.read_gatt_char(CHARACTERISTIC_UUID)
                print(f"Initial characteristic value: {initial_value}")
            except Exception as e:
                print(f"Error reading characteristic: {e}")
            
            # Keep connection alive to receive data
            print(f"Waiting for data (press Ctrl+C to exit)...")
            print(f"Current time: {time.strftime('%H:%M:%S')}")
            
            # Counter for no-data warnings
            warning_count = 0
            
            try:
                while True:
                    await asyncio.sleep(1)
                    
                    # Print status every 5 seconds if no data received
                    if (start_time is None or warning_count % 5 == 0) and warning_count > 0:
                        print(f"⚠️ No data received for {warning_count} seconds. Current time: {time.strftime('%H:%M:%S')}")
                        
                        # After 10 seconds with no data, try to read the characteristic again
                        if warning_count % 10 == 0:
                            try:
                                print("Attempting to read characteristic...")
                                value = await client.read_gatt_char(CHARACTERISTIC_UUID)
                                print(f"Read value: {value}")
                            except Exception as e:
                                print(f"Error reading characteristic: {e}")
                    
                    if start_time is None:
                        warning_count += 1
                        
            except KeyboardInterrupt:
                print("\nStopping...")
            finally:
                # Clean up
                print("Stopping notifications...")
                await client.stop_notify(CHARACTERISTIC_UUID)
                print("Notifications stopped")
    
    except BleakError as e:
        print(f"Error connecting to device: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    print("BLE Connection Test for Halo Camera")
    print(f"Target device: {TARGET_DEVICE_NAME}")
    print(f"Target address: {TARGET_DEVICE_ADDRESS}")
    print(f"Target characteristic: {CHARACTERISTIC_UUID}")
    asyncio.run(connect_and_wait_for_data()) 