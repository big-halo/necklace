import asyncio
from bleak import BleakClient, BleakScanner

# This callback will be called whenever the ESP32 sends data
def notification_handler(sender, data):
    print(f"Received: {data.hex()}")
    # You can also print as raw bytes or try to decode as string:
    try:
        print(f"As string: {data.decode('utf-8')}")
    except:
        pass  # Not valid UTF-8

async def main():
    print("Scanning for BLE devices...")
    
    # Scan for devices
    devices = await BleakScanner.discover()
    
    # List all found devices
    print("Found devices:")
    for i, device in enumerate(devices):
        print(f"{i+1}. {device.name or 'Unknown'} ({device.address})")
    
    if not devices:
        print("No devices found. Exiting.")
        return
    
    # Let user select device
    selection = int(input("Select device number to connect: "))
    selected_device = devices[selection-1]
    
    print(f"Connecting to {selected_device.name or 'Unknown'} ({selected_device.address})...")
    
    # Connect to the selected device
    async with BleakClient(selected_device.address) as client:
        print(f"Connected: {client.is_connected}")
        
        # Discover services and characteristics
        print("Available services and characteristics:")
        notify_chars = []
        
        for service in client.services:
            print(f"Service: {service.uuid}")
            for char in service.characteristics:
                props = char.properties
                print(f"  Characteristic: {char.uuid}")
                print(f"    Properties: {','.join(props)}")
                
                # Collect characteristics that support notifications
                if "notify" in props:
                    notify_chars.append(char)
                    
        if not notify_chars:
            print("No notifiable characteristics found.")
            return
        
        # Let user select which characteristic to subscribe to
        print("\nCharacteristics that support notifications:")
        for i, char in enumerate(notify_chars):
            print(f"{i+1}. {char.uuid}")
        
        char_selection = int(input("Select characteristic to subscribe to: "))
        selected_char = notify_chars[char_selection-1]
        
        # Subscribe to notifications
        await client.start_notify(selected_char.uuid, notification_handler)
        print(f"Subscribed to notifications on {selected_char.uuid}")
        
        print("Waiting for data from ESP32. Press Ctrl+C to exit.")
        
        # Keep the connection open to receive notifications
        while True:
            await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDisconnected")
    except Exception as e:
        print(f"Error: {e}") 