#!/usr/bin/env python3
import asyncio
from bleak import BleakScanner

async def scan():
    print("Scanning for BLE devices...")
    devices = await BleakScanner.discover()
    print(f"Found {len(devices)} devices:")
    
    for i, device in enumerate(devices):
        print(f"  {i+1}. Name: {device.name}, Address: {device.address}")
        
if __name__ == "__main__":
    asyncio.run(scan()) 