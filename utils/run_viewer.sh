#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate

# Serial port used by viewer
PORT="/dev/cu.usbmodem101"
BAUD="460800"  # Increased from 115200 to 460800

# Check if any process is using the port
echo "Checking if any process is using $PORT..."
PORT_PROCESSES=$(lsof | grep $PORT | awk '{print $2}')

if [ -n "$PORT_PROCESSES" ]; then
  echo "Found processes using $PORT: $PORT_PROCESSES"
  echo "Killing processes..."
  for PID in $PORT_PROCESSES; do
    echo "Killing process $PID..."
    kill $PID
    sleep 0.5
  done
  echo "Waiting a moment for the port to be freed..."
  sleep 1
else
  echo "No processes found using $PORT"
fi

# Run the Python viewer
echo "Starting Python viewer..."
python3 python_viewer.py 