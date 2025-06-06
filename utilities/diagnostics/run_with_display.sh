#!/bin/bash

# Script to run commands with virtual display for Puppeteer/browser tools
# Usage: ./run_with_display.sh [command]

# Check if Xvfb is installed
if ! command -v Xvfb &> /dev/null; then
    echo "Xvfb not found. Installing..."
    sudo apt-get update && sudo apt-get install -y xvfb
fi

# Start Xvfb on display :99
echo "Starting virtual display..."
Xvfb :99 -screen 0 1920x1080x24 &
XVFB_PID=$!

# Export display
export DISPLAY=:99

# Wait a moment for Xvfb to start
sleep 2

# Run the command
echo "Running: $@"
"$@"
EXIT_CODE=$?

# Kill Xvfb
kill $XVFB_PID 2>/dev/null

exit $EXIT_CODE