#!/bin/bash

echo "Testing MCP with Xvfb setup..."

# Check if Xvfb is installed
if ! command -v Xvfb &> /dev/null; then
    echo "❌ Xvfb is not installed. Please run:"
    echo "   sudo apt-get install -y xvfb"
    exit 1
fi

# Check if Chromium is installed
if ! command -v chromium-browser &> /dev/null; then
    echo "❌ Chromium is not installed. Please run:"
    echo "   sudo apt-get install -y chromium-browser"
    exit 1
fi

echo "✅ Xvfb is installed"
echo "✅ Chromium is installed"

# Start Xvfb on display :99
echo "Starting Xvfb on display :99..."
Xvfb :99 -screen 0 1920x1080x24 &
XVFB_PID=$!
sleep 2

# Set display
export DISPLAY=:99
echo "DISPLAY set to: $DISPLAY"

# Test if display works
if xset q &>/dev/null; then
    echo "✅ Display is working!"
else
    echo "❌ Display is not working"
fi

# Test Chromium in headless mode
echo -e "\nTesting Chromium in headless mode..."
chromium-browser --headless --disable-gpu --no-sandbox --dump-dom https://example.com 2>/dev/null | head -5
if [ $? -eq 0 ]; then
    echo "✅ Chromium headless mode works!"
else
    echo "❌ Chromium headless mode failed"
fi

# Kill Xvfb
kill $XVFB_PID 2>/dev/null

echo -e "\nSetup test complete!"
echo -e "\nTo use MCP tools with Xvfb, you can either:"
echo "1. Run your command with: xvfb-run -a <your-command>"
echo "2. Or start Xvfb manually and set DISPLAY=:99"