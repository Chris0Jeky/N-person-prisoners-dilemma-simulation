#!/bin/bash

echo "Testing display environment..."
echo "Current DISPLAY: $DISPLAY"

# Test 1: Check if X server is accessible
echo -e "\n1. Testing X server connection..."
if xset q &>/dev/null; then
    echo "✓ X server is accessible"
else
    echo "✗ X server is not accessible"
    echo "  Trying to set up Xvfb..."
    
    # Start Xvfb on display :99
    Xvfb :99 -screen 0 1920x1080x24 &
    XVFB_PID=$!
    sleep 2
    
    export DISPLAY=:99
    echo "  DISPLAY set to: $DISPLAY"
    
    if xset q &>/dev/null; then
        echo "✓ Xvfb is now running"
    else
        echo "✗ Failed to start Xvfb"
    fi
fi

# Test 2: Try to run a simple X11 application
echo -e "\n2. Testing X11 application..."
if command -v xclock &> /dev/null; then
    timeout 2s xclock &>/dev/null &
    XCLOCK_PID=$!
    sleep 1
    if ps -p $XCLOCK_PID > /dev/null; then
        echo "✓ X11 applications can run"
        kill $XCLOCK_PID 2>/dev/null
    else
        echo "✗ X11 applications cannot run"
    fi
else
    echo "- xclock not installed (install x11-apps)"
fi

# Test 3: Check Chrome/Chromium
echo -e "\n3. Checking Chrome/Chromium..."
if command -v chromium-browser &> /dev/null; then
    echo "✓ Chromium is installed"
    chromium-browser --version
elif command -v google-chrome &> /dev/null; then
    echo "✓ Google Chrome is installed"
    google-chrome --version
else
    echo "✗ No Chrome/Chromium found"
fi

echo -e "\n4. Testing headless Chrome..."
if command -v chromium-browser &> /dev/null; then
    chromium-browser --headless --disable-gpu --dump-dom https://example.com 2>/dev/null | head -5
    if [ $? -eq 0 ]; then
        echo "✓ Headless Chrome works"
    else
        echo "✗ Headless Chrome failed"
    fi
fi

echo -e "\nDisplay environment test complete!"