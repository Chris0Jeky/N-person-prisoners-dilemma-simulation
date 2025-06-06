#!/bin/bash

echo "Setting up display environment for browser-based MCP tools..."

# Update package list
sudo apt-get update

# Install dependencies for headless Chrome
echo "Installing Chrome/Chromium dependencies..."
sudo apt-get install -y \
    wget \
    gnupg \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libc6 \
    libcairo2 \
    libcups2 \
    libdbus-1-3 \
    libexpat1 \
    libfontconfig1 \
    libgbm1 \
    libgcc1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libstdc++6 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxrandr2 \
    libxrender1 \
    libxss1 \
    libxtst6 \
    lsb-release \
    xdg-utils

# Install additional X11 utilities
echo "Installing X11 utilities..."
sudo apt-get install -y \
    x11-apps \
    x11-utils \
    x11-xserver-utils \
    dbus-x11

# For better compatibility, install xvfb (X Virtual Framebuffer)
echo "Installing Xvfb for headless operation..."
sudo apt-get install -y xvfb

# Install Chromium browser (lighter than Chrome)
echo "Installing Chromium browser..."
sudo apt-get install -y chromium-browser

echo "Setup complete!"
echo ""
echo "For WSL with Windows 11 (WSLg):"
echo "  Your DISPLAY is already set to: $DISPLAY"
echo ""
echo "For headless operation (no GUI needed):"
echo "  You can use: xvfb-run -a npm start"
echo "  Or set up Xvfb manually:"
echo "    Xvfb :99 -screen 0 1920x1080x24 &"
echo "    export DISPLAY=:99"