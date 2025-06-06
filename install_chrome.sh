#!/bin/bash

echo "Installing Google Chrome for WSL..."

# Download and add Google Chrome's signing key
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -

# Add the Chrome repository
sudo sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'

# Update package list
sudo apt-get update

# Install Google Chrome
sudo apt-get install -y google-chrome-stable

# Verify installation
if command -v google-chrome &> /dev/null; then
    echo "✅ Google Chrome installed successfully!"
    google-chrome --version
else
    echo "❌ Installation failed"
fi