# Display Environment Setup Guide for WSL

This guide will help you set up a display environment to use browser-based MCP tools.

## Option 1: WSLg (Windows 11) - Easiest

If you're on Windows 11, WSLg should work out of the box:

1. Update WSL:
   ```bash
   wsl --update
   ```

2. Restart WSL:
   ```bash
   wsl --shutdown
   # Then reopen your WSL terminal
   ```

3. Check if it's working:
   ```bash
   echo $DISPLAY  # Should show something like :0 or localhost:0.0
   ```

## Option 2: X Server on Windows (Windows 10 or 11)

### Install an X Server on Windows:

**VcXsrv (Recommended)**
1. Download from: https://sourceforge.net/projects/vcxsrv/
2. Install and run XLaunch
3. Choose "Multiple windows" → "Start no client" → Check "Disable access control"
4. Save the configuration for easy launching

**OR X410** (Paid, from Microsoft Store)
- More integrated experience but costs money

### Configure WSL:

1. In WSL, add to your `~/.bashrc`:
   ```bash
   export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
   export LIBGL_ALWAYS_INDIRECT=1
   ```

2. Reload bashrc:
   ```bash
   source ~/.bashrc
   ```

## Option 3: Headless Operation (No GUI needed)

This is perfect for running browser automation without seeing the browser:

1. Install required packages:
   ```bash
   sudo apt-get update
   sudo apt-get install -y xvfb chromium-browser
   ```

2. Install Chrome dependencies:
   ```bash
   sudo apt-get install -y \
     fonts-liberation libasound2 libatk-bridge2.0-0 libatk1.0-0 \
     libatspi2.0-0 libcups2 libdbus-1-3 libdrm2 libgbm1 libgtk-3-0 \
     libnspr4 libnss3 libxcomposite1 libxdamage1 libxfixes3 \
     libxkbcommon0 libxrandr2 xdg-utils
   ```

3. Run with Xvfb:
   ```bash
   # Method 1: Using xvfb-run
   xvfb-run -a node your-script.js
   
   # Method 2: Manual Xvfb
   Xvfb :99 -screen 0 1920x1080x24 &
   export DISPLAY=:99
   ```

## Testing Your Setup

Run this test command:
```bash
# Test X connection
xset q

# Test with xclock (install x11-apps first)
xclock

# Test headless Chrome
chromium-browser --headless --disable-gpu --dump-dom https://example.com
```

## For MCP Tools Specifically

Once your display is set up, the Puppeteer-based MCP tools should work. If you're running headless:

1. You might need to modify the launch options to include:
   ```javascript
   {
     headless: true,
     args: ['--no-sandbox', '--disable-setuid-sandbox']
   }
   ```

2. Or run your entire session under Xvfb:
   ```bash
   xvfb-run -a npm start  # or however you launch your app
   ```

## Troubleshooting

1. **"Missing X server or $DISPLAY"**
   - Make sure DISPLAY is set: `echo $DISPLAY`
   - For headless: Use Xvfb as shown above

2. **"Failed to connect to the bus"**
   - Install dbus: `sudo apt-get install dbus-x11`
   - Start dbus: `sudo service dbus start`

3. **Chrome won't start**
   - Try with --no-sandbox flag
   - Make sure all dependencies are installed
   - For WSL2, you might need: `--disable-dev-shm-usage`

## Quick Start Script

Save and run this script to set up headless operation:

```bash
#!/bin/bash
# setup-headless.sh

# Install packages
sudo apt-get update
sudo apt-get install -y xvfb chromium-browser x11-apps

# Start Xvfb
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99

# Test
xset q && echo "Display is working!"
```

Remember to run your MCP tools with the appropriate DISPLAY setting!