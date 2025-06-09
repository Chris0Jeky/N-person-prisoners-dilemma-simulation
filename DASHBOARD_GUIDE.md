# Dashboard Guide - N-Person Prisoner's Dilemma

## Overview

This project contains two different dashboard implementations:

1. **Interactive Web Dashboard** (`web-dashboard/`) - The main, fully-featured dashboard
2. **Static Demo Dashboard** (`npd_cooperation_dashboard.html`) - A standalone demo with hardcoded data

## Main Dashboard (Recommended)

The interactive dashboard in the `web-dashboard/` directory is the primary visualization tool for this project.

### Features
- **Dynamic Data Loading**: Upload JSON/CSV experiment results
- **Experiment Management**: Add, delete, and export experiments
- **Interactive Visualizations**: 
  - Cooperation evolution charts with zoom/pan
  - Strategy performance analysis
  - Network topology visualization with pause/play controls
- **Data Export**: Export individual or all experiments as JSON
- **Responsive Design**: Works on desktop and mobile devices
- **Dark/Light Theme**: Toggle between themes

### How to Use

1. **Open the Dashboard**:
   ```bash
   cd web-dashboard
   # Open index.html in a web browser
   # Or serve locally:
   python -m http.server 8000
   ```

2. **Load Data**:
   - Click the upload button (↑) in the navigation bar
   - Select JSON files from simulation results
   - Or click "Load Example Data" to generate sample data

3. **Manage Experiments**:
   - View all loaded experiments on the Experiments page
   - Click an experiment to analyze it
   - Use the export button to download data
   - Use the delete button to remove experiments

4. **Network Visualization**:
   - View agent interaction networks
   - Use Pause/Resume button to control animation
   - Use Reset button to regenerate the network
   - Drag nodes to rearrange the layout

### Recent Improvements

- ✅ **Fixed Scenario Management**: Added delete buttons for experiments and export functionality
- ✅ **Network Controls**: Added pause/resume and reset buttons for network visualization
- ✅ **Chart Readability**: Improved chart sizing, colors, and responsive layout
- ✅ **Data Export**: Added individual and bulk export features
- ✅ **Better Example Data**: Loads from actual scenario files when available

## Static Demo Dashboard

The `npd_cooperation_dashboard.html` file is a standalone demonstration page that shows:
- Key theoretical insights about cooperation
- Pre-computed results from specific scenarios
- Static visualizations of cooperation dynamics

This is useful for:
- Quick demonstrations without running simulations
- Educational purposes
- Sharing results without the full dashboard

## Test Dashboard

The `web-dashboard/test-dashboard.html` file is a utility for:
- Testing the main dashboard
- Generating sample data for testing
- Quick access to dashboard features

## File Structure

```
/
├── npd_cooperation_dashboard.html    # Static demo (standalone)
└── web-dashboard/                    # Main interactive dashboard
    ├── index.html                    # Dashboard entry point
    ├── test-dashboard.html           # Testing utility
    ├── css/
    │   └── style.css                 # Dashboard styles
    ├── js/
    │   ├── dashboard.js              # Core dashboard logic
    │   ├── data-loader.js            # Data loading utilities
    │   ├── network-vis.js            # Network visualization
    │   └── visualizations.js         # Chart components
    └── README.md                     # Dashboard documentation
```

## Troubleshooting

### Charts Too Large/Small
- Charts now have maximum heights and responsive sizing
- Use browser zoom (Ctrl/Cmd +/-) for additional control

### Can't Delete Experiments
- Fixed: Delete buttons now appear next to each experiment
- Deleted experiments are removed from local storage

### Network Visualization Issues
- Fixed: Pause/Resume and Reset buttons now work properly
- Drag nodes to manually arrange the network

### Loading Real Data
- The dashboard accepts JSON files in the format produced by the simulation
- Use the export feature from completed simulations
- Example data now attempts to load from actual scenario files

## Development

To modify the dashboard:

1. **Styles**: Edit `web-dashboard/css/style.css`
2. **Logic**: Edit `web-dashboard/js/dashboard.js`
3. **Visualizations**: Edit `web-dashboard/js/visualizations.js`
4. **Network**: Edit `web-dashboard/js/network-vis.js`

All changes are immediately visible upon page refresh.