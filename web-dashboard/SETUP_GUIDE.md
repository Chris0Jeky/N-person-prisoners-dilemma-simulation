# N-Person Prisoner's Dilemma Dashboard Setup Guide

## Quick Start

### Option 1: Local Viewing (Simplest)
1. Navigate to the `web-dashboard` directory
2. Open `index.html` in a modern web browser (Chrome, Firefox, Safari, Edge)
3. That's it! The dashboard should load immediately

### Option 2: Local Server (Recommended for Development)
```bash
# Using Python 3
cd web-dashboard
python -m http.server 8000

# Using Node.js
npx serve .

# Using PHP
php -S localhost:8000
```

Then open http://localhost:8000 in your browser.

## GitHub Pages Deployment

### Step 1: Prepare Your Repository
1. Ensure the `web-dashboard` folder is committed to your repository
2. Push all changes to GitHub

### Step 2: Enable GitHub Pages
1. Go to your repository on GitHub
2. Click on "Settings" tab
3. Scroll down to "Pages" section
4. Under "Source", select "Deploy from a branch"
5. Choose your branch (usually `main` or `master`)
6. Select "/ (root)" as the folder
7. Click "Save"

### Step 3: Access Your Dashboard
After a few minutes, your dashboard will be available at:
```
https://[your-username].github.io/[repository-name]/web-dashboard/
```

For example:
- Username: `jekyt`
- Repository: `IHateSoar`
- URL: `https://jekyt.github.io/IHateSoar/web-dashboard/`

## Testing the Dashboard

### 1. Generate Test Data
Open `test-dashboard.html` in your browser to:
- Generate sample experiment data
- Test dashboard functionality
- Download test JSON files

### 2. Load Example Data
Click "Load Example Data" on the Experiments page to load pre-configured examples.

### 3. Upload Your Own Data
Use the upload button (↑) to load JSON or CSV files with experiment results.

## Data Format

### JSON Format
```json
{
    "scenario_name": "My Experiment",
    "config": {
        "num_agents": 30,
        "num_rounds": 100,
        "network_type": "small_world",
        "agent_strategies": {
            "tit_for_tat": 10,
            "always_cooperate": 10,
            "always_defect": 10
        }
    },
    "results": [
        {
            "round": 0,
            "cooperation_rate": 0.5,
            "avg_score": 2.5
        }
        // ... more rounds
    ]
}
```

### CSV Format
```csv
round,cooperation_rate,avg_score,experiment_name
0,0.5,2.5,My Experiment
1,0.52,2.6,My Experiment
```

## Features Overview

### 📊 Overview Page
- Real-time statistics
- Cooperation evolution charts
- Strategy performance graphs
- Key metrics display

### 🧪 Experiments Page
- List of loaded experiments
- Quick statistics for each experiment
- Click to analyze individual experiments
- Bulk operations support

### 📈 Analysis Tools
Four analysis modes:
1. **Comparison**: Compare multiple experiments
2. **Sensitivity**: Parameter sensitivity analysis
3. **Statistics**: Statistical analysis and hypothesis testing
4. **Advanced Viz**: 3D plots, heatmaps, animated visualizations

### 👥 Strategies Page
- Visual strategy explorer
- Performance metrics
- Strategy descriptions
- Icon-based identification

### 🌐 Network Visualization
- Interactive force-directed graphs
- Three network types: Fully Connected, Small World, Scale-Free
- Drag and zoom support
- Real-time network generation

## Advanced Features

### Dark Mode
Click the moon icon (🌙) in the navigation to toggle dark mode.

### Data Caching
Experiments are automatically cached in browser storage for quick access.

### Export Options
- Download visualizations as images (coming soon)
- Export analysis results as CSV (coming soon)

## Troubleshooting

### Dashboard Not Loading
1. Check browser console for errors (F12)
2. Ensure you're using a modern browser
3. Try clearing cache and reloading

### Charts Not Displaying
1. Check if JavaScript is enabled
2. Ensure CDN resources are loading (requires internet)
3. Try a different browser

### File Upload Issues
1. Ensure files are valid JSON or CSV
2. Check file size (large files may take time)
3. Verify data format matches examples

### GitHub Pages Not Working
1. Wait 10-15 minutes after enabling
2. Check repository settings for errors
3. Ensure index.html is in the correct path
4. Try hard refresh (Ctrl+F5)

## Browser Requirements
- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Performance Tips
1. Load fewer experiments for better performance
2. Use the data filtering options
3. Close unused visualizations
4. Consider using a local server for large datasets

## Future Enhancements
- [ ] Real-time experiment execution
- [ ] WebSocket support for live updates
- [ ] Export visualizations as PNG/SVG
- [ ] Collaborative annotations
- [ ] Machine learning insights
- [ ] Parameter optimization tools

## Getting Help
1. Check the console for error messages
2. Review the test page for examples
3. Examine the sample data format
4. File issues on GitHub

## Contributing
Feel free to submit pull requests for:
- New visualization types
- Additional analysis tools
- Performance improvements
- Bug fixes
- Documentation updates

---

Happy analyzing! 🎯📊🔬