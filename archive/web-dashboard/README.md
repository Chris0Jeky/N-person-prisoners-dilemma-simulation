# N-Person Prisoner's Dilemma Web Dashboard

A modern, interactive web dashboard for visualizing and analyzing N-Person Prisoner's Dilemma experiments. Built with vanilla JavaScript and designed to work as a static site on GitHub Pages.

## Features

- 📊 **Interactive Visualizations**: Dynamic charts for cooperation evolution, strategy performance, and network topology
- 🎨 **Modern UI/UX**: Clean, responsive design with dark/light theme support
- 📁 **File Upload**: Load experiment results from JSON or CSV files
- 🔍 **Analysis Tools**: Compare experiments, analyze strategies, and explore parameter sensitivity
- 🌐 **Network Visualization**: Interactive force-directed graphs for different network topologies
- 💾 **Local Storage**: Caches loaded experiments for quick access
- 📱 **Responsive**: Works on desktop, tablet, and mobile devices

## Quick Start

### Option 1: Open Locally
Simply open `index.html` in a modern web browser (Chrome, Firefox, Safari, Edge).

### Option 2: Serve Locally
```bash
# Using Python
python -m http.server 8000

# Using Node.js
npx serve .

# Then open http://localhost:8000
```

### Option 3: GitHub Pages
1. Push to GitHub repository
2. Enable GitHub Pages in repository settings
3. Access at `https://[username].github.io/[repository]/web-dashboard/`

## Usage

### Loading Data

1. **Upload Files**: Click the upload button and select JSON/CSV experiment results
2. **Load Example Data**: Click "Load Example Data" to see sample experiments
3. **Drag & Drop**: Drag files directly onto the dashboard (coming soon)

### Data Format

#### JSON Format
```json
{
  "scenario_name": "Example Experiment",
  "num_agents": 30,
  "num_rounds": 100,
  "network_type": "small_world",
  "agent_strategies": {
    "tit_for_tat": 10,
    "always_cooperate": 10,
    "always_defect": 10
  },
  "results": [
    {
      "round": 0,
      "cooperation_rate": 0.5,
      "avg_score": 2.5
    }
  ]
}
```

#### CSV Format
```csv
round,cooperation_rate,avg_score,experiment_name
0,0.5,2.5,Example Experiment
1,0.52,2.6,Example Experiment
```

### Navigation

- **Overview**: Summary statistics and main visualizations
- **Experiments**: List and manage loaded experiments
- **Analysis**: Advanced analysis tools and comparisons
- **Strategies**: Explore strategy definitions and performance
- **Network**: Interactive network topology visualization

## Customization

### Adding New Visualizations

Edit `js/visualizations.js` to add custom charts:

```javascript
function createCustomChart(data) {
    const ctx = document.getElementById('myChart');
    return new Chart(ctx, {
        type: 'scatter',
        data: processData(data),
        options: { /* ... */ }
    });
}
```

### Modifying Strategies

Edit `js/strategies.js` to add or modify strategy definitions:

```javascript
STRATEGIES['my_strategy'] = {
    name: 'My Strategy',
    icon: '🎯',
    color: '#123456',
    description: 'Description of my strategy'
};
```

### Styling

Modify `css/style.css` to change colors, spacing, or layout. The dashboard uses CSS variables for easy theming:

```css
:root {
    --accent-blue: #3b82f6;
    --bg-primary: #ffffff;
    /* ... */
}
```

## Browser Support

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Dependencies

All dependencies are loaded from CDN for easy deployment:

- [Chart.js](https://www.chartjs.org/) - For charts and graphs
- [D3.js](https://d3js.org/) - For network visualizations
- [Plotly.js](https://plotly.com/javascript/) - For 3D visualizations
- [Lucide Icons](https://lucide.dev/) - For UI icons

## Future Enhancements

- [ ] Real-time experiment execution
- [ ] WebSocket support for live updates
- [ ] Export visualizations as PNG/SVG
- [ ] Share links for specific views
- [ ] Advanced statistical analysis
- [ ] Machine learning insights
- [ ] Collaborative annotations

## License

Part of the N-Person Prisoner's Dilemma Learning (NPDL) project.