# N-Person Prisoner's Dilemma Web Dashboard Plan

## Vision
Create a modern, interactive web dashboard for visualizing and analyzing N-Person Prisoner's Dilemma experiments that can be hosted on GitHub Pages as a static site.

## Architecture Overview

### Phase 1: Static Dashboard (GitHub Pages Compatible)
**Goal**: Beautiful, interactive dashboard that works without a server

#### Core Features:
1. **Modern UI/UX**
   - Responsive design (mobile/tablet/desktop)
   - Dark/light theme toggle
   - Smooth animations and transitions
   - Material Design or similar modern framework

2. **Data Visualization**
   - Interactive charts using D3.js/Plotly.js
   - Network visualizations with force-directed graphs
   - Heatmaps for strategy interactions
   - Time series for cooperation evolution
   - 3D visualizations for complex relationships

3. **Analysis Tools**
   - Strategy performance comparisons
   - Network topology effects
   - Parameter sensitivity analysis
   - Statistical summaries
   - Export charts as PNG/SVG

4. **Data Management**
   - Load pre-generated JSON files from results/
   - Local file upload support
   - Compare multiple experiments
   - Filter and search capabilities

### Phase 2: Enhanced Interactivity
**Goal**: Add dynamic features while maintaining static hosting

1. **Experiment Builder**
   - Visual scenario configuration
   - Parameter presets library
   - Export configurations as JSON

2. **Real-time Analysis**
   - Client-side data processing
   - Advanced filtering and aggregation
   - Custom metrics calculator

3. **Sharing & Collaboration**
   - Shareable visualization links
   - Embed codes for visualizations
   - Export reports as PDF

### Phase 3: Local Computation Bridge (Optional)
**Goal**: Run experiments from the web interface

1. **Local Server Mode**
   - Lightweight Python server
   - WebSocket communication
   - Real-time experiment progress

2. **Hybrid Approach**
   - Static site connects to local server
   - Fallback to file upload if no server
   - Security via local-only connections

## Technology Stack

### Frontend
- **Framework**: React or Vue.js (for component management)
- **Build Tool**: Vite (fast, modern bundler)
- **Styling**: Tailwind CSS + shadcn/ui components
- **Visualizations**: 
  - D3.js for custom visualizations
  - Plotly.js for standard charts
  - Three.js for 3D visualizations
- **State Management**: Zustand or Pinia
- **Routing**: React Router or Vue Router

### Data Processing
- **In-browser computation**: Web Workers
- **Data manipulation**: Danfo.js (pandas-like for JS)
- **Statistics**: Simple Statistics library

### Deployment
- **Hosting**: GitHub Pages
- **CI/CD**: GitHub Actions
- **CDN**: jsDelivr for dependencies

## Implementation Plan

### Step 1: Setup & Foundation (Day 1)
- [ ] Create new `web-dashboard/` directory
- [ ] Setup Vite + React/Vue project
- [ ] Configure GitHub Pages deployment
- [ ] Implement basic routing and layout

### Step 2: Core Visualizations (Days 2-3)
- [ ] Cooperation rate time series
- [ ] Strategy distribution pie/donut charts
- [ ] Network topology visualization
- [ ] Payoff matrix heatmap

### Step 3: Data Management (Day 4)
- [ ] JSON data loader service
- [ ] File upload functionality
- [ ] Data caching system
- [ ] Export functionality

### Step 4: Analysis Tools (Days 5-6)
- [ ] Comparison views
- [ ] Statistical summaries
- [ ] Parameter exploration
- [ ] Custom metrics

### Step 5: Polish & Deploy (Day 7)
- [ ] Responsive design fixes
- [ ] Performance optimization
- [ ] Documentation
- [ ] Deploy to GitHub Pages

## Design Mockup Ideas

### Dashboard Layout
```
┌─────────────────────────────────────────────────┐
│  [Logo] N-Person Prisoner's Dilemma    [Theme]  │
├─────────────────────────────────────────────────┤
│  📊 Overview  🔬 Analysis  🧪 Experiments  ⚙️   │
├────────────┬────────────────────────────────────┤
│            │                                     │
│  Scenario  │     Main Visualization Area        │
│  Selector  │                                     │
│            │   [Interactive Chart/Network]       │
│  Filters   │                                     │
│            │                                     │
│  Metrics   ├─────────────────────────────────────┤
│            │     Secondary Charts Grid          │
│            │  ┌─────────┐ ┌─────────┐          │
│            │  │ Chart 1 │ │ Chart 2 │          │
│            │  └─────────┘ └─────────┘          │
└────────────┴────────────────────────────────────┘
```

### Key Visual Elements
1. **Hero Visualization**: Large, interactive network or time series
2. **Strategy Cards**: Colorful cards showing strategy performance
3. **Comparison Matrix**: Side-by-side experiment comparison
4. **Parameter Explorer**: Interactive sliders with live preview

## Example Pages

### 1. Home/Overview
- Project introduction with animated demo
- Quick start guide
- Featured experiments gallery

### 2. Experiment Viewer
- Load and visualize single experiment
- All charts and metrics for one scenario
- Export and share options

### 3. Comparison Tool
- Compare 2-4 experiments side by side
- Highlight differences
- Statistical significance tests

### 4. Strategy Explorer
- Deep dive into each strategy
- Performance across different networks
- Head-to-head matchups

### 5. Network Analysis
- Interactive network visualizations
- Topology effects on cooperation
- Community detection

## Progressive Enhancement

The dashboard will work in three modes:

1. **Static Mode** (Default)
   - Load pre-generated data files
   - All processing in browser
   - Works on GitHub Pages

2. **Enhanced Mode** (With Local Files)
   - Upload experiment results
   - Process custom data
   - Save visualizations

3. **Connected Mode** (With Local Server)
   - Run new experiments
   - Real-time updates
   - Full control

## Success Metrics

- Load time < 3 seconds
- Mobile responsive
- Accessibility (WCAG 2.1 AA)
- Works offline after first load
- Intuitive navigation
- Beautiful visualizations