/**
 * Analysis Tools Module
 * Interactive tools for analyzing experiment data
 */

class AnalysisTools {
    constructor() {
        this.comparisons = [];
        this.filters = {
            strategies: [],
            networkTypes: [],
            minAgents: 0,
            maxAgents: 1000,
            minRounds: 0,
            maxRounds: 10000
        };
    }

    /**
     * Create experiment comparison interface
     */
    createComparisonInterface(containerId, experiments) {
        const container = document.getElementById(containerId);
        if (!container) return;

        container.innerHTML = `
            <div class="comparison-tools">
                <div class="tool-section">
                    <h3>Select Experiments to Compare</h3>
                    <div class="experiment-selector">
                        ${experiments.map((exp, idx) => `
                            <label class="experiment-item">
                                <input type="checkbox" value="${idx}" class="exp-checkbox">
                                <span class="exp-name">${exp.name}</span>
                                <span class="exp-info">${exp.config.num_agents} agents, ${exp.config.num_rounds} rounds</span>
                            </label>
                        `).join('')}
                    </div>
                </div>
                
                <div class="tool-section">
                    <h3>Comparison Type</h3>
                    <div class="comparison-options">
                        <button class="compare-btn" data-type="cooperation">Cooperation Evolution</button>
                        <button class="compare-btn" data-type="strategy">Strategy Performance</button>
                        <button class="compare-btn" data-type="network">Network Effects</button>
                        <button class="compare-btn" data-type="convergence">Convergence Analysis</button>
                    </div>
                </div>
                
                <div id="comparison-results" class="results-section"></div>
            </div>
        `;

        // Event listeners
        const checkboxes = container.querySelectorAll('.exp-checkbox');
        const compareButtons = container.querySelectorAll('.compare-btn');

        checkboxes.forEach(cb => {
            cb.addEventListener('change', () => {
                this.comparisons = Array.from(checkboxes)
                    .filter(cb => cb.checked)
                    .map(cb => experiments[parseInt(cb.value)]);
            });
        });

        compareButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const type = btn.dataset.type;
                this.runComparison(type, 'comparison-results');
            });
        });
    }

    /**
     * Run comparison analysis
     */
    runComparison(type, resultContainerId) {
        const container = document.getElementById(resultContainerId);
        if (!container || this.comparisons.length === 0) {
            container.innerHTML = '<p class="warning">Please select at least one experiment to analyze.</p>';
            return;
        }

        switch (type) {
            case 'cooperation':
                this.compareCooperationEvolution(container);
                break;
            case 'strategy':
                this.compareStrategyPerformance(container);
                break;
            case 'network':
                this.compareNetworkEffects(container);
                break;
            case 'convergence':
                this.analyzeConvergence(container);
                break;
        }
    }

    /**
     * Compare cooperation evolution across experiments
     */
    compareCooperationEvolution(container) {
        container.innerHTML = `
            <h4>Cooperation Evolution Comparison</h4>
            <canvas id="coop-comparison-chart" width="800" height="400"></canvas>
            <div class="stats-grid" id="coop-stats"></div>
        `;

        // Create comparison chart
        const viz = new Visualizations();
        viz.createCooperationEvolutionChart('coop-comparison-chart', this.comparisons);

        // Calculate statistics
        const stats = this.comparisons.map(exp => {
            const coopRates = exp.results.map(r => r.cooperation_rate);
            return {
                name: exp.name,
                initial: coopRates[0],
                final: coopRates[coopRates.length - 1],
                average: coopRates.reduce((a, b) => a + b) / coopRates.length,
                variance: this.calculateVariance(coopRates),
                trend: this.calculateTrend(coopRates)
            };
        });

        // Display statistics
        const statsContainer = document.getElementById('coop-stats');
        statsContainer.innerHTML = `
            <table class="stats-table">
                <thead>
                    <tr>
                        <th>Experiment</th>
                        <th>Initial</th>
                        <th>Final</th>
                        <th>Average</th>
                        <th>Variance</th>
                        <th>Trend</th>
                    </tr>
                </thead>
                <tbody>
                    ${stats.map(s => `
                        <tr>
                            <td>${s.name}</td>
                            <td>${(s.initial * 100).toFixed(1)}%</td>
                            <td>${(s.final * 100).toFixed(1)}%</td>
                            <td>${(s.average * 100).toFixed(1)}%</td>
                            <td>${s.variance.toFixed(4)}</td>
                            <td class="${s.trend > 0 ? 'positive' : s.trend < 0 ? 'negative' : 'neutral'}">
                                ${s.trend > 0 ? '📈' : s.trend < 0 ? '📉' : '➡️'} ${(s.trend * 100).toFixed(2)}%
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
    }

    /**
     * Compare strategy performance
     */
    compareStrategyPerformance(container) {
        container.innerHTML = `
            <h4>Strategy Performance Analysis</h4>
            <div class="strategy-analysis">
                <div id="strategy-radar" style="width: 400px; height: 400px;">
                    <canvas id="radar-chart"></canvas>
                </div>
                <div id="strategy-rankings"></div>
            </div>
        `;

        // Create radar chart
        const viz = new Visualizations();
        viz.createStrategyRadarChart('radar-chart', this.comparisons);

        // Calculate strategy rankings
        const strategyScores = {};
        
        this.comparisons.forEach(exp => {
            const finalCoop = exp.results[exp.results.length - 1]?.cooperation_rate || 0;
            
            Object.entries(exp.config.agent_strategies || {}).forEach(([strategy, count]) => {
                if (!strategyScores[strategy]) {
                    strategyScores[strategy] = {
                        experiments: 0,
                        totalScore: 0,
                        cooperationRates: []
                    };
                }
                
                strategyScores[strategy].experiments++;
                strategyScores[strategy].totalScore += finalCoop * count;
                strategyScores[strategy].cooperationRates.push(finalCoop);
            });
        });

        // Sort strategies by performance
        const rankings = Object.entries(strategyScores)
            .map(([strategy, data]) => ({
                strategy,
                avgScore: data.totalScore / data.experiments,
                avgCooperation: data.cooperationRates.reduce((a, b) => a + b) / data.cooperationRates.length,
                appearances: data.experiments
            }))
            .sort((a, b) => b.avgScore - a.avgScore);

        // Display rankings
        document.getElementById('strategy-rankings').innerHTML = `
            <h5>Strategy Rankings</h5>
            <div class="rankings-list">
                ${rankings.map((r, idx) => `
                    <div class="ranking-item">
                        <span class="rank">#${idx + 1}</span>
                        <span class="strategy-icon">${STRATEGIES[r.strategy]?.icon || '?'}</span>
                        <span class="strategy-name">${STRATEGIES[r.strategy]?.name || r.strategy}</span>
                        <span class="score">Score: ${r.avgScore.toFixed(3)}</span>
                        <span class="coop-rate">Cooperation: ${(r.avgCooperation * 100).toFixed(1)}%</span>
                        <span class="appearances">(${r.appearances} experiments)</span>
                    </div>
                `).join('')}
            </div>
        `;
    }

    /**
     * Compare network effects
     */
    compareNetworkEffects(container) {
        // Group experiments by network type
        const byNetwork = {};
        this.comparisons.forEach(exp => {
            const netType = exp.config.network_type;
            if (!byNetwork[netType]) byNetwork[netType] = [];
            byNetwork[netType].push(exp);
        });

        container.innerHTML = `
            <h4>Network Effects Analysis</h4>
            <div class="network-comparison">
                ${Object.entries(byNetwork).map(([netType, exps]) => `
                    <div class="network-group">
                        <h5>${this.formatNetworkType(netType)}</h5>
                        <canvas id="network-${netType}" width="400" height="300"></canvas>
                    </div>
                `).join('')}
            </div>
            <div id="network-insights"></div>
        `;

        // Create charts for each network type
        Object.entries(byNetwork).forEach(([netType, exps]) => {
            const ctx = document.getElementById(`network-${netType}`);
            if (ctx) {
                this.createNetworkComparisonChart(ctx, exps);
            }
        });

        // Generate insights
        this.generateNetworkInsights(byNetwork, 'network-insights');
    }

    /**
     * Analyze convergence patterns
     */
    analyzeConvergence(container) {
        container.innerHTML = `
            <h4>Convergence Analysis</h4>
            <div class="convergence-analysis">
                <canvas id="convergence-chart" width="800" height="400"></canvas>
                <div id="convergence-metrics"></div>
            </div>
        `;

        // Calculate convergence metrics
        const convergenceData = this.comparisons.map(exp => {
            const rates = exp.results.map(r => r.cooperation_rate);
            return {
                name: exp.name,
                convergenceRound: this.findConvergenceRound(rates),
                oscillationMagnitude: this.calculateOscillation(rates),
                steadyStateValue: this.calculateSteadyState(rates),
                convergenceSpeed: this.calculateConvergenceSpeed(rates)
            };
        });

        // Create convergence visualization
        this.createConvergenceChart('convergence-chart', this.comparisons, convergenceData);

        // Display metrics
        document.getElementById('convergence-metrics').innerHTML = `
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>Experiment</th>
                        <th>Convergence Round</th>
                        <th>Steady State</th>
                        <th>Oscillation</th>
                        <th>Speed</th>
                    </tr>
                </thead>
                <tbody>
                    ${convergenceData.map(c => `
                        <tr>
                            <td>${c.name}</td>
                            <td>${c.convergenceRound === -1 ? 'No convergence' : c.convergenceRound}</td>
                            <td>${(c.steadyStateValue * 100).toFixed(1)}%</td>
                            <td>${c.oscillationMagnitude.toFixed(4)}</td>
                            <td>${c.convergenceSpeed.toFixed(3)}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
    }

    /**
     * Parameter sensitivity analysis
     */
    createParameterSensitivityAnalysis(containerId, experiments) {
        const container = document.getElementById(containerId);
        if (!container) return;

        container.innerHTML = `
            <div class="sensitivity-analysis">
                <h3>Parameter Sensitivity Analysis</h3>
                
                <div class="parameter-selector">
                    <label>X-Axis Parameter:
                        <select id="x-param">
                            <option value="num_agents">Number of Agents</option>
                            <option value="num_rounds">Number of Rounds</option>
                            <option value="network_density">Network Density</option>
                        </select>
                    </label>
                    
                    <label>Y-Axis Metric:
                        <select id="y-metric">
                            <option value="final_cooperation">Final Cooperation Rate</option>
                            <option value="avg_cooperation">Average Cooperation</option>
                            <option value="convergence_time">Convergence Time</option>
                            <option value="stability">Stability</option>
                        </select>
                    </label>
                    
                    <button id="analyze-sensitivity" class="action-btn">Analyze</button>
                </div>
                
                <div id="sensitivity-plot"></div>
                <div id="sensitivity-insights"></div>
            </div>
        `;

        // Event listener
        document.getElementById('analyze-sensitivity').addEventListener('click', () => {
            const xParam = document.getElementById('x-param').value;
            const yMetric = document.getElementById('y-metric').value;
            this.runSensitivityAnalysis(experiments, xParam, yMetric, 'sensitivity-plot', 'sensitivity-insights');
        });
    }

    /**
     * Statistical analysis dashboard
     */
    createStatisticalDashboard(containerId, experiments) {
        const container = document.getElementById(containerId);
        if (!container) return;

        container.innerHTML = `
            <div class="statistical-dashboard">
                <h3>Statistical Analysis</h3>
                
                <div class="stat-cards">
                    <div class="stat-card">
                        <h4>Correlation Analysis</h4>
                        <canvas id="correlation-heatmap" width="300" height="300"></canvas>
                    </div>
                    
                    <div class="stat-card">
                        <h4>Distribution Analysis</h4>
                        <canvas id="distribution-chart" width="300" height="300"></canvas>
                    </div>
                    
                    <div class="stat-card">
                        <h4>Hypothesis Testing</h4>
                        <div id="hypothesis-results"></div>
                    </div>
                </div>
                
                <div class="detailed-stats">
                    <h4>Detailed Statistics</h4>
                    <div id="stats-table"></div>
                </div>
            </div>
        `;

        // Run statistical analyses
        this.createCorrelationHeatmap(experiments, 'correlation-heatmap');
        this.createDistributionChart(experiments, 'distribution-chart');
        this.runHypothesisTests(experiments, 'hypothesis-results');
        this.generateDetailedStatistics(experiments, 'stats-table');
    }

    // Helper methods
    calculateVariance(values) {
        const mean = values.reduce((a, b) => a + b) / values.length;
        const squaredDiffs = values.map(v => Math.pow(v - mean, 2));
        return squaredDiffs.reduce((a, b) => a + b) / values.length;
    }

    calculateTrend(values) {
        const n = values.length;
        const x = Array.from({ length: n }, (_, i) => i);
        const y = values;
        
        const sumX = x.reduce((a, b) => a + b);
        const sumY = y.reduce((a, b) => a + b);
        const sumXY = x.reduce((acc, xi, i) => acc + xi * y[i], 0);
        const sumX2 = x.reduce((acc, xi) => acc + xi * xi, 0);
        
        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        return slope;
    }

    findConvergenceRound(values, threshold = 0.01) {
        for (let i = 10; i < values.length - 10; i++) {
            const window = values.slice(i, i + 10);
            const variance = this.calculateVariance(window);
            if (variance < threshold) return i;
        }
        return -1;
    }

    calculateOscillation(values) {
        let oscillations = 0;
        for (let i = 1; i < values.length - 1; i++) {
            if ((values[i] > values[i-1] && values[i] > values[i+1]) ||
                (values[i] < values[i-1] && values[i] < values[i+1])) {
                oscillations++;
            }
        }
        return oscillations / values.length;
    }

    calculateSteadyState(values) {
        const last10 = values.slice(-10);
        return last10.reduce((a, b) => a + b) / last10.length;
    }

    calculateConvergenceSpeed(values) {
        const convergenceRound = this.findConvergenceRound(values);
        if (convergenceRound === -1) return 0;
        return 1 / convergenceRound;
    }

    formatNetworkType(type) {
        const formats = {
            'fully_connected': 'Fully Connected',
            'small_world': 'Small World',
            'scale_free': 'Scale-Free',
            'random': 'Random',
            'lattice': 'Lattice'
        };
        return formats[type] || type;
    }

    createNetworkComparisonChart(canvas, experiments) {
        const ctx = canvas.getContext('2d');
        
        new Chart(ctx, {
            type: 'box',
            data: {
                labels: experiments.map(e => e.name),
                datasets: [{
                    label: 'Cooperation Distribution',
                    data: experiments.map(exp => {
                        const rates = exp.results.map(r => r.cooperation_rate * 100);
                        return {
                            min: Math.min(...rates),
                            q1: this.percentile(rates, 25),
                            median: this.percentile(rates, 50),
                            q3: this.percentile(rates, 75),
                            max: Math.max(...rates)
                        };
                    }),
                    backgroundColor: '#3b82f6' + '40',
                    borderColor: '#3b82f6'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }

    percentile(values, p) {
        const sorted = values.slice().sort((a, b) => a - b);
        const index = (p / 100) * (sorted.length - 1);
        const lower = Math.floor(index);
        const upper = Math.ceil(index);
        const weight = index % 1;
        
        if (lower === upper) return sorted[lower];
        return sorted[lower] * (1 - weight) + sorted[upper] * weight;
    }

    generateNetworkInsights(byNetwork, containerId) {
        const container = document.getElementById(containerId);
        
        const insights = [];
        Object.entries(byNetwork).forEach(([netType, exps]) => {
            const avgCoop = exps.reduce((sum, exp) => {
                const final = exp.results[exp.results.length - 1]?.cooperation_rate || 0;
                return sum + final;
            }, 0) / exps.length;
            
            insights.push({
                network: this.formatNetworkType(netType),
                avgCooperation: avgCoop,
                numExperiments: exps.length
            });
        });
        
        insights.sort((a, b) => b.avgCooperation - a.avgCooperation);
        
        container.innerHTML = `
            <h5>Network Insights</h5>
            <ul class="insights-list">
                ${insights.map((insight, idx) => `
                    <li>
                        <strong>${insight.network}</strong> networks achieved 
                        <span class="highlight">${(insight.avgCooperation * 100).toFixed(1)}%</span> 
                        average cooperation
                        ${idx === 0 ? ' (best performance)' : ''}
                        <span class="sample-size">(n=${insight.numExperiments})</span>
                    </li>
                `).join('')}
            </ul>
        `;
    }

    createConvergenceChart(canvasId, experiments, convergenceData) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        
        // Prepare data showing convergence patterns
        const datasets = experiments.map((exp, idx) => {
            const rates = exp.results.map(r => r.cooperation_rate * 100);
            const smoothed = this.smoothData(rates, 5);
            
            return {
                label: exp.name,
                data: smoothed.map((v, i) => ({ x: i, y: v })),
                borderColor: `hsl(${idx * 360 / experiments.length}, 70%, 50%)`,
                backgroundColor: 'transparent',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3
            };
        });

        new Chart(ctx, {
            type: 'line',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Convergence Patterns'
                    },
                    annotation: {
                        annotations: convergenceData.map((c, idx) => ({
                            type: 'line',
                            xMin: c.convergenceRound,
                            xMax: c.convergenceRound,
                            borderColor: `hsl(${idx * 360 / experiments.length}, 70%, 50%)`,
                            borderWidth: 2,
                            borderDash: [5, 5],
                            label: {
                                content: `${c.name} converged`,
                                enabled: false
                            }
                        })).filter(a => a.xMin !== -1)
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        title: { display: true, text: 'Round' }
                    },
                    y: {
                        title: { display: true, text: 'Cooperation Rate (%)' },
                        min: 0,
                        max: 100
                    }
                }
            }
        });
    }

    smoothData(data, windowSize) {
        const smoothed = [];
        for (let i = 0; i < data.length; i++) {
            const start = Math.max(0, i - Math.floor(windowSize / 2));
            const end = Math.min(data.length, i + Math.floor(windowSize / 2) + 1);
            const window = data.slice(start, end);
            smoothed.push(window.reduce((a, b) => a + b) / window.length);
        }
        return smoothed;
    }

    runSensitivityAnalysis(experiments, xParam, yMetric, plotId, insightsId) {
        // Extract data points
        const dataPoints = experiments.map(exp => {
            let x, y;
            
            // Extract x value
            switch (xParam) {
                case 'num_agents':
                    x = exp.config.num_agents;
                    break;
                case 'num_rounds':
                    x = exp.config.num_rounds;
                    break;
                case 'network_density':
                    x = this.estimateNetworkDensity(exp);
                    break;
            }
            
            // Extract y value
            switch (yMetric) {
                case 'final_cooperation':
                    y = exp.results[exp.results.length - 1]?.cooperation_rate || 0;
                    break;
                case 'avg_cooperation':
                    const rates = exp.results.map(r => r.cooperation_rate);
                    y = rates.reduce((a, b) => a + b) / rates.length;
                    break;
                case 'convergence_time':
                    y = this.findConvergenceRound(exp.results.map(r => r.cooperation_rate));
                    break;
                case 'stability':
                    y = 1 - this.calculateVariance(exp.results.map(r => r.cooperation_rate));
                    break;
            }
            
            return { x, y, name: exp.name };
        });
        
        // Create scatter plot with trend line
        const plotContainer = document.getElementById(plotId);
        const data = [{
            x: dataPoints.map(p => p.x),
            y: dataPoints.map(p => p.y),
            text: dataPoints.map(p => p.name),
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: 10,
                color: dataPoints.map(p => p.y),
                colorscale: 'Viridis',
                showscale: true
            }
        }];
        
        // Add trend line
        const trendline = this.calculateTrendline(dataPoints);
        data.push({
            x: [Math.min(...dataPoints.map(p => p.x)), Math.max(...dataPoints.map(p => p.x))],
            y: [trendline.predict(Math.min(...dataPoints.map(p => p.x))), 
                trendline.predict(Math.max(...dataPoints.map(p => p.x)))],
            mode: 'lines',
            name: 'Trend',
            line: { color: 'red', dash: 'dash' }
        });
        
        const layout = {
            title: `${this.formatParamName(xParam)} vs ${this.formatMetricName(yMetric)}`,
            xaxis: { title: this.formatParamName(xParam) },
            yaxis: { title: this.formatMetricName(yMetric) },
            height: 400
        };
        
        Plotly.newPlot(plotContainer, data, layout);
        
        // Generate insights
        const insightsContainer = document.getElementById(insightsId);
        const correlation = this.calculateCorrelation(dataPoints.map(p => p.x), dataPoints.map(p => p.y));
        
        insightsContainer.innerHTML = `
            <div class="sensitivity-insights">
                <h5>Key Insights</h5>
                <ul>
                    <li>Correlation coefficient: <strong>${correlation.toFixed(3)}</strong></li>
                    <li>Relationship: <strong>${Math.abs(correlation) > 0.7 ? 'Strong' : Math.abs(correlation) > 0.4 ? 'Moderate' : 'Weak'}</strong> ${correlation > 0 ? 'positive' : 'negative'}</li>
                    <li>R-squared: <strong>${(correlation * correlation).toFixed(3)}</strong></li>
                    <li>Trend: ${trendline.slope > 0 ? 'Increasing' : 'Decreasing'} (slope: ${trendline.slope.toFixed(4)})</li>
                </ul>
            </div>
        `;
    }

    estimateNetworkDensity(experiment) {
        const n = experiment.config.num_agents;
        const networkType = experiment.config.network_type;
        
        switch (networkType) {
            case 'fully_connected':
                return 1.0;
            case 'small_world':
                return 4 / n; // Assuming k=4 in Watts-Strogatz
            case 'scale_free':
                return 2 / n; // Assuming m=2 in Barabási-Albert
            default:
                return 0.5; // Default estimate
        }
    }

    calculateTrendline(points) {
        const n = points.length;
        const sumX = points.reduce((sum, p) => sum + p.x, 0);
        const sumY = points.reduce((sum, p) => sum + p.y, 0);
        const sumXY = points.reduce((sum, p) => sum + p.x * p.y, 0);
        const sumX2 = points.reduce((sum, p) => sum + p.x * p.x, 0);
        
        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;
        
        return {
            slope,
            intercept,
            predict: (x) => slope * x + intercept
        };
    }

    calculateCorrelation(x, y) {
        const n = x.length;
        const sumX = x.reduce((a, b) => a + b);
        const sumY = y.reduce((a, b) => a + b);
        const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
        const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
        const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);
        
        const numerator = n * sumXY - sumX * sumY;
        const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
        
        return denominator === 0 ? 0 : numerator / denominator;
    }

    formatParamName(param) {
        const names = {
            'num_agents': 'Number of Agents',
            'num_rounds': 'Number of Rounds',
            'network_density': 'Network Density'
        };
        return names[param] || param;
    }

    formatMetricName(metric) {
        const names = {
            'final_cooperation': 'Final Cooperation Rate',
            'avg_cooperation': 'Average Cooperation',
            'convergence_time': 'Convergence Time',
            'stability': 'Stability Index'
        };
        return names[metric] || metric;
    }

    createCorrelationHeatmap(experiments, canvasId) {
        // Implementation for correlation heatmap
        // This would calculate correlations between different metrics
    }

    createDistributionChart(experiments, canvasId) {
        // Implementation for distribution analysis
        // This would show distributions of cooperation rates, scores, etc.
    }

    runHypothesisTests(experiments, containerId) {
        // Implementation for hypothesis testing
        // This would run statistical tests comparing different conditions
    }

    generateDetailedStatistics(experiments, containerId) {
        // Implementation for detailed statistics table
        // This would show comprehensive statistical measures
    }
}

// Export for use
window.AnalysisTools = AnalysisTools;