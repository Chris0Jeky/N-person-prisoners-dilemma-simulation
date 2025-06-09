/**
 * Advanced Visualizations Module
 * Interactive charts and visualizations for experiment data
 */

class Visualizations {
    constructor() {
        this.charts = {};
        this.colorSchemes = {
            cooperation: ['#ef4444', '#f97316', '#f59e0b', '#84cc16', '#10b981'],
            strategies: ['#3b82f6', '#8b5cf6', '#ec4899', '#f97316', '#10b981', '#06b6d4', '#84cc16', '#f59e0b', '#94a3b8'],
            network: ['#3b82f6', '#1e293b', '#64748b'],
            heatmap: ['#eff6ff', '#dbeafe', '#93c5fd', '#3b82f6', '#1e40af', '#1e3a8a']
        };
    }

    /**
     * Create an interactive cooperation evolution chart with advanced features
     */
    createCooperationEvolutionChart(containerId, experiments) {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Clear previous chart
        if (this.charts[containerId]) {
            this.charts[containerId].destroy();
        }

        const ctx = container.getContext('2d');
        
        // Prepare datasets with smooth curves
        const datasets = experiments.map((exp, idx) => {
            // Get primary strategy from experiment
            const strategies = Object.keys(exp.config.agent_strategies || {});
            const primaryStrategy = strategies[0] || 'unknown';
            const color = getStrategyColor(primaryStrategy);
            
            return {
                label: exp.name,
                data: exp.results.map(r => ({
                    x: r.round,
                    y: r.cooperation_rate * 100
                })),
                borderColor: color,
                backgroundColor: getStrategyColorWithOpacity(primaryStrategy, STRATEGY_COLORS.opacity.fill),
                tension: 0.3, // Smooth curves
                fill: false,
                pointRadius: 0,
                pointHoverRadius: 6,
                borderWidth: 2
            };
        });

        this.charts[containerId] = new Chart(ctx, {
            type: 'line',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                    },
                    title: {
                        display: true,
                        text: 'Evolution of Cooperation',
                        font: { size: 16, weight: 'bold' }
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                return `${context.dataset.label}: ${context.parsed.y.toFixed(1)}%`;
                            }
                        }
                    },
                    annotation: {
                        annotations: {
                            // Add threshold line
                            threshold: {
                                type: 'line',
                                yMin: 50,
                                yMax: 50,
                                borderColor: 'rgba(0, 0, 0, 0.3)',
                                borderWidth: 2,
                                borderDash: [5, 5],
                                label: {
                                    content: '50% Cooperation',
                                    enabled: true,
                                    position: 'end'
                                }
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        title: {
                            display: true,
                            text: 'Round'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Cooperation Rate (%)'
                        },
                        min: 0,
                        max: 100
                    }
                }
            }
        });
    }

    /**
     * Create an interactive strategy performance heatmap
     */
    createStrategyHeatmap(containerId, experiments) {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Calculate strategy performance matrix
        const strategies = new Set();
        const performanceData = {};

        experiments.forEach(exp => {
            Object.keys(exp.config.agent_strategies || {}).forEach(s => strategies.add(s));
        });

        const strategyList = Array.from(strategies);
        
        // Initialize performance matrix
        strategyList.forEach(s1 => {
            performanceData[s1] = {};
            strategyList.forEach(s2 => {
                performanceData[s1][s2] = [];
            });
        });

        // Calculate pairwise performance
        experiments.forEach(exp => {
            const finalResults = exp.results[exp.results.length - 1] || {};
            const strategies = Object.keys(exp.config.agent_strategies || {});
            
            strategies.forEach(s1 => {
                strategies.forEach(s2 => {
                    if (performanceData[s1] && performanceData[s1][s2]) {
                        performanceData[s1][s2].push(finalResults.cooperation_rate || 0);
                    }
                });
            });
        });

        // Average the performances
        const data = [];
        strategyList.forEach((s1, i) => {
            strategyList.forEach((s2, j) => {
                const values = performanceData[s1][s2];
                const avg = values.length > 0 ? values.reduce((a, b) => a + b) / values.length : 0;
                data.push({
                    x: j,
                    y: i,
                    v: avg * 100
                });
            });
        });

        // Create heatmap using D3.js
        const width = container.offsetWidth;
        const height = 400;
        const margin = { top: 50, right: 50, bottom: 100, left: 100 };
        
        // Clear previous
        d3.select(container).selectAll("*").remove();

        const svg = d3.select(container)
            .append("svg")
            .attr("width", width)
            .attr("height", height);

        const innerWidth = width - margin.left - margin.right;
        const innerHeight = height - margin.top - margin.bottom;

        const g = svg.append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        // Scales
        const xScale = d3.scaleBand()
            .domain(d3.range(strategyList.length))
            .range([0, innerWidth])
            .padding(0.05);

        const yScale = d3.scaleBand()
            .domain(d3.range(strategyList.length))
            .range([0, innerHeight])
            .padding(0.05);

        const colorScale = d3.scaleSequential()
            .domain([0, 100])
            .interpolator(d3.interpolateBlues);

        // Create cells
        g.selectAll(".cell")
            .data(data)
            .enter().append("rect")
            .attr("class", "cell")
            .attr("x", d => xScale(d.x))
            .attr("y", d => yScale(d.y))
            .attr("width", xScale.bandwidth())
            .attr("height", yScale.bandwidth())
            .attr("fill", d => colorScale(d.v))
            .attr("stroke", "#fff")
            .attr("stroke-width", 2)
            .on("mouseover", function(event, d) {
                // Tooltip
                const tooltip = d3.select("body").append("div")
                    .attr("class", "tooltip")
                    .style("position", "absolute")
                    .style("background", "rgba(0, 0, 0, 0.8)")
                    .style("color", "#fff")
                    .style("padding", "8px")
                    .style("border-radius", "4px")
                    .style("font-size", "12px")
                    .style("pointer-events", "none");

                tooltip.html(`${strategyList[d.y]} vs ${strategyList[d.x]}<br>Cooperation: ${d.v.toFixed(1)}%`)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
            })
            .on("mouseout", function() {
                d3.selectAll(".tooltip").remove();
            });

        // Add labels
        g.selectAll(".x-label")
            .data(strategyList)
            .enter().append("text")
            .attr("class", "x-label")
            .attr("x", (d, i) => xScale(i) + xScale.bandwidth() / 2)
            .attr("y", innerHeight + 20)
            .attr("text-anchor", "middle")
            .style("font-size", "12px")
            .text(d => STRATEGIES[d]?.name || d);

        g.selectAll(".y-label")
            .data(strategyList)
            .enter().append("text")
            .attr("class", "y-label")
            .attr("x", -10)
            .attr("y", (d, i) => yScale(i) + yScale.bandwidth() / 2)
            .attr("text-anchor", "end")
            .attr("alignment-baseline", "middle")
            .style("font-size", "12px")
            .text(d => STRATEGIES[d]?.name || d);

        // Title
        svg.append("text")
            .attr("x", width / 2)
            .attr("y", 20)
            .attr("text-anchor", "middle")
            .style("font-size", "16px")
            .style("font-weight", "bold")
            .text("Strategy Performance Heatmap");
    }

    /**
     * Create an animated network visualization
     */
    createNetworkVisualization(containerId, experiment) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const width = container.offsetWidth;
        const height = 500;

        // Clear previous
        d3.select(container).selectAll("*").remove();

        const svg = d3.select(container)
            .append("svg")
            .attr("width", width)
            .attr("height", height);

        // Generate network data based on network type
        const networkData = this.generateNetworkData(experiment);
        const numNodes = networkData.nodes.length;

        // Optimize for large networks
        const isLargeNetwork = numNodes > 100;
        const nodeRadius = isLargeNetwork ? 8 : 15;
        const linkDistance = isLargeNetwork ? 30 : 50;
        const chargeStrength = isLargeNetwork ? -50 : -100;

        // Create force simulation with optimized parameters
        const simulation = d3.forceSimulation(networkData.nodes)
            .force("link", d3.forceLink(networkData.links).id(d => d.id).distance(linkDistance))
            .force("charge", d3.forceManyBody().strength(chargeStrength))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(nodeRadius + 2));

        // Reduce simulation iterations for large networks
        if (isLargeNetwork) {
            simulation.alphaDecay(0.05); // Faster cooling
            simulation.velocityDecay(0.8); // More damping
        }

        // Add zoom behavior
        const g = svg.append("g");
        
        svg.call(d3.zoom()
            .extent([[0, 0], [width, height]])
            .scaleExtent([0.5, 4])
            .on("zoom", (event) => {
                g.attr("transform", event.transform);
            }));

        // Draw links
        const link = g.append("g")
            .selectAll("line")
            .data(networkData.links)
            .enter().append("line")
            .attr("stroke", "#64748b")
            .attr("stroke-opacity", 0.6)
            .attr("stroke-width", 2);

        // Draw nodes
        const node = g.append("g")
            .selectAll("circle")
            .data(networkData.nodes)
            .enter().append("circle")
            .attr("r", 15)
            .attr("fill", d => getStrategyColor(d.strategy))
            .attr("stroke", "#fff")
            .attr("stroke-width", 2)
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        // Add icons
        const nodeIcons = g.append("g")
            .selectAll("text")
            .data(networkData.nodes)
            .enter().append("text")
            .attr("text-anchor", "middle")
            .attr("alignment-baseline", "middle")
            .style("font-size", "16px")
            .style("user-select", "none")
            .style("pointer-events", "none")
            .text(d => STRATEGIES[d.strategy]?.icon || "?");

        // Add tooltips
        node.append("title")
            .text(d => `Agent ${d.id}\nStrategy: ${STRATEGIES[d.strategy]?.name || d.strategy}\nCooperation: ${(d.cooperationRate * 100).toFixed(1)}%`);

        // Update positions on tick
        simulation.on("tick", () => {
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);

            nodeIcons
                .attr("x", d => d.x)
                .attr("y", d => d.y);
        });

        // Drag functions
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }

        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }

        // Add play/pause button
        const controlsDiv = d3.select(container)
            .append("div")
            .style("position", "absolute")
            .style("top", "10px")
            .style("right", "10px");

        let playing = true;
        const playButton = controlsDiv.append("button")
            .attr("class", "control-button")
            .style("padding", "8px 16px")
            .style("background", "#3b82f6")
            .style("color", "#fff")
            .style("border", "none")
            .style("border-radius", "4px")
            .style("cursor", "pointer")
            .text("Pause")
            .on("click", () => {
                playing = !playing;
                if (playing) {
                    simulation.alpha(0.3).restart();
                    playButton.text("Pause");
                } else {
                    simulation.stop();
                    playButton.text("Play");
                }
            });
    }

    /**
     * Generate network data based on experiment configuration
     */
    generateNetworkData(experiment) {
        const numAgents = experiment.config.num_agents || 30;
        const networkType = experiment.config.network_type || 'fully_connected';
        const strategies = experiment.config.agent_strategies || {};
        
        // Create nodes
        const nodes = [];
        let agentId = 0;
        
        Object.entries(strategies).forEach(([strategy, count]) => {
            for (let i = 0; i < count; i++) {
                nodes.push({
                    id: agentId++,
                    strategy: strategy,
                    cooperationRate: Math.random() // Simulated
                });
            }
        });

        // Create links based on network type
        const links = [];
        
        switch (networkType) {
            case 'fully_connected':
                // Everyone connected to everyone
                for (let i = 0; i < nodes.length; i++) {
                    for (let j = i + 1; j < nodes.length; j++) {
                        links.push({ source: i, target: j });
                    }
                }
                break;
                
            case 'small_world':
                // Watts-Strogatz small world
                const k = 4; // Each node connected to k nearest neighbors
                const p = 0.1; // Rewiring probability
                
                // Create ring lattice
                for (let i = 0; i < nodes.length; i++) {
                    for (let j = 1; j <= k / 2; j++) {
                        const target = (i + j) % nodes.length;
                        links.push({ source: i, target: target });
                    }
                }
                
                // Rewire with probability p
                links.forEach(link => {
                    if (Math.random() < p) {
                        link.target = Math.floor(Math.random() * nodes.length);
                    }
                });
                break;
                
            case 'scale_free':
                // Barabási-Albert scale-free network
                const m = 2; // Number of edges to attach from new node
                
                // Start with m+1 fully connected nodes
                for (let i = 0; i <= m; i++) {
                    for (let j = i + 1; j <= m; j++) {
                        if (i < nodes.length && j < nodes.length) {
                            links.push({ source: i, target: j });
                        }
                    }
                }
                
                // Add remaining nodes using preferential attachment
                for (let i = m + 1; i < nodes.length; i++) {
                    const degrees = new Array(i).fill(0);
                    links.forEach(link => {
                        if (link.source < i) degrees[link.source]++;
                        if (link.target < i) degrees[link.target]++;
                    });
                    
                    const totalDegree = degrees.reduce((a, b) => a + b, 0);
                    const targets = new Set();
                    
                    while (targets.size < m && targets.size < i) {
                        let r = Math.random() * totalDegree;
                        let sum = 0;
                        for (let j = 0; j < i; j++) {
                            sum += degrees[j];
                            if (r <= sum) {
                                targets.add(j);
                                break;
                            }
                        }
                    }
                    
                    targets.forEach(target => {
                        links.push({ source: i, target: target });
                    });
                }
                break;
        }

        return { nodes, links };
    }

    /**
     * Create a 3D scatter plot for parameter exploration
     */
    create3DParameterPlot(containerId, experiments) {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Extract parameter dimensions
        const data = experiments.map(exp => ({
            x: exp.config.num_agents,
            y: exp.config.num_rounds,
            z: exp.results[exp.results.length - 1]?.cooperation_rate || 0,
            text: exp.name,
            marker: {
                size: 8,
                color: exp.results[exp.results.length - 1]?.cooperation_rate || 0,
                colorscale: 'Viridis',
                showscale: true
            }
        }));

        const layout = {
            title: 'Parameter Space Exploration',
            scene: {
                xaxis: { title: 'Number of Agents' },
                yaxis: { title: 'Number of Rounds' },
                zaxis: { title: 'Final Cooperation Rate' }
            },
            height: 500,
            margin: { l: 0, r: 0, b: 0, t: 40 }
        };

        Plotly.newPlot(container, [{
            type: 'scatter3d',
            mode: 'markers',
            ...data
        }], layout);
    }

    /**
     * Create an animated time series with playback controls
     */
    createAnimatedTimeSeries(containerId, experiment) {
        const container = document.getElementById(containerId);
        if (!container) return;

        let currentRound = 0;
        let animationId = null;
        let isPlaying = false;

        // Create controls
        const controls = document.createElement('div');
        controls.className = 'animation-controls';
        controls.innerHTML = `
            <button id="play-btn" class="control-btn">▶️ Play</button>
            <button id="reset-btn" class="control-btn">🔄 Reset</button>
            <input type="range" id="round-slider" min="0" max="${experiment.results.length - 1}" value="0" style="width: 200px;">
            <span id="round-label">Round: 0</span>
        `;
        container.appendChild(controls);

        // Create chart container
        const chartContainer = document.createElement('canvas');
        chartContainer.id = containerId + '_chart';
        container.appendChild(chartContainer);

        // Initialize chart with empty data
        const ctx = chartContainer.getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Cooperation Rate',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: '#3b82f6' + '20',
                    tension: 0.3,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 100
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Cooperation Rate'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Round'
                        }
                    }
                }
            }
        });

        // Update function
        const updateChart = (round) => {
            const data = experiment.results.slice(0, round + 1);
            chart.data.labels = data.map(r => r.round);
            chart.data.datasets[0].data = data.map(r => r.cooperation_rate);
            chart.update('none'); // No animation for smooth playback
            
            document.getElementById('round-slider').value = round;
            document.getElementById('round-label').textContent = `Round: ${round}`;
        };

        // Animation loop
        const animate = () => {
            if (currentRound < experiment.results.length - 1) {
                currentRound++;
                updateChart(currentRound);
                animationId = setTimeout(animate, 100); // 10 fps
            } else {
                isPlaying = false;
                document.getElementById('play-btn').textContent = '▶️ Play';
            }
        };

        // Event listeners
        document.getElementById('play-btn').addEventListener('click', () => {
            if (isPlaying) {
                clearTimeout(animationId);
                isPlaying = false;
                document.getElementById('play-btn').textContent = '▶️ Play';
            } else {
                isPlaying = true;
                document.getElementById('play-btn').textContent = '⏸️ Pause';
                animate();
            }
        });

        document.getElementById('reset-btn').addEventListener('click', () => {
            clearTimeout(animationId);
            isPlaying = false;
            currentRound = 0;
            updateChart(0);
            document.getElementById('play-btn').textContent = '▶️ Play';
        });

        document.getElementById('round-slider').addEventListener('input', (e) => {
            clearTimeout(animationId);
            isPlaying = false;
            currentRound = parseInt(e.target.value);
            updateChart(currentRound);
            document.getElementById('play-btn').textContent = '▶️ Play';
        });

        // Initialize with first frame
        updateChart(0);
    }

    /**
     * Create a radar chart for strategy comparison
     */
    createStrategyRadarChart(containerId, experiments) {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Calculate strategy metrics
        const strategies = new Set();
        const metrics = ['Cooperation', 'Score', 'Stability', 'Adaptability', 'Robustness'];
        
        experiments.forEach(exp => {
            Object.keys(exp.config.agent_strategies || {}).forEach(s => strategies.add(s));
        });

        // Generate random metrics for demonstration
        const datasets = Array.from(strategies).slice(0, 5).map((strategy, idx) => ({
            label: STRATEGIES[strategy]?.name || strategy,
            data: metrics.map(() => Math.random() * 0.6 + 0.4), // Random between 0.4 and 1
            borderColor: this.colorSchemes.strategies[idx],
            backgroundColor: this.colorSchemes.strategies[idx] + '40',
            pointBackgroundColor: this.colorSchemes.strategies[idx],
            pointBorderColor: '#fff',
            pointHoverBackgroundColor: '#fff',
            pointHoverBorderColor: this.colorSchemes.strategies[idx]
        }));

        const ctx = container.getContext('2d');
        
        if (this.charts[containerId]) {
            this.charts[containerId].destroy();
        }

        this.charts[containerId] = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: metrics,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Strategy Performance Comparison',
                        font: { size: 16, weight: 'bold' }
                    }
                },
                elements: {
                    line: {
                        borderWidth: 3
                    }
                },
                scales: {
                    r: {
                        angleLines: {
                            display: false
                        },
                        suggestedMin: 0,
                        suggestedMax: 1
                    }
                }
            }
        });
    }
}

// Export for use
window.Visualizations = Visualizations;