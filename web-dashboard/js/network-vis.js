/**
 * Network Visualization using D3.js
 * Creates interactive network graphs for different topologies
 */

class NetworkVisualization {
    constructor(containerId, experiment) {
        this.container = document.getElementById(containerId);
        this.experiment = experiment;
        this.width = this.container.clientWidth;
        this.height = 600;
        this.simulation = null;
        
        this.init();
    }

    init() {
        // Clear existing content
        this.container.innerHTML = '';
        
        // Create SVG
        this.svg = d3.select(this.container)
            .append('svg')
            .attr('width', this.width)
            .attr('height', this.height);
        
        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on('zoom', (event) => {
                this.g.attr('transform', event.transform);
            });
        
        this.svg.call(zoom);
        
        // Create main group
        this.g = this.svg.append('g');
        
        // Define arrow markers for directed edges
        this.svg.append('defs').append('marker')
            .attr('id', 'arrowhead')
            .attr('viewBox', '-0 -5 10 10')
            .attr('refX', 20)
            .attr('refY', 0)
            .attr('orient', 'auto')
            .attr('markerWidth', 10)
            .attr('markerHeight', 10)
            .append('path')
            .attr('d', 'M 0,-5 L 10,0 L 0,5')
            .attr('fill', '#999');
        
        // Generate network based on type
        const networkType = document.getElementById('networkType').value;
        this.generateNetwork(networkType);
    }

    generateNetwork(type) {
        let nodes = [];
        let links = [];
        const numNodes = 30;
        
        // Create nodes
        for (let i = 0; i < numNodes; i++) {
            nodes.push({
                id: i,
                name: `Agent ${i}`,
                strategy: this.getRandomStrategy(),
                cooperation: Math.random()
            });
        }
        
        // Create links based on network type
        switch(type) {
            case 'fully_connected':
                for (let i = 0; i < numNodes; i++) {
                    for (let j = i + 1; j < numNodes; j++) {
                        links.push({ source: i, target: j });
                    }
                }
                break;
                
            case 'small_world':
                // Watts-Strogatz model
                const k = 4; // Each node connected to k nearest neighbors
                const p = 0.3; // Rewiring probability
                
                // Create ring lattice
                for (let i = 0; i < numNodes; i++) {
                    for (let j = 1; j <= k/2; j++) {
                        const target = (i + j) % numNodes;
                        links.push({ source: i, target: target });
                    }
                }
                
                // Rewire edges
                links = links.map(link => {
                    if (Math.random() < p) {
                        let newTarget;
                        do {
                            newTarget = Math.floor(Math.random() * numNodes);
                        } while (newTarget === link.source);
                        return { source: link.source, target: newTarget };
                    }
                    return link;
                });
                break;
                
            case 'scale_free':
                // Barabási-Albert model
                const m = 2; // Number of edges to attach from new node
                
                // Start with m+1 fully connected nodes
                for (let i = 0; i <= m; i++) {
                    for (let j = i + 1; j <= m; j++) {
                        links.push({ source: i, target: j });
                    }
                }
                
                // Add remaining nodes using preferential attachment
                for (let i = m + 1; i < numNodes; i++) {
                    const degrees = new Array(i).fill(0);
                    links.forEach(link => {
                        degrees[link.source]++;
                        degrees[link.target]++;
                    });
                    
                    const totalDegree = degrees.reduce((a, b) => a + b, 0);
                    const targets = new Set();
                    
                    while (targets.size < m) {
                        let r = Math.random() * totalDegree;
                        let sum = 0;
                        for (let j = 0; j < i; j++) {
                            sum += degrees[j];
                            if (sum > r && !targets.has(j)) {
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
        
        this.renderNetwork(nodes, links);
    }

    getRandomStrategy() {
        const strategies = ['tit_for_tat', 'always_cooperate', 'always_defect', 'q_learning', 'pavlov'];
        return strategies[Math.floor(Math.random() * strategies.length)];
    }

    getStrategyColor(strategy) {
        const colors = {
            'tit_for_tat': '#3b82f6',
            'always_cooperate': '#10b981',
            'always_defect': '#ef4444',
            'q_learning': '#8b5cf6',
            'pavlov': '#f59e0b',
            'generous_tit_for_tat': '#ec4899',
            'wolf_phc': '#06b6d4',
            'hysteretic_q': '#f97316'
        };
        return colors[strategy] || '#94a3b8';
    }

    renderNetwork(nodes, links) {
        // Create force simulation
        this.simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(50))
            .force('charge', d3.forceManyBody().strength(-100))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(20));
        
        // Create links
        const link = this.g.append('g')
            .attr('class', 'links')
            .selectAll('line')
            .data(links)
            .enter().append('line')
            .attr('stroke', '#999')
            .attr('stroke-opacity', 0.6)
            .attr('stroke-width', 1);
        
        // Create nodes
        const node = this.g.append('g')
            .attr('class', 'nodes')
            .selectAll('g')
            .data(nodes)
            .enter().append('g')
            .call(this.drag(this.simulation));
        
        // Add circles to nodes
        node.append('circle')
            .attr('r', d => 10 + d.cooperation * 10)
            .attr('fill', d => this.getStrategyColor(d.strategy))
            .attr('stroke', '#fff')
            .attr('stroke-width', 2);
        
        // Add labels
        node.append('text')
            .text(d => d.id)
            .attr('x', 0)
            .attr('y', 0)
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'central')
            .attr('fill', '#fff')
            .attr('font-size', '10px')
            .attr('font-weight', 'bold');
        
        // Add tooltips
        node.append('title')
            .text(d => `${d.name}\nStrategy: ${d.strategy}\nCooperation: ${(d.cooperation * 100).toFixed(1)}%`);
        
        // Update positions on tick
        this.simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            node
                .attr('transform', d => `translate(${d.x},${d.y})`);
        });
        
        // Add legend
        this.addLegend();
    }

    drag(simulation) {
        function dragstarted(event) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }
        
        function dragged(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }
        
        function dragended(event) {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }
        
        return d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended);
    }

    addLegend() {
        const strategies = [
            { name: 'Tit for Tat', color: '#3b82f6' },
            { name: 'Always Cooperate', color: '#10b981' },
            { name: 'Always Defect', color: '#ef4444' },
            { name: 'Q-Learning', color: '#8b5cf6' },
            { name: 'Pavlov', color: '#f59e0b' }
        ];
        
        const legend = this.svg.append('g')
            .attr('class', 'legend')
            .attr('transform', 'translate(20, 20)');
        
        const legendItem = legend.selectAll('.legend-item')
            .data(strategies)
            .enter().append('g')
            .attr('class', 'legend-item')
            .attr('transform', (d, i) => `translate(0, ${i * 25})`);
        
        legendItem.append('circle')
            .attr('r', 8)
            .attr('fill', d => d.color);
        
        legendItem.append('text')
            .attr('x', 20)
            .attr('y', 0)
            .attr('dominant-baseline', 'central')
            .attr('font-size', '12px')
            .attr('fill', 'var(--text-primary)')
            .text(d => d.name);
    }

    reset() {
        if (this.simulation) {
            this.simulation.stop();
        }
        this.init();
    }
}

// Make available globally
window.NetworkVisualization = NetworkVisualization;

// Reset function for button
window.dashboard = window.dashboard || {};
window.dashboard.resetNetwork = function() {
    const viz = new NetworkVisualization('networkVisualization');
};