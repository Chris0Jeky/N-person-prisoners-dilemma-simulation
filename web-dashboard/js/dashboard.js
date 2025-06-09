/**
 * Main Dashboard Controller
 * Handles navigation, data management, and core functionality
 */

class Dashboard {
    constructor() {
        this.currentPage = 'overview';
        this.experiments = [];
        this.currentExperiment = null;
        this.theme = localStorage.getItem('theme') || 'light';
        this.charts = {};
        this.settings = {
            animations: true,
            use3D: false
        };
    }

    init() {
        this.setupEventListeners();
        this.applyTheme();
        this.loadCachedData();
        this.updateStats();
        
        // Initialize visualizations
        this.initCharts();
        
        // Load example data if first visit
        if (this.experiments.length === 0 && !localStorage.getItem('visited')) {
            localStorage.setItem('visited', 'true');
            // this.loadExampleData();
        }
    }

    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-item').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const page = e.currentTarget.dataset.page;
                this.navigateTo(page);
            });
        });

        // Theme toggle
        document.getElementById('themeBtn').addEventListener('click', () => {
            this.toggleTheme();
        });

        // File upload
        document.getElementById('uploadBtn').addEventListener('click', () => {
            document.getElementById('fileInput').click();
        });

        document.getElementById('fileInput').addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files);
        });

        // Settings
        document.getElementById('settingsBtn').addEventListener('click', () => {
            this.openSettings();
        });

        // Load example data
        const loadExampleBtn = document.getElementById('loadExampleBtn');
        if (loadExampleBtn) {
            loadExampleBtn.addEventListener('click', () => {
                this.loadExampleData();
            });
        }

        // Settings toggles
        document.getElementById('animationsToggle').addEventListener('change', (e) => {
            this.settings.animations = e.target.checked;
            this.saveSettings();
        });

        document.getElementById('3dToggle').addEventListener('change', (e) => {
            this.settings.use3D = e.target.checked;
            this.saveSettings();
            this.refreshVisualizations();
        });
    }

    navigateTo(page) {
        // Update nav
        document.querySelectorAll('.nav-item').forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.page === page) {
                btn.classList.add('active');
            }
        });

        // Update pages
        document.querySelectorAll('.page').forEach(p => {
            p.classList.remove('active');
        });
        document.getElementById(`${page}-page`).classList.add('active');

        this.currentPage = page;

        // Page-specific initialization
        switch(page) {
            case 'experiments':
                this.renderExperimentsList();
                break;
            case 'strategies':
                this.renderStrategies();
                break;
            case 'network':
                this.initNetworkVisualization();
                break;
            case 'analysis':
                this.initAnalysisTools();
                break;
        }
    }

    toggleTheme() {
        this.theme = this.theme === 'light' ? 'dark' : 'light';
        this.applyTheme();
        localStorage.setItem('theme', this.theme);
        
        // Update icon
        const icon = document.querySelector('#themeBtn i');
        icon.setAttribute('data-lucide', this.theme === 'light' ? 'moon' : 'sun');
        lucide.createIcons();
    }

    applyTheme() {
        document.documentElement.setAttribute('data-theme', this.theme);
        
        // Update charts theme
        if (this.charts) {
            Object.values(this.charts).forEach(chart => {
                if (chart.options) {
                    this.updateChartTheme(chart);
                    chart.update();
                }
            });
        }
    }

    updateChartTheme(chart) {
        const textColor = getComputedStyle(document.documentElement)
            .getPropertyValue('--text-primary').trim();
        const gridColor = getComputedStyle(document.documentElement)
            .getPropertyValue('--border').trim();

        chart.options.scales.x.ticks.color = textColor;
        chart.options.scales.y.ticks.color = textColor;
        chart.options.scales.x.grid.color = gridColor;
        chart.options.scales.y.grid.color = gridColor;
        chart.options.plugins.legend.labels.color = textColor;
    }

    async handleFileUpload(files) {
        // Check if any files are large
        const largeFiles = Array.from(files).filter(f => f.size > 1024 * 1024); // 1MB
        const useWorker = largeFiles.length > 0;
        
        if (useWorker && typeof Worker !== 'undefined') {
            // Use Web Worker for large files
            this.processFilesWithWorker(files);
        } else {
            // Process small files directly
            this.processFilesDirectly(files);
        }
    }

    async processFilesDirectly(files) {
        const dataLoader = new DataLoader();
        
        for (const file of files) {
            try {
                this.showNotification(`Loading ${file.name}...`, 'info');
                const experiments = await dataLoader.loadFromFile(file);
                
                if (Array.isArray(experiments)) {
                    experiments.forEach(exp => this.addExperiment(exp));
                } else {
                    this.addExperiment(experiments);
                }
                
                this.showNotification(`Successfully loaded ${file.name}`, 'success');
            } catch (error) {
                console.error('Error loading file:', error);
                this.showNotification(`Error loading ${file.name}: ${error.message}`, 'error');
            }
        }
        
        this.updateStats();
        this.refreshVisualizations();
    }

    async processFilesWithWorker(files) {
        // Create loading modal
        const loadingModal = this.createLoadingModal();
        document.body.appendChild(loadingModal);
        
        const worker = new Worker('js/file-worker.js');
        let filesProcessed = 0;
        
        // Handle worker messages
        worker.onmessage = (event) => {
            const { type, result, error, progress, message, id } = event.data;
            
            if (type === 'progress') {
                // Update progress bar
                const progressBar = loadingModal.querySelector('.progress-bar');
                const progressText = loadingModal.querySelector('.progress-text');
                if (progressBar) progressBar.style.width = `${progress}%`;
                if (progressText) progressText.textContent = message;
            } else if (type === 'success') {
                // Process result
                if (Array.isArray(result)) {
                    result.forEach(exp => this.addExperiment(exp));
                } else {
                    this.addExperiment(result);
                }
                
                filesProcessed++;
                if (filesProcessed >= files.length) {
                    // All files processed
                    worker.terminate();
                    document.body.removeChild(loadingModal);
                    this.updateStats();
                    this.refreshVisualizations();
                    this.showNotification(`Loaded ${filesProcessed} file(s) successfully`, 'success');
                }
            } else if (type === 'error') {
                // Handle error
                console.error('Worker error:', error);
                this.showNotification(`Error processing file: ${error}`, 'error');
                filesProcessed++;
                
                if (filesProcessed >= files.length) {
                    worker.terminate();
                    document.body.removeChild(loadingModal);
                }
            }
        };
        
        // Process each file
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const content = await file.text();
            
            worker.postMessage({
                id: i,
                type: 'parseFile',
                data: {
                    content: content,
                    filename: file.name,
                    fileSize: file.size
                }
            });
        }
    }

    createLoadingModal() {
        const modal = document.createElement('div');
        modal.className = 'loading-modal';
        modal.innerHTML = `
            <div class="loading-content">
                <h3>Processing Large File</h3>
                <div class="progress-container">
                    <div class="progress-bar" style="width: 0%"></div>
                </div>
                <p class="progress-text">Starting...</p>
            </div>
        `;
        
        // Add styles
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 9999;
        `;
        
        const content = modal.querySelector('.loading-content');
        content.style.cssText = `
            background: var(--bg-primary);
            padding: 2rem;
            border-radius: var(--radius-lg);
            min-width: 300px;
            text-align: center;
        `;
        
        const progressContainer = modal.querySelector('.progress-container');
        progressContainer.style.cssText = `
            background: var(--bg-secondary);
            height: 20px;
            border-radius: 10px;
            overflow: hidden;
            margin: 1rem 0;
        `;
        
        const progressBar = modal.querySelector('.progress-bar');
        progressBar.style.cssText = `
            background: var(--accent);
            height: 100%;
            transition: width 0.3s ease;
        `;
        
        return modal;
    }

    addExperiment(data) {
        // Standardize experiment structure
        const experiment = {
            id: data.id || Date.now(),
            name: data.scenario_name || data.experiment_name || 'Unnamed Experiment',
            timestamp: data.timestamp || new Date().toISOString(),
            config: data.config || data,
            results: data.results || data,
            metrics: this.calculateMetrics(data)
        };
        
        this.experiments.push(experiment);
        this.saveCachedData();
    }

    calculateMetrics(data) {
        const metrics = {
            totalAgents: 0,
            avgCooperation: 0,
            finalCooperation: 0,
            cooperationTrend: [],
            strategyScores: {},
            networkType: data.network_type || 'unknown'
        };

        // Extract metrics based on data structure
        if (data.results && Array.isArray(data.results)) {
            // Calculate from round data
            const rounds = data.results;
            metrics.cooperationTrend = rounds.map(round => 
                round.cooperation_rate || 0
            );
            metrics.finalCooperation = metrics.cooperationTrend[metrics.cooperationTrend.length - 1];
            metrics.avgCooperation = metrics.cooperationTrend.reduce((a, b) => a + b, 0) / metrics.cooperationTrend.length;
        }

        return metrics;
    }

    loadCachedData() {
        const cached = localStorage.getItem('experiments');
        if (cached) {
            try {
                this.experiments = JSON.parse(cached);
            } catch (e) {
                console.error('Error loading cached data:', e);
            }
        }
    }

    saveCachedData() {
        localStorage.setItem('experiments', JSON.stringify(this.experiments));
    }

    clearCache() {
        if (confirm('This will remove all loaded experiments. Continue?')) {
            this.experiments = [];
            localStorage.removeItem('experiments');
            this.updateStats();
            this.refreshVisualizations();
            this.showNotification('Cache cleared', 'info');
        }
    }

    async loadExampleData() {
        // Check if example data already exists
        const existingExample = this.experiments.find(exp => exp.isExample);
        if (existingExample) {
            this.showNotification('Example data already loaded', 'info');
            return;
        }
        
        try {
            const dataLoader = new DataLoader();
            const exampleData = await dataLoader.loadExampleData();
            
            if (exampleData && exampleData.length > 0) {
                exampleData.forEach(exp => this.addExperiment(exp));
                this.updateStats();
                this.refreshVisualizations();
                this.showNotification(`Loaded ${exampleData.length} example scenarios`, 'success');
            } else {
                this.showNotification('No example data available', 'warning');
            }
        } catch (error) {
            console.error('Error loading example data:', error);
            this.showNotification('Failed to load example data', 'error');
        }
    }


    updateStats() {
        // Calculate aggregate statistics
        const totalAgents = this.experiments.reduce((sum, exp) => 
            sum + (exp.config.num_agents || 0), 0
        );
        
        const avgCooperation = this.experiments.length > 0
            ? this.experiments.reduce((sum, exp) => 
                sum + (exp.metrics.avgCooperation || 0), 0) / this.experiments.length
            : 0;
        
        // Update UI
        document.getElementById('totalAgents').textContent = totalAgents || '-';
        document.getElementById('avgCooperation').textContent = 
            avgCooperation ? `${(avgCooperation * 100).toFixed(1)}%` : '-';
        document.getElementById('totalExperiments').textContent = this.experiments.length;
        
        // Find best strategy
        const strategyScores = {};
        this.experiments.forEach(exp => {
            Object.entries(exp.metrics.strategyScores || {}).forEach(([strategy, score]) => {
                if (!strategyScores[strategy]) strategyScores[strategy] = [];
                strategyScores[strategy].push(score);
            });
        });
        
        let bestStrategy = '-';
        let bestScore = -Infinity;
        Object.entries(strategyScores).forEach(([strategy, scores]) => {
            const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
            if (avgScore > bestScore) {
                bestScore = avgScore;
                bestStrategy = strategy.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            }
        });
        
        document.getElementById('bestStrategy').textContent = bestStrategy;
    }

    initCharts() {
        // Initialize cooperation evolution chart
        const cooperationCtx = document.getElementById('cooperationChart');
        if (cooperationCtx) {
            this.charts.cooperation = new Chart(cooperationCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: []
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    aspectRatio: 2,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: {
                                usePointStyle: true,
                                padding: 15,
                                font: {
                                    size: 12
                                }
                            }
                        },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            padding: 12,
                            cornerRadius: 8,
                            titleFont: {
                                size: 14
                            },
                            bodyFont: {
                                size: 12
                            }
                        },
                        zoom: {
                            zoom: {
                                wheel: {
                                    enabled: true,
                                },
                                pinch: {
                                    enabled: true
                                },
                                mode: 'x',
                            },
                            pan: {
                                enabled: true,
                                mode: 'x',
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Round'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Cooperation Rate'
                            },
                            min: 0,
                            max: 1,
                            ticks: {
                                callback: function(value) {
                                    return (value * 100).toFixed(0) + '%';
                                },
                                stepSize: 0.2
                            }
                        }
                    }
                }
            });
        }

        // Initialize strategy performance chart
        const strategyCtx = document.getElementById('strategyChart');
        if (strategyCtx) {
            this.charts.strategy = new Chart(strategyCtx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Average Score',
                        data: [],
                        backgroundColor: getComputedStyle(document.documentElement)
                            .getPropertyValue('--chart-1').trim()
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Average Score'
                            }
                        }
                    }
                }
            });
        }

        this.refreshVisualizations();
    }

    refreshVisualizations() {
        if (this.experiments.length === 0) return;

        // Update cooperation evolution chart
        if (this.charts.cooperation) {
            const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];
            const datasets = this.experiments.slice(0, 5).map((exp, i) => {
                const trend = exp.metrics.cooperationTrend || [];
                // Downsample data if too many points
                const data = trend.length > 100 ? 
                    trend.filter((_, idx) => idx % Math.ceil(trend.length / 100) === 0) : 
                    trend;
                
                return {
                    label: exp.name,
                    data: data,
                    borderColor: colors[i % colors.length],
                    backgroundColor: colors[i % colors.length] + '20',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    tension: 0.3,
                    fill: false
                };
            });

            const maxLength = Math.max(...datasets.map(d => d.data.length));
            this.charts.cooperation.data = {
                labels: Array.from({length: maxLength}, (_, i) => 
                    i % 10 === 0 ? (i * (100 / maxLength)).toFixed(0) : ''
                ),
                datasets: datasets
            };
            this.charts.cooperation.update('none');
        }

        // Update strategy performance
        this.updateStrategyChart();
    }

    updateStrategyChart() {
        // Aggregate strategy performance across all experiments
        const strategyData = {};
        
        this.experiments.forEach(exp => {
            if (exp.config.agent_strategies) {
                Object.entries(exp.config.agent_strategies).forEach(([strategy, count]) => {
                    if (!strategyData[strategy]) {
                        strategyData[strategy] = { total: 0, count: 0 };
                    }
                    strategyData[strategy].total += exp.metrics.avgCooperation || 0;
                    strategyData[strategy].count += 1;
                });
            }
        });

        const labels = Object.keys(strategyData);
        const data = labels.map(strategy => 
            strategyData[strategy].total / strategyData[strategy].count
        );

        if (this.charts.strategy) {
            this.charts.strategy.data = {
                labels: labels.map(s => s.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())),
                datasets: [{
                    label: 'Average Cooperation Rate',
                    data: data,
                    backgroundColor: labels.map((_, i) => 
                        getComputedStyle(document.documentElement)
                            .getPropertyValue(`--chart-${(i % 8) + 1}`).trim()
                    )
                }]
            };
            this.charts.strategy.update();
        }
    }

    renderExperimentsList() {
        const container = document.getElementById('experimentsList');
        
        if (this.experiments.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <i data-lucide="inbox"></i>
                    <h3>No experiments loaded</h3>
                    <p>Upload experiment results or load example data to get started</p>
                    <button class="btn btn-secondary" onclick="document.getElementById('uploadBtn').click()">
                        <i data-lucide="upload"></i>
                        Upload Results
                    </button>
                </div>
            `;
        } else {
            container.innerHTML = this.experiments.map(exp => `
                <div class="experiment-item">
                    <div onclick="dashboard.selectExperiment('${exp.id}')" style="flex: 1; cursor: pointer;">
                        <h3>${exp.name}</h3>
                        <div style="display: flex; gap: 2rem; margin-top: 0.5rem; color: var(--text-secondary); font-size: 0.875rem;">
                            <span><i data-lucide="users" style="width: 16px; height: 16px; display: inline;"></i> ${exp.config.num_agents || '-'} agents</span>
                            <span><i data-lucide="refresh-cw" style="width: 16px; height: 16px; display: inline;"></i> ${exp.config.num_rounds || '-'} rounds</span>
                            <span><i data-lucide="network" style="width: 16px; height: 16px; display: inline;"></i> ${exp.metrics.networkType}</span>
                            <span><i data-lucide="trending-up" style="width: 16px; height: 16px; display: inline;"></i> ${(exp.metrics.avgCooperation * 100).toFixed(1)}% cooperation</span>
                        </div>
                    </div>
                    <div class="experiment-actions">
                        <button class="icon-btn-sm" onclick="dashboard.exportExperiment('${exp.id}')" title="Export experiment">
                            <i data-lucide="download"></i>
                        </button>
                        <button class="icon-btn-sm" onclick="dashboard.deleteExperiment('${exp.id}')" title="Delete experiment">
                            <i data-lucide="trash-2"></i>
                        </button>
                    </div>
                </div>
            `).join('');
        }
        
        lucide.createIcons();
    }

    selectExperiment(id) {
        this.currentExperiment = this.experiments.find(exp => exp.id == id);
        this.navigateTo('analysis');
    }

    deleteExperiment(id) {
        if (confirm('Delete this experiment?')) {
            this.experiments = this.experiments.filter(exp => exp.id != id);
            if (this.currentExperiment && this.currentExperiment.id == id) {
                this.currentExperiment = null;
            }
            this.saveCachedData();
            this.updateStats();
            this.refreshVisualizations();
            this.renderExperimentsList();
            this.showNotification('Experiment deleted', 'info');
        }
    }

    exportExperiment(id) {
        const experiment = this.experiments.find(exp => exp.id == id);
        if (!experiment) return;
        
        const dataStr = JSON.stringify(experiment, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = `${experiment.name.replace(/[^a-z0-9]/gi, '_')}_${new Date().toISOString().split('T')[0]}.json`;
        link.click();
        
        URL.revokeObjectURL(url);
        this.showNotification('Experiment exported', 'success');
    }
    
    exportAllExperiments() {
        if (this.experiments.length === 0) {
            this.showNotification('No experiments to export', 'info');
            return;
        }
        
        const dataStr = JSON.stringify({
            experiments: this.experiments,
            exportDate: new Date().toISOString(),
            version: '1.0'
        }, null, 2);
        
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = `npd_experiments_${new Date().toISOString().split('T')[0]}.json`;
        link.click();
        
        URL.revokeObjectURL(url);
        this.showNotification(`Exported ${this.experiments.length} experiments`, 'success');
    }

    renderStrategies() {
        // Strategy definitions
        const strategies = [
            { id: 'tit_for_tat', name: 'Tit for Tat', icon: '🤝', color: '--chart-1',
              description: 'Cooperates based on proportion of neighbors who cooperated' },
            { id: 'always_cooperate', name: 'Always Cooperate', icon: '😇', color: '--chart-2',
              description: 'Always chooses to cooperate' },
            { id: 'always_defect', name: 'Always Defect', icon: '😈', color: '--chart-4',
              description: 'Always chooses to defect' },
            { id: 'q_learning', name: 'Q-Learning', icon: '🧠', color: '--chart-5',
              description: 'Learns optimal strategy through reinforcement' },
            { id: 'pavlov', name: 'Pavlov', icon: '🔄', color: '--chart-6',
              description: 'Win-stay, lose-shift strategy' },
            { id: 'generous_tit_for_tat', name: 'Generous TFT', icon: '💝', color: '--chart-3',
              description: 'Forgives defection occasionally' },
            { id: 'wolf_phc', name: 'WoLF-PHC', icon: '🐺', color: '--chart-7',
              description: 'Adaptive learning with variable rates' },
            { id: 'hysteretic_q', name: 'Hysteretic Q', icon: '📈', color: '--chart-8',
              description: 'Optimistic Q-learning variant' }
        ];

        const container = document.getElementById('strategiesGrid');
        container.innerHTML = strategies.map(strategy => `
            <div class="strategy-card">
                <div class="strategy-header">
                    <div class="strategy-icon" style="background-color: var(${strategy.color})20; color: var(${strategy.color})">
                        ${strategy.icon}
                    </div>
                    <h3>${strategy.name}</h3>
                </div>
                <p style="color: var(--text-secondary); font-size: 0.875rem; margin-bottom: 1rem;">
                    ${strategy.description}
                </p>
                <div style="display: flex; justify-content: space-between; font-size: 0.875rem;">
                    <span>Avg Performance:</span>
                    <span style="font-weight: 600;">${this.getStrategyPerformance(strategy.id)}%</span>
                </div>
            </div>
        `).join('');
    }

    getStrategyPerformance(strategyId) {
        let total = 0;
        let count = 0;
        
        this.experiments.forEach(exp => {
            if (exp.config.agent_strategies && exp.config.agent_strategies[strategyId]) {
                total += exp.metrics.avgCooperation || 0;
                count++;
            }
        });
        
        return count > 0 ? (total / count * 100).toFixed(1) : '-';
    }

    initNetworkVisualization() {
        const container = document.getElementById('networkVisualization');
        if (!container) return;
        
        if (this.currentExperiment || this.experiments.length > 0) {
            const viz = new Visualizations();
            const experiment = this.currentExperiment || this.experiments[0];
            viz.createNetworkVisualization('networkVisualization', experiment);
            
            // Add network type selector
            const networkSelector = document.getElementById('networkTypeSelector');
            if (networkSelector) {
                networkSelector.innerHTML = `
                    <label>Network Type: 
                        <select id="networkType" class="network-select">
                            <option value="fully_connected">Fully Connected</option>
                            <option value="small_world">Small World</option>
                            <option value="scale_free">Scale-Free</option>
                        </select>
                    </label>
                    <button class="btn btn-secondary" onclick="dashboard.regenerateNetwork()">
                        <i data-lucide="refresh-cw"></i> Regenerate
                    </button>
                `;
                
                document.getElementById('networkType').value = experiment.config.network_type || 'fully_connected';
                lucide.createIcons();
            }
        } else {
            container.innerHTML = '<p class="placeholder">Load experiments to view network visualization</p>';
        }
    }

    regenerateNetwork() {
        const networkType = document.getElementById('networkType').value;
        const experiment = this.currentExperiment || this.experiments[0];
        
        // Update experiment network type temporarily for visualization
        const modifiedExp = {
            ...experiment,
            config: {
                ...experiment.config,
                network_type: networkType
            }
        };
        
        const viz = new Visualizations();
        viz.createNetworkVisualization('networkVisualization', modifiedExp);
    }

    initAnalysisTools() {
        const container = document.getElementById('analysisContainer');
        if (!container) return;

        if (this.experiments.length > 0) {
            // Create tabs for different analysis views
            container.innerHTML = `
                <div class="analysis-tabs">
                    <button class="tab-btn active" onclick="dashboard.showAnalysisTab('comparison')">
                        <i data-lucide="git-compare"></i> Comparison
                    </button>
                    <button class="tab-btn" onclick="dashboard.showAnalysisTab('sensitivity')">
                        <i data-lucide="sliders"></i> Sensitivity
                    </button>
                    <button class="tab-btn" onclick="dashboard.showAnalysisTab('statistics')">
                        <i data-lucide="bar-chart-3"></i> Statistics
                    </button>
                    <button class="tab-btn" onclick="dashboard.showAnalysisTab('visualizations')">
                        <i data-lucide="eye"></i> Advanced Viz
                    </button>
                </div>
                
                <div id="analysisContent" class="analysis-content">
                    <!-- Content will be loaded here -->
                </div>
            `;
            
            lucide.createIcons();
            this.showAnalysisTab('comparison');
        } else {
            container.innerHTML = `
                <div class="empty-state">
                    <i data-lucide="bar-chart"></i>
                    <h3>No data to analyze</h3>
                    <p>Load experiments to access analysis tools</p>
                </div>
            `;
            lucide.createIcons();
        }
    }

    showAnalysisTab(tab) {
        // Update active tab
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
            if (btn.textContent.toLowerCase().includes(tab.substring(0, 4))) {
                btn.classList.add('active');
            }
        });

        const content = document.getElementById('analysisContent');
        const analysisTools = new AnalysisTools();
        const visualizations = new Visualizations();

        switch(tab) {
            case 'comparison':
                content.innerHTML = '<div id="comparison-interface"></div>';
                analysisTools.createComparisonInterface('comparison-interface', this.experiments);
                break;
                
            case 'sensitivity':
                content.innerHTML = '<div id="sensitivity-analysis"></div>';
                analysisTools.createParameterSensitivityAnalysis('sensitivity-analysis', this.experiments);
                break;
                
            case 'statistics':
                content.innerHTML = '<div id="statistical-dashboard"></div>';
                analysisTools.createStatisticalDashboard('statistical-dashboard', this.experiments);
                break;
                
            case 'visualizations':
                content.innerHTML = `
                    <div class="viz-grid">
                        <div class="viz-card">
                            <h3>Strategy Heatmap</h3>
                            <div id="strategy-heatmap" style="height: 400px;"></div>
                        </div>
                        <div class="viz-card">
                            <h3>3D Parameter Space</h3>
                            <div id="param-3d" style="height: 400px;"></div>
                        </div>
                        <div class="viz-card">
                            <h3>Strategy Radar</h3>
                            <canvas id="radar-chart" width="400" height="400"></canvas>
                        </div>
                        <div class="viz-card">
                            <h3>Animated Evolution</h3>
                            <div id="animated-series" style="height: 400px;"></div>
                        </div>
                    </div>
                `;
                
                // Create visualizations
                setTimeout(() => {
                    visualizations.createStrategyHeatmap('strategy-heatmap', this.experiments);
                    visualizations.create3DParameterPlot('param-3d', this.experiments);
                    visualizations.createStrategyRadarChart('radar-chart', this.experiments);
                    if (this.currentExperiment || this.experiments[0]) {
                        visualizations.createAnimatedTimeSeries('animated-series', 
                            this.currentExperiment || this.experiments[0]);
                    }
                }, 100);
                break;
        }
    }

    resetZoom(chartId) {
        if (this.charts[chartId.replace('Chart', '')]) {
            this.charts[chartId.replace('Chart', '')].resetZoom();
        }
    }

    openSettings() {
        document.getElementById('settingsModal').classList.add('active');
    }

    closeSettings() {
        document.getElementById('settingsModal').classList.remove('active');
    }

    saveSettings() {
        localStorage.setItem('settings', JSON.stringify(this.settings));
    }

    showNotification(message, type = 'info') {
        // Simple notification implementation
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 1rem 1.5rem;
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            box-shadow: 0 4px 12px var(--shadow);
            z-index: 1000;
            animation: slideInRight 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
}

// Add animation keyframes
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOutRight {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);