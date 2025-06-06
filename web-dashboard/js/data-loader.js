/**
 * Data Loader Module
 * Handles loading and parsing experiment data from various sources
 */

class DataLoader {
    constructor() {
        this.supportedFormats = ['.json', '.csv'];
    }

    async loadFromFile(file) {
        const extension = file.name.substring(file.name.lastIndexOf('.'));
        
        if (!this.supportedFormats.includes(extension)) {
            throw new Error(`Unsupported file format: ${extension}`);
        }

        const text = await file.text();

        switch (extension) {
            case '.json':
                return this.parseJSON(text);
            case '.csv':
                return this.parseCSV(text);
            default:
                throw new Error(`Unknown format: ${extension}`);
        }
    }

    parseJSON(text) {
        try {
            const data = JSON.parse(text);
            return this.normalizeData(data);
        } catch (error) {
            throw new Error('Invalid JSON format: ' + error.message);
        }
    }

    parseCSV(text) {
        const lines = text.trim().split('\n');
        if (lines.length < 2) {
            throw new Error('CSV file is empty or has no data');
        }

        const headers = lines[0].split(',').map(h => h.trim());
        const data = [];

        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',').map(v => v.trim());
            const row = {};
            
            headers.forEach((header, index) => {
                const value = values[index];
                // Try to parse numbers
                row[header] = isNaN(value) ? value : parseFloat(value);
            });
            
            data.push(row);
        }

        return this.normalizeCSVData(data);
    }

    normalizeData(data) {
        // Handle different data structures
        if (Array.isArray(data)) {
            return data.map(item => this.normalizeExperiment(item));
        } else if (data.experiments) {
            return data.experiments.map(item => this.normalizeExperiment(item));
        } else {
            return [this.normalizeExperiment(data)];
        }
    }

    normalizeExperiment(exp) {
        return {
            id: exp.id || Date.now() + Math.random(),
            name: exp.scenario_name || exp.experiment_name || exp.name || 'Unnamed',
            timestamp: exp.timestamp || new Date().toISOString(),
            config: {
                num_agents: exp.num_agents || exp.config?.num_agents || 0,
                num_rounds: exp.num_rounds || exp.config?.num_rounds || 0,
                network_type: exp.network_type || exp.config?.network_type || 'unknown',
                interaction_mode: exp.interaction_mode || exp.config?.interaction_mode || 'neighborhood',
                agent_strategies: exp.agent_strategies || exp.config?.agent_strategies || {}
            },
            results: this.extractResults(exp),
            raw: exp
        };
    }

    extractResults(exp) {
        // Try different result formats
        if (exp.results && Array.isArray(exp.results)) {
            return exp.results;
        } else if (exp.rounds && Array.isArray(exp.rounds)) {
            return exp.rounds;
        } else if (exp.data && Array.isArray(exp.data)) {
            return exp.data;
        } else {
            // Generate synthetic results if none found
            return [];
        }
    }

    normalizeCSVData(rows) {
        // Group by experiment/scenario if possible
        const experiments = {};
        
        rows.forEach(row => {
            const expId = row.experiment_id || row.scenario || 'default';
            
            if (!experiments[expId]) {
                experiments[expId] = {
                    id: expId,
                    name: row.experiment_name || row.scenario_name || expId,
                    config: {
                        num_agents: row.num_agents || 0,
                        num_rounds: row.num_rounds || 0,
                        network_type: row.network_type || 'unknown'
                    },
                    results: []
                };
            }
            
            experiments[expId].results.push({
                round: row.round || experiments[expId].results.length,
                cooperation_rate: row.cooperation_rate || row.coop_rate || 0,
                avg_score: row.avg_score || row.average_score || 0,
                ...row
            });
        });
        
        return Object.values(experiments);
    }

    async loadFromURL(url) {
        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const contentType = response.headers.get('content-type');
            
            if (contentType.includes('json')) {
                const data = await response.json();
                return this.normalizeData(data);
            } else if (contentType.includes('csv') || url.endsWith('.csv')) {
                const text = await response.text();
                return this.parseCSV(text);
            } else {
                throw new Error('Unsupported content type: ' + contentType);
            }
        } catch (error) {
            throw new Error('Failed to load from URL: ' + error.message);
        }
    }

    // Load example data from the project
    async loadExampleData() {
        const exampleURLs = [
            '../results/Baseline_QL_BasicState/run_00/experiment_results_rounds.csv',
            '../analysis_results/comparative_analysis.csv',
            '../results/generated_scenarios/selected_scenarios.json'
        ];

        const loadedData = [];

        for (const url of exampleURLs) {
            try {
                const data = await this.loadFromURL(url);
                loadedData.push(...data);
            } catch (error) {
                console.warn(`Failed to load example from ${url}:`, error);
            }
        }

        if (loadedData.length === 0) {
            // Return synthetic data as fallback
            return this.generateSyntheticData();
        }

        return loadedData;
    }

    generateSyntheticData() {
        const scenarios = [
            {
                name: 'Cooperative Environment',
                strategies: { 'tit_for_tat': 15, 'always_cooperate': 10, 'generous_tit_for_tat': 5 },
                trend: (r) => 0.7 + 0.2 * (r / 100)
            },
            {
                name: 'Competitive Environment',
                strategies: { 'always_defect': 10, 'q_learning': 10, 'wolf_phc': 10 },
                trend: (r) => 0.3 - 0.2 * (r / 100)
            },
            {
                name: 'Mixed Strategies',
                strategies: { 'tit_for_tat': 8, 'always_cooperate': 8, 'always_defect': 8, 'pavlov': 6 },
                trend: (r) => 0.5 + 0.1 * Math.sin(r / 10)
            },
            {
                name: 'Learning Agents Dominant',
                strategies: { 'q_learning': 10, 'hysteretic_q': 10, 'lra_q': 10 },
                trend: (r) => 0.4 + 0.3 * (r / 100) * (1 - Math.exp(-r / 20))
            }
        ];

        return scenarios.map((scenario, i) => ({
            id: `synthetic_${i}`,
            name: scenario.name,
            timestamp: new Date().toISOString(),
            config: {
                num_agents: 30,
                num_rounds: 100,
                network_type: ['fully_connected', 'small_world', 'scale_free'][i % 3],
                interaction_mode: 'neighborhood',
                agent_strategies: scenario.strategies
            },
            results: Array.from({ length: 100 }, (_, round) => ({
                round: round,
                cooperation_rate: Math.max(0, Math.min(1, 
                    scenario.trend(round) + (Math.random() - 0.5) * 0.1
                )),
                avg_score: 2 + Math.random() * 2
            }))
        }));
    }
}

// Export for use
window.DataLoader = DataLoader;