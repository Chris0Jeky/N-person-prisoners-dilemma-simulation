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
        const lines = text.trim().split('\n').filter(line => line.trim());
        if (lines.length < 2) {
            throw new Error('CSV file is empty or has no data');
        }

        // Handle different delimiters
        const firstLine = lines[0];
        const delimiter = firstLine.includes('\t') ? '\t' : ',';
        
        const headers = firstLine.split(delimiter).map(h => h.trim().replace(/^["']|["']$/g, ''));
        const data = [];

        for (let i = 1; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue; // Skip empty lines
            
            // Simple CSV parsing that handles quoted values
            const values = this.parseCSVLine(line, delimiter);
            const row = {};
            
            headers.forEach((header, index) => {
                if (index < values.length) {
                    const value = values[index];
                    // Try to parse numbers
                    if (value === '' || value === null || value === undefined) {
                        row[header] = null;
                    } else if (!isNaN(value) && value !== '') {
                        row[header] = parseFloat(value);
                    } else {
                        row[header] = value;
                    }
                }
            });
            
            data.push(row);
        }

        return this.normalizeCSVData(data);
    }

    // Parse a single CSV line handling quoted values
    parseCSVLine(line, delimiter = ',') {
        const result = [];
        let current = '';
        let inQuotes = false;
        
        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            const nextChar = line[i + 1];
            
            if (char === '"' || char === "'") {
                if (inQuotes && nextChar === char) {
                    // Escaped quote
                    current += char;
                    i++; // Skip next char
                } else {
                    // Toggle quote state
                    inQuotes = !inQuotes;
                }
            } else if (char === delimiter && !inQuotes) {
                // End of field
                result.push(current.trim());
                current = '';
            } else {
                current += char;
            }
        }
        
        // Don't forget the last field
        result.push(current.trim());
        
        return result;
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

    // Load scenario files from the project
    async loadScenarioFiles() {
        const scenarioFiles = [
            '../scenarios/scenarios.json',
            '../scenarios/enhanced_scenarios.json',
            '../scenarios/pairwise_scenarios.json',
            '../scenarios/true_pairwise_scenarios.json'
        ];

        for (const file of scenarioFiles) {
            try {
                const response = await fetch(file);
                if (response.ok) {
                    const scenarios = await response.json();
                    return this.processScenarios(scenarios);
                }
            } catch (error) {
                console.warn(`Failed to load scenario file ${file}:`, error);
            }
        }

        // Fallback to synthetic data
        return this.generateSyntheticData();
    }

    // Process scenario definitions into experiment data
    processScenarios(scenarios) {
        if (!Array.isArray(scenarios)) {
            scenarios = [scenarios];
        }

        return scenarios.slice(0, 3).map(scenario => ({
            id: `scenario_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            name: scenario.scenario_name || 'Unnamed Scenario',
            timestamp: new Date().toISOString(),
            config: {
                num_agents: scenario.num_agents || 30,
                num_rounds: scenario.num_rounds || 100,
                network_type: scenario.network_type || 'fully_connected',
                network_params: scenario.network_params || {},
                interaction_mode: scenario.interaction_mode || 'neighborhood',
                agent_strategies: scenario.agent_strategies || {},
                learning_rate: scenario.learning_rate || 0.1,
                discount_factor: scenario.discount_factor || 0.9,
                epsilon: scenario.epsilon || 0.1
            },
            results: this.simulateScenarioResults(scenario),
            isExample: true
        }));
    }

    // Simulate results for a scenario
    simulateScenarioResults(scenario) {
        const numRounds = scenario.num_rounds || 100;
        const results = [];
        let cooperationRate = 0.5;
        
        // Different evolution patterns based on strategies
        const hasLearning = Object.keys(scenario.agent_strategies || {}).some(s => 
            s.includes('q_learning') || s.includes('lra_q') || s.includes('hysteretic_q')
        );
        const hasCooperative = Object.keys(scenario.agent_strategies || {}).some(s => 
            s.includes('cooperate') || s.includes('tit_for_tat') || s.includes('generous')
        );
        const hasDefectors = Object.keys(scenario.agent_strategies || {}).some(s => 
            s.includes('defect')
        );

        for (let round = 0; round < numRounds; round++) {
            // Simulate cooperation evolution based on strategy mix
            if (hasLearning) {
                // Learning agents: gradual improvement with exploration noise
                cooperationRate += (0.7 - cooperationRate) * 0.01;
                cooperationRate += (Math.random() - 0.5) * 0.05;
            } else if (hasCooperative && !hasDefectors) {
                // Cooperative environment: high and stable
                cooperationRate += (0.9 - cooperationRate) * 0.05;
                cooperationRate += (Math.random() - 0.5) * 0.02;
            } else if (hasDefectors && !hasCooperative) {
                // Defection dominant: low and declining
                cooperationRate += (0.1 - cooperationRate) * 0.05;
                cooperationRate += (Math.random() - 0.5) * 0.02;
            } else {
                // Mixed: oscillating around 0.5
                cooperationRate = 0.5 + 0.2 * Math.sin(round / 20);
                cooperationRate += (Math.random() - 0.5) * 0.1;
            }

            // Clamp between 0 and 1
            cooperationRate = Math.max(0, Math.min(1, cooperationRate));

            results.push({
                round: round,
                cooperation_rate: cooperationRate,
                avg_score: 1 + cooperationRate * 3 + (Math.random() - 0.5) * 0.5,
                num_cooperators: Math.round(cooperationRate * (scenario.num_agents || 30)),
                num_defectors: Math.round((1 - cooperationRate) * (scenario.num_agents || 30))
            });
        }

        return results;
    }

    // Load example data from the project
    async loadExampleData() {
        // First try to load real scenario files
        const scenarioData = await this.loadScenarioFiles();
        if (scenarioData && scenarioData.length > 0) {
            return scenarioData;
        }

        // Then try to load result files
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