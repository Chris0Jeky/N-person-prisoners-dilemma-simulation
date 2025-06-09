/**
 * Data Processing Module for Web Worker
 * Handles CSV and JSON parsing without DOM dependencies
 */

class DataProcessor {
    constructor() {
        this.supportedFormats = ['.json', '.csv'];
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
}

// Export for Web Worker use
if (typeof self !== 'undefined') {
    self.DataProcessor = DataProcessor;
}