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
        // Check if this is agent-level data (has agent_id column)
        if (rows.length > 0 && rows[0].hasOwnProperty('agent_id')) {
            return this.processAgentLevelData(rows);
        }
        
        // Check if this is a single experiment with round data
        if (rows.length > 0 && rows[0].hasOwnProperty('round')) {
            // This appears to be round-by-round data for a single experiment
            const firstRow = rows[0];
            
            // Extract experiment metadata from first row if available
            const experiment = {
                id: firstRow.experiment_id || firstRow.scenario || Date.now().toString(),
                name: firstRow.experiment_name || firstRow.scenario_name || 'CSV Import',
                config: {
                    num_agents: firstRow.num_agents || this.inferNumAgents(rows) || 0,
                    num_rounds: rows.length,
                    network_type: firstRow.network_type || 'unknown',
                    agent_strategies: this.inferStrategies(rows)
                },
                results: rows.map(row => ({
                    round: parseInt(row.round) || 0,
                    cooperation_rate: parseFloat(row.cooperation_rate || row.coop_rate || row.avg_cooperation || 0),
                    avg_score: parseFloat(row.avg_score || row.average_score || row.mean_score || 0),
                    num_cooperators: parseInt(row.num_cooperators || row.cooperators || 0),
                    num_defectors: parseInt(row.num_defectors || row.defectors || 0),
                    ...row
                }))
            };
            
            return [experiment];
        }
        
        // Otherwise, try to group by experiment/scenario
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
                        network_type: row.network_type || 'unknown',
                        agent_strategies: {}
                    },
                    results: []
                };
            }
            
            experiments[expId].results.push({
                round: parseInt(row.round) || experiments[expId].results.length,
                cooperation_rate: parseFloat(row.cooperation_rate || row.coop_rate || 0),
                avg_score: parseFloat(row.avg_score || row.average_score || 0),
                ...row
            });
        });
        
        return Object.values(experiments);
    }

    inferNumAgents(rows) {
        // Try to infer number of agents from cooperators + defectors
        for (const row of rows) {
            const cooperators = parseInt(row.num_cooperators || row.cooperators || 0);
            const defectors = parseInt(row.num_defectors || row.defectors || 0);
            if (cooperators + defectors > 0) {
                return cooperators + defectors;
            }
        }
        return 0;
    }

    inferStrategies(rows) {
        // Try to infer strategies from column names
        const strategies = {};
        const firstRow = rows[0];
        
        // Look for strategy-related columns
        Object.keys(firstRow).forEach(key => {
            if (key.includes('strategy_') || key.includes('num_') && !key.includes('round')) {
                const strategyName = key.replace('strategy_', '').replace('num_', '');
                if (strategyName !== 'cooperators' && strategyName !== 'defectors') {
                    strategies[strategyName] = parseInt(firstRow[key]) || 0;
                }
            }
        });
        
        return strategies;
    }
    
    // Process agent-level data (like from experiment_results_rounds.csv)
    processAgentLevelData(rows) {
        // Group by scenario/experiment
        const experiments = {};
        
        rows.forEach(row => {
            const expId = row.scenario_name || row.experiment_id || 'default';
            const round = parseInt(row.round) || 0;
            
            if (!experiments[expId]) {
                experiments[expId] = {
                    id: expId + '_' + Date.now(),
                    name: expId,
                    config: {
                        num_agents: 0,
                        num_rounds: 0,
                        network_type: 'unknown',
                        agent_strategies: {}
                    },
                    roundData: {},
                    agentSet: new Set(),
                    strategySet: {}
                };
            }
            
            // Track agents and strategies
            experiments[expId].agentSet.add(row.agent_id);
            if (row.strategy) {
                experiments[expId].strategySet[row.strategy] = (experiments[expId].strategySet[row.strategy] || 0) + 1;
            }
            
            // Initialize round data if needed
            if (!experiments[expId].roundData[round]) {
                experiments[expId].roundData[round] = {
                    round: round,
                    cooperators: 0,
                    defectors: 0,
                    totalPayoff: 0,
                    agentCount: 0
                };
            }
            
            // Aggregate round data
            const roundData = experiments[expId].roundData[round];
            if (row.move === 'cooperate') {
                roundData.cooperators++;
            } else if (row.move === 'defect') {
                roundData.defectors++;
            }
            roundData.totalPayoff += parseFloat(row.payoff) || 0;
            roundData.agentCount++;
        });
        
        // Convert to experiment format
        return Object.entries(experiments).map(([expId, expData]) => {
            const rounds = Object.values(expData.roundData).sort((a, b) => a.round - b.round);
            const numAgents = expData.agentSet.size;
            
            // Count unique strategies
            const strategies = {};
            Object.entries(expData.strategySet).forEach(([strategy, count]) => {
                // Normalize strategy counts to get number of agents per strategy
                strategies[strategy] = Math.round(count / rounds.length);
            });
            
            return {
                id: expData.id,
                name: expData.name,
                config: {
                    num_agents: numAgents,
                    num_rounds: rounds.length,
                    network_type: 'unknown',
                    agent_strategies: strategies
                },
                results: rounds.map(round => ({
                    round: round.round,
                    cooperation_rate: round.cooperators / (round.cooperators + round.defectors),
                    avg_score: round.totalPayoff / round.agentCount,
                    num_cooperators: round.cooperators,
                    num_defectors: round.defectors
                }))
            };
        });
    }
}

// Export for Web Worker use
if (typeof self !== 'undefined') {
    self.DataProcessor = DataProcessor;
}