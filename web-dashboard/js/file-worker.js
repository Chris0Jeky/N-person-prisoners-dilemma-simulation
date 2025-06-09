/**
 * Web Worker for processing large files
 * Handles CSV and JSON parsing in a background thread
 */

// Import data processing functions
self.importScripts('data-processor.js');

// Message handler
self.addEventListener('message', async (event) => {
    const { type, data, id } = event.data;
    
    try {
        let result;
        
        switch (type) {
            case 'parseFile':
                result = await processFile(data);
                break;
                
            case 'parseCSV':
                result = await processCSV(data);
                break;
                
            case 'parseJSON':
                result = await processJSON(data);
                break;
                
            default:
                throw new Error(`Unknown message type: ${type}`);
        }
        
        // Send success response
        self.postMessage({
            id: id,
            type: 'success',
            result: result
        });
        
    } catch (error) {
        // Send error response
        self.postMessage({
            id: id,
            type: 'error',
            error: error.message
        });
    }
});

// Process a file with progress updates
async function processFile(data) {
    const { content, filename, fileSize } = data;
    const extension = filename.substring(filename.lastIndexOf('.'));
    
    // Send initial progress
    self.postMessage({
        type: 'progress',
        progress: 0,
        message: 'Starting file processing...'
    });
    
    let result;
    
    if (extension === '.json') {
        result = await processJSON({ content, fileSize });
    } else if (extension === '.csv') {
        result = await processCSV({ content, fileSize });
    } else {
        throw new Error(`Unsupported file type: ${extension}`);
    }
    
    // Send completion progress
    self.postMessage({
        type: 'progress',
        progress: 100,
        message: 'Processing complete!'
    });
    
    return result;
}

// Process JSON with chunking for large files
async function processJSON(data) {
    const { content, fileSize } = data;
    const processor = new DataProcessor();
    
    // For very large files, parse in chunks
    if (fileSize > 10 * 1024 * 1024) { // 10MB
        self.postMessage({
            type: 'progress',
            progress: 10,
            message: 'Parsing large JSON file...'
        });
        
        // Parse JSON
        const parsed = JSON.parse(content);
        
        self.postMessage({
            type: 'progress',
            progress: 50,
            message: 'Normalizing data...'
        });
        
        // Normalize data
        const normalized = processor.normalizeData(parsed);
        
        self.postMessage({
            type: 'progress',
            progress: 90,
            message: 'Finalizing...'
        });
        
        return normalized;
    } else {
        // Small file, process normally
        return processor.parseJSON(content);
    }
}

// Process CSV with streaming for large files
async function processCSV(data) {
    const { content, fileSize } = data;
    const processor = new DataProcessor();
    
    // For large CSV files, process in chunks
    if (fileSize > 5 * 1024 * 1024) { // 5MB
        const lines = content.split('\n');
        const totalLines = lines.length;
        const chunkSize = 1000;
        const results = [];
        
        self.postMessage({
            type: 'progress',
            progress: 5,
            message: `Processing ${totalLines} lines...`
        });
        
        // Process header
        const headers = lines[0].split(',').map(h => h.trim());
        
        // Process data in chunks
        for (let i = 1; i < totalLines; i += chunkSize) {
            const chunk = lines.slice(i, Math.min(i + chunkSize, totalLines));
            const chunkData = [];
            
            for (const line of chunk) {
                if (!line.trim()) continue;
                
                const values = processor.parseCSVLine(line);
                const row = {};
                
                headers.forEach((header, index) => {
                    if (index < values.length) {
                        const value = values[index];
                        row[header] = isNaN(value) || value === '' ? value : parseFloat(value);
                    }
                });
                
                chunkData.push(row);
            }
            
            results.push(...chunkData);
            
            // Update progress
            const progress = Math.round((i / totalLines) * 90) + 5;
            self.postMessage({
                type: 'progress',
                progress: progress,
                message: `Processed ${Math.min(i + chunkSize, totalLines)} of ${totalLines} lines...`
            });
            
            // Allow UI to update
            await new Promise(resolve => setTimeout(resolve, 0));
        }
        
        self.postMessage({
            type: 'progress',
            progress: 95,
            message: 'Normalizing data...'
        });
        
        // Normalize the complete dataset
        return processor.normalizeCSVData(results);
    } else {
        // Small file, process normally
        return processor.parseCSV(content);
    }
}

// Utility function to estimate memory usage
function estimateMemoryUsage(data) {
    const jsonString = JSON.stringify(data);
    const bytes = new Blob([jsonString]).size;
    return {
        bytes: bytes,
        megabytes: (bytes / (1024 * 1024)).toFixed(2)
    };
}