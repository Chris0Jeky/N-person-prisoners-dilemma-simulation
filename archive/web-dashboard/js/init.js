/**
 * Initialization Module
 * Ensures all dependencies are loaded before initializing the dashboard
 */

// Global initialization state
window.dashboardInit = {
    lucideReady: false,
    domReady: false,
    dashboardInstance: null
};

// Check if lucide is ready
function checkLucide() {
    if (typeof lucide !== 'undefined' && lucide.createIcons) {
        window.dashboardInit.lucideReady = true;
        console.log('Lucide is ready');
        tryInitDashboard();
        return true;
    }
    return false;
}

// Check if DOM is ready
function checkDOM() {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            window.dashboardInit.domReady = true;
            console.log('DOM is ready');
            tryInitDashboard();
        });
    } else {
        window.dashboardInit.domReady = true;
        console.log('DOM is ready');
        tryInitDashboard();
    }
}

// Try to initialize the dashboard
function tryInitDashboard() {
    if (window.dashboardInit.lucideReady && window.dashboardInit.domReady && !window.dashboardInit.dashboardInstance) {
        console.log('All dependencies ready, initializing dashboard...');
        
        // Create and initialize dashboard
        const dashboard = new Dashboard();
        window.dashboard = dashboard;
        window.dashboardInit.dashboardInstance = dashboard;
        dashboard.init();
        
        // Create initial icons
        lucide.createIcons();
    }
}

// Safe icon creation function for global use
window.safeCreateIcons = function() {
    if (typeof lucide !== 'undefined' && lucide.createIcons) {
        try {
            lucide.createIcons();
        } catch (e) {
            console.warn('Error creating lucide icons:', e);
        }
    }
};

// Start initialization checks
console.log('Starting dashboard initialization...');

// Check lucide periodically until it's ready
if (!checkLucide()) {
    const lucideCheckInterval = setInterval(() => {
        if (checkLucide()) {
            clearInterval(lucideCheckInterval);
        }
    }, 50);
    
    // Timeout after 5 seconds
    setTimeout(() => {
        clearInterval(lucideCheckInterval);
        if (!window.dashboardInit.lucideReady) {
            console.error('Lucide failed to load after 5 seconds');
            // Initialize dashboard anyway
            window.dashboardInit.lucideReady = true;
            tryInitDashboard();
        }
    }, 5000);
}

// Check DOM
checkDOM();

// Also listen for window load as a fallback
window.addEventListener('load', () => {
    if (!window.dashboardInit.lucideReady) {
        checkLucide();
    }
    if (!window.dashboardInit.domReady) {
        window.dashboardInit.domReady = true;
        tryInitDashboard();
    }
});