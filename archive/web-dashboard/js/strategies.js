/**
 * Strategies Module
 * Strategy definitions and analysis
 */

// Centralized color system for consistency
const STRATEGY_COLORS = {
    palette: [
        '#3b82f6', // Blue
        '#10b981', // Green
        '#ef4444', // Red
        '#8b5cf6', // Purple
        '#f59e0b', // Orange
        '#06b6d4', // Cyan
        '#ec4899', // Pink
        '#84cc16', // Lime
        '#6366f1', // Indigo
        '#14b8a6', // Teal
        '#f97316', // Dark Orange
        '#a855f7', // Violet
        '#0ea5e9', // Light Blue
        '#dc2626', // Dark Red
        '#94a3b8'  // Gray
    ],
    semantic: {
        cooperative: '#10b981',
        defective: '#ef4444',
        learning: '#8b5cf6',
        adaptive: '#f59e0b',
        neutral: '#94a3b8'
    },
    opacity: {
        line: 1,
        fill: 0.2,
        hover: 0.8,
        disabled: 0.3
    }
};

// Strategy color assignment tracker
let strategyColorMap = new Map();
let colorIndex = 0;

// Get consistent color for a strategy
function getStrategyColor(strategyId) {
    if (!strategyColorMap.has(strategyId)) {
        // First check if strategy has predefined color
        if (STRATEGIES[strategyId] && STRATEGIES[strategyId].color) {
            strategyColorMap.set(strategyId, STRATEGIES[strategyId].color);
        } else {
            // Assign next color from palette
            strategyColorMap.set(strategyId, STRATEGY_COLORS.palette[colorIndex % STRATEGY_COLORS.palette.length]);
            colorIndex++;
        }
    }
    return strategyColorMap.get(strategyId);
}

// Get color with opacity
function getStrategyColorWithOpacity(strategyId, opacity) {
    const color = getStrategyColor(strategyId);
    // Convert hex to rgba
    const r = parseInt(color.slice(1, 3), 16);
    const g = parseInt(color.slice(3, 5), 16);
    const b = parseInt(color.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${opacity})`;
}

const STRATEGIES = {
    'always_cooperate': {
        name: 'Always Cooperate',
        icon: '😇',
        color: '#10b981',
        description: 'Always chooses to cooperate, regardless of opponent behavior'
    },
    'always_defect': {
        name: 'Always Defect',
        icon: '😈',
        color: '#ef4444',
        description: 'Always chooses to defect, maximizing individual gain'
    },
    'tit_for_tat': {
        name: 'Tit for Tat',
        icon: '🤝',
        color: '#3b82f6',
        description: 'Cooperates based on proportion of neighbors who cooperated'
    },
    'proportional_tit_for_tat': {
        name: 'Proportional TFT',
        icon: '📊',
        color: '#06b6d4',
        description: 'Cooperates with probability equal to neighborhood cooperation'
    },
    'generous_tit_for_tat': {
        name: 'Generous TFT',
        icon: '💝',
        color: '#ec4899',
        description: 'Like TFT but occasionally forgives defection'
    },
    'suspicious_tit_for_tat': {
        name: 'Suspicious TFT',
        icon: '🤨',
        color: '#f97316',
        description: 'Starts with defection, then follows TFT'
    },
    'tit_for_two_tats': {
        name: 'Tit for Two Tats',
        icon: '✌️',
        color: '#84cc16',
        description: 'Only defects after two consecutive defections'
    },
    'pavlov': {
        name: 'Pavlov',
        icon: '🔄',
        color: '#f59e0b',
        description: 'Win-stay, lose-shift strategy'
    },
    'random': {
        name: 'Random',
        icon: '🎲',
        color: '#94a3b8',
        description: 'Randomly cooperates or defects'
    },
    'q_learning': {
        name: 'Q-Learning',
        icon: '🧠',
        color: '#8b5cf6',
        description: 'Learns optimal strategy through reinforcement'
    },
    'hysteretic_q': {
        name: 'Hysteretic Q',
        icon: '📈',
        color: '#f97316',
        description: 'Optimistic Q-learning that learns positive outcomes faster'
    },
    'lra_q': {
        name: 'LRA-Q',
        icon: '🎯',
        color: '#0ea5e9',
        description: 'Adjusts learning rate based on cooperation levels'
    },
    'wolf_phc': {
        name: 'WoLF-PHC',
        icon: '🐺',
        color: '#06b6d4',
        description: 'Win or Learn Fast - adaptive learning rates'
    },
    'ucb1_q': {
        name: 'UCB1-Q',
        icon: '🎰',
        color: '#dc2626',
        description: 'Balances exploration and exploitation'
    }
};

// Export for use
window.STRATEGIES = STRATEGIES;