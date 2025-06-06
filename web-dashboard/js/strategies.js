/**
 * Strategies Module
 * Strategy definitions and analysis
 */

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