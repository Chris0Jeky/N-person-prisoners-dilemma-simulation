#!/bin/bash
# Simple run script for different experiment modes

echo "N-Person Prisoner's Dilemma Simulation Framework"
echo "=============================================="
echo ""
echo "Select experiment mode:"
echo "1) Default experiments"
echo "2) Q-learning experiments"
echo "3) Basic example"
echo "4) Q-learning comparison"
echo "5) Full analysis"
echo ""

read -p "Enter choice (1-5): " choice

case $choice in
    1)
        echo "Running default experiments..."
        python main.py --mode default
        ;;
    2)
        echo "Running Q-learning experiments..."
        python main.py --mode qlearning
        ;;
    3)
        echo "Running basic example..."
        python examples/basic_experiment.py
        ;;
    4)
        echo "Running Q-learning comparison..."
        python examples/qlearning_comparison.py
        ;;
    5)
        echo "Running full analysis..."
        python examples/full_analysis.py
        ;;
    *)
        echo "Invalid choice. Please run again and select 1-5."
        exit 1
        ;;
esac

echo ""
echo "Experiment completed!"