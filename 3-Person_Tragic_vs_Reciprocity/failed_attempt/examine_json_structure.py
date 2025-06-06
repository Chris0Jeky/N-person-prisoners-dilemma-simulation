import json
import os

# Use a specific file - find the most recent results directory
import glob

# Find the most recent results
result_dirs = glob.glob("3-Person_Tragic_vs_Reciprocity/results/chris_huyck_*")
if not result_dirs:
    print("No results directories found!")
    exit(1)

# Sort by timestamp and get the latest
latest_dir = sorted(result_dirs)[-1]
print(f"Using results from: {latest_dir}")

# Look for pairwise results file
json_files = glob.glob(f"{latest_dir}/*_Pairwise.json")
if not json_files:
    print("No pairwise JSON files found!")
    # Try all JSON files
    json_files = glob.glob(f"{latest_dir}/*.json")
    if not json_files:
        print("No JSON files found!")
        exit(1)

json_file = json_files[0]
print(f"Examining file: {json_file}\n")

with open(json_file, 'r') as f:
    data = json.load(f)

# Print top-level keys
print("Top-level keys:")
for key in data.keys():
    print(f"  - {key}")

# Examine episode structure
if 'episode_data' in data and data['episode_data']:
    print("\nEpisode data structure:")
    first_episode = data['episode_data'][0]
    for key in first_episode.keys():
        print(f"  - {key}")
    
    # Check round data structure
    if 'round_data' in first_episode and first_episode['round_data']:
        print("\nRound data structure:")
        first_round = first_episode['round_data'][0]
        for key in first_round.keys():
            print(f"  - {key}")
        
        # Check if there's a details section with interactions
        if 'details' in first_round:
            print("\nRound details structure:")
            for key in first_round['details'].keys():
                print(f"  - {key}")
            
            # Check interactions structure
            if 'interactions' in first_round['details'] and first_round['details']['interactions']:
                print("\nInteraction structure (first interaction):")
                first_interaction = first_round['details']['interactions'][0]
                for key, value in first_interaction.items():
                    print(f"  - {key}: {value}")
                    
                print(f"\nTotal interactions in first round: {len(first_round['details']['interactions'])}")
                
                # Show all interactions to understand pattern
                print("\nAll interactions in first round:")
                for i, interaction in enumerate(first_round['details']['interactions']):
                    print(f"  Interaction {i+1}: {interaction['agent1']} vs {interaction['agent2']} - "
                          f"{interaction['action1']} vs {interaction['action2']}")

# Check if pairwise analysis already exists
if 'pairwise_analysis' in data:
    print("\npairwise_analysis already exists:")
    for key in data['pairwise_analysis'].keys():
        print(f"  - {key}")

# Check agent IDs
if 'config' in data and 'agents' in data['config']:
    print("\nAgent configurations:")
    for agent in data['config']['agents']:
        print(f"  - {agent['name']} (ID: {agent['id']}, Strategy: {agent['strategy']}")
