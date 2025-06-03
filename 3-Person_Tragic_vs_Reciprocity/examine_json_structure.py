import json
import os

# Check if results directory exists
results_dir = "3-Person_Tragic_vs_Reciprocity/results/chris_huyck_updated_20250603_040854"
if not os.path.exists(results_dir):
    # Try alternative paths
    alt_dirs = [
        "results/chris_huyck_updated_20250603_040854",
        "3-Person_Tragic_vs_Reciprocity/results",
        "results"
    ]
    for alt_dir in alt_dirs:
        if os.path.exists(alt_dir):
            results_dir = alt_dir
            break

print(f"Looking in directory: {results_dir}")

# Find a JSON file to examine
json_files = []
for root, dirs, files in os.walk(results_dir):
    for file in files:
        if file.endswith('.json') and 'pairwise' in file.lower():
            json_files.append(os.path.join(root, file))
            
if not json_files:
    print("No pairwise JSON files found")
    exit()

json_file = json_files[0]
print(f"Examining: {json_file}")

with open(json_file, 'r') as f:
    data = json.load(f)

# Print top-level keys
print("\nTop-level keys:")
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
