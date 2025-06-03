import json
import glob
import os

# Find a results directory
results_dirs = glob.glob("3-Person_Tragic_vs_Reciprocity/results/chris_huyck_*")
if results_dirs:
    results_dir = results_dirs[-1]
    json_file = os.path.join(results_dir, "results_2TFT_1AllD_NoExpl_Pairwise.json")
    
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Check the structure
        print("Keys in data:", data.keys())
        
        if 'round_data' in data and len(data['round_data']) > 0:
            sample_round = data['round_data'][0]
            print("\nKeys in round_data[0]:", sample_round.keys())
            
            # Check if interactions exist
            if 'interactions' in sample_round:
                print("\nFound interactions! Sample:")
                print(json.dumps(sample_round['interactions'][0], indent=2))
            else:
                print("\nNo 'interactions' key found. Available keys:", sample_round.keys())
                print("\nSample round data:")
                print(json.dumps(sample_round, indent=2))
