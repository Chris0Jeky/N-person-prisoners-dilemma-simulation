"""
Experiment Profile System

Allows loading and managing experiment configurations from YAML profiles.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from copy import deepcopy
import logging


class ProfileLoader:
    """
    Loads and manages experiment profiles from YAML files.
    
    Profiles define reusable experiment configurations that can be:
    - Parameterized with variables
    - Extended with variations
    - Combined to create complex experiments
    """
    
    def __init__(self, profiles_dir: Optional[Path] = None):
        """
        Initialize profile loader.
        
        Args:
            profiles_dir: Directory containing profile YAML files
        """
        self.profiles_dir = profiles_dir or Path(__file__).parent
        self.logger = logging.getLogger(__name__)
        self._loaded_profiles: Dict[str, Dict] = {}
    
    def load_profile(self, profile_name: str) -> Dict[str, Any]:
        """
        Load a profile by name.
        
        Args:
            profile_name: Name of the profile (without .yaml extension)
            
        Returns:
            Profile configuration dictionary
        """
        if profile_name in self._loaded_profiles:
            return deepcopy(self._loaded_profiles[profile_name])
        
        # Try to find the profile file
        profile_path = self._find_profile_file(profile_name)
        
        with open(profile_path, 'r') as f:
            profile = yaml.safe_load(f)
        
        # Process includes if present
        if 'include' in profile:
            profile = self._process_includes(profile)
        
        # Cache the loaded profile
        self._loaded_profiles[profile_name] = profile
        
        return deepcopy(profile)
    
    def _find_profile_file(self, profile_name: str) -> Path:
        """Find profile file by name."""
        # Check with .yaml extension
        yaml_path = self.profiles_dir / f"{profile_name}.yaml"
        if yaml_path.exists():
            return yaml_path
        
        # Check with .yml extension
        yml_path = self.profiles_dir / f"{profile_name}.yml"
        if yml_path.exists():
            return yml_path
        
        # Check if full path was provided
        full_path = Path(profile_name)
        if full_path.exists():
            return full_path
        
        raise FileNotFoundError(f"Profile not found: {profile_name}")
    
    def _process_includes(self, profile: Dict) -> Dict:
        """Process profile includes."""
        includes = profile.pop('include')
        if isinstance(includes, str):
            includes = [includes]
        
        # Start with empty base
        merged = {}
        
        # Merge each included profile
        for include in includes:
            included = self.load_profile(include)
            merged = self._deep_merge(merged, included)
        
        # Merge current profile on top
        return self._deep_merge(merged, profile)
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = deepcopy(base)
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def expand_variations(self, profile: Dict) -> List[Dict[str, Any]]:
        """
        Expand a profile with variations into multiple configurations.
        
        Args:
            profile: Profile with potential variations
            
        Returns:
            List of expanded configurations
        """
        if 'variations' not in profile:
            return [self._prepare_config(profile)]
        
        base_config = {k: v for k, v in profile.items() if k != 'variations'}
        variations = profile['variations']
        
        configs = []
        for variation in variations:
            # Create config for this variation
            config = deepcopy(base_config)
            
            # Apply variation name
            if 'name' in variation:
                config['name'] = f"{base_config.get('name', 'experiment')}_{variation['name']}"
            
            # Apply replacements
            if 'replace' in variation:
                config = self._apply_replacements(config, variation['replace'])
            
            # Apply additions
            if 'add' in variation:
                config = self._apply_additions(config, variation['add'])
            
            # Apply parameter overrides
            if 'parameters' in variation:
                if 'parameters' not in config:
                    config['parameters'] = {}
                config['parameters'].update(variation['parameters'])
            
            configs.append(self._prepare_config(config))
        
        return configs
    
    def _apply_replacements(self, config: Dict, replacements: Dict) -> Dict:
        """Apply agent replacements to configuration."""
        if 'agents' not in config:
            return config
        
        for index, replacement in replacements.items():
            if isinstance(index, str) and index.isdigit():
                index = int(index)
            
            if 0 <= index < len(config['agents']):
                # Preserve agent ID
                agent_id = config['agents'][index].get('id', index)
                config['agents'][index] = replacement
                config['agents'][index]['id'] = agent_id
        
        return config
    
    def _apply_additions(self, config: Dict, additions: List[Dict]) -> Dict:
        """Apply agent additions to configuration."""
        if 'agents' not in config:
            config['agents'] = []
        
        next_id = len(config['agents'])
        for addition in additions:
            agent = deepcopy(addition)
            if 'id' not in agent:
                agent['id'] = next_id
                next_id += 1
            config['agents'].append(agent)
        
        return config
    
    def _prepare_config(self, profile: Dict) -> Dict[str, Any]:
        """Prepare a profile for use as experiment configuration."""
        config = deepcopy(profile)
        
        # Ensure required fields
        if 'num_rounds' not in config:
            config['num_rounds'] = 100
        
        if 'rounds_per_pair' not in config:
            config['rounds_per_pair'] = 100
        
        # Process agent defaults
        if 'agents' in config:
            for i, agent in enumerate(config['agents']):
                if 'id' not in agent:
                    agent['id'] = i
                if 'exploration_rate' not in agent:
                    agent['exploration_rate'] = 0.0
        
        return config
    
    def list_profiles(self) -> List[str]:
        """List all available profiles."""
        profiles = []
        
        for file in self.profiles_dir.glob("*.yaml"):
            profiles.append(file.stem)
        
        for file in self.profiles_dir.glob("*.yml"):
            if file.stem not in profiles:
                profiles.append(file.stem)
        
        return sorted(profiles)
    
    def validate_profile(self, profile: Dict) -> List[str]:
        """
        Validate a profile configuration.
        
        Args:
            profile: Profile to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required fields
        if 'agents' not in profile and 'base_agents' not in profile:
            errors.append("Profile must define 'agents' or 'base_agents'")
        
        # Validate agents
        if 'agents' in profile:
            for i, agent in enumerate(profile['agents']):
                if 'type' not in agent:
                    errors.append(f"Agent {i} missing 'type' field")
        
        # Validate variations
        if 'variations' in profile:
            for i, var in enumerate(profile['variations']):
                if 'name' not in var:
                    errors.append(f"Variation {i} missing 'name' field")
        
        return errors


class ProfileGenerator:
    """Generate profile files from experiment configurations."""
    
    @staticmethod
    def from_experiment(config: Dict, output_path: Path):
        """
        Generate a profile from an experiment configuration.
        
        Args:
            config: Experiment configuration
            output_path: Path to save profile
        """
        profile = {
            'name': config.get('name', 'generated_profile'),
            'description': 'Auto-generated from experiment',
            'agents': config.get('agents', []),
            'num_rounds': config.get('num_rounds', 100),
            'parameters': config.get('parameters', {})
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(profile, f, default_flow_style=False, sort_keys=False)
    
    @staticmethod
    def create_template(template_type: str, output_path: Path):
        """
        Create a profile template.
        
        Args:
            template_type: Type of template ('basic', 'variations', 'qlearning')
            output_path: Path to save template
        """
        templates = {
            'basic': {
                'name': 'basic_experiment',
                'description': 'Basic experiment template',
                'agents': [
                    {'type': 'TFT', 'exploration_rate': 0.0},
                    {'type': 'TFT', 'exploration_rate': 0.0},
                    {'type': 'AllD', 'exploration_rate': 0.0}
                ],
                'num_rounds': 100
            },
            'variations': {
                'name': 'experiment_with_variations',
                'description': 'Template showing variations',
                'base_agents': [
                    {'type': 'TFT', 'exploration_rate': 0.1},
                    {'type': 'TFT', 'exploration_rate': 0.1},
                    {'type': 'AllD', 'exploration_rate': 0.0}
                ],
                'variations': [
                    {
                        'name': 'vs_AllC',
                        'replace': {2: {'type': 'AllC'}}
                    },
                    {
                        'name': 'mixed',
                        'replace': {1: {'type': 'Random', 'cooperation_probability': 0.5}}
                    }
                ]
            },
            'qlearning': {
                'name': 'qlearning_experiment',
                'description': 'Q-Learning agents template',
                'agents': [
                    {
                        'type': 'EnhancedQLearning',
                        'learning_rate': 0.1,
                        'discount_factor': 0.9,
                        'epsilon': 0.1,
                        'epsilon_decay': 0.995
                    },
                    {'type': 'TFT', 'exploration_rate': 0.0},
                    {'type': 'AllD', 'exploration_rate': 0.0}
                ],
                'num_rounds': 1000,
                'parameters': {
                    'episode_length': 100
                }
            }
        }
        
        if template_type not in templates:
            raise ValueError(f"Unknown template type: {template_type}")
        
        with open(output_path, 'w') as f:
            yaml.dump(templates[template_type], f, default_flow_style=False, sort_keys=False)