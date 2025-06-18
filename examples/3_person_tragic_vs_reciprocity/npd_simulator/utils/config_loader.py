"""
Configuration loading and validation utilities
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Union
from jsonschema import validate, ValidationError


def load_config(config_path: Union[str, Path]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Load configuration from JSON or YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary or list of configurations
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load based on extension
    if config_path.suffix.lower() == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif config_path.suffix.lower() in ['.yml', '.yaml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    # Validate configuration
    if isinstance(config, dict):
        validate_config(config)
    elif isinstance(config, list):
        for cfg in config:
            validate_config(cfg)
    
    return config


def validate_config(config: Dict[str, Any]):
    """
    Validate experiment configuration.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValidationError: If configuration is invalid
    """
    # Basic validation
    if 'agents' not in config:
        raise ValidationError("Configuration must include 'agents'")
    
    if not isinstance(config['agents'], list):
        raise ValidationError("'agents' must be a list")
    
    if len(config['agents']) < 2:
        raise ValidationError("At least 2 agents required")
    
    # Validate each agent
    for i, agent in enumerate(config['agents']):
        if 'type' not in agent:
            raise ValidationError(f"Agent {i} missing 'type'")
        if 'id' not in agent:
            raise ValidationError(f"Agent {i} missing 'id'")
    
    # Validate game parameters
    if 'num_rounds' in config and config['num_rounds'] < 1:
        raise ValidationError("'num_rounds' must be >= 1")
    
    if 'rounds_per_pair' in config and config['rounds_per_pair'] < 1:
        raise ValidationError("'rounds_per_pair' must be >= 1")


# Configuration schemas for validation
AGENT_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "type": {"type": "string"},
        "exploration_rate": {"type": "number", "minimum": 0, "maximum": 1},
        "learning_rate": {"type": "number", "minimum": 0, "maximum": 1},
        "discount_factor": {"type": "number", "minimum": 0, "maximum": 1},
        "epsilon": {"type": "number", "minimum": 0, "maximum": 1},
        "epsilon_decay": {"type": "number", "minimum": 0, "maximum": 1},
        "epsilon_min": {"type": "number", "minimum": 0, "maximum": 1},
        "exclude_self": {"type": "boolean"},
        "opponent_modeling": {"type": "boolean"},
        "state_type": {"type": "string"},
        "cooperation_probability": {"type": "number", "minimum": 0, "maximum": 1},
        "threshold": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "required": ["id", "type"]
}

NPD_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "agents": {
            "type": "array",
            "items": AGENT_SCHEMA,
            "minItems": 2
        },
        "num_rounds": {"type": "integer", "minimum": 1},
        "episode_length": {"type": "integer", "minimum": 1}
    },
    "required": ["agents", "num_rounds"]
}

PAIRWISE_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "agents": {
            "type": "array",
            "items": AGENT_SCHEMA,
            "minItems": 2
        },
        "rounds_per_pair": {"type": "integer", "minimum": 1},
        "num_episodes": {"type": "integer", "minimum": 1}
    },
    "required": ["agents", "rounds_per_pair"]
}


def create_sample_config(config_type: str = "npd") -> Dict[str, Any]:
    """
    Create a sample configuration for testing.
    
    Args:
        config_type: Type of configuration ("npd" or "pairwise")
        
    Returns:
        Sample configuration
    """
    if config_type == "npd":
        return {
            "name": "sample_npd_experiment",
            "agents": [
                {"id": 0, "type": "TFT", "exploration_rate": 0.1},
                {"id": 1, "type": "AllD", "exploration_rate": 0.0},
                {"id": 2, "type": "QLearning", "exploration_rate": 0.0,
                 "learning_rate": 0.1, "epsilon": 0.1}
            ],
            "num_rounds": 1000
        }
    elif config_type == "pairwise":
        return {
            "name": "sample_pairwise_experiment",
            "agents": [
                {"id": 0, "type": "TFT", "exploration_rate": 0.0},
                {"id": 1, "type": "AllD", "exploration_rate": 0.0},
                {"id": 2, "type": "AllC", "exploration_rate": 0.0}
            ],
            "rounds_per_pair": 100,
            "num_episodes": 1
        }
    else:
        raise ValueError(f"Unknown config type: {config_type}")