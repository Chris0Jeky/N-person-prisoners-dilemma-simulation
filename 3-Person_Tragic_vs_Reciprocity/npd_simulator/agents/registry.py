"""
Agent Registry Plugin System

Provides dynamic agent registration and creation for easy extensibility.
"""

from typing import Dict, Type, Any, Optional, List
from importlib import import_module
import inspect
import logging
from pathlib import Path

from .base import Agent, NPDAgent, PairwiseAgent


class AgentRegistry:
    """
    Central registry for all agent types.
    
    Supports:
    - Dynamic agent registration
    - Automatic discovery of agents in packages
    - Factory pattern for agent creation
    - Plugin-style agent additions
    """
    
    _agents: Dict[str, Type[Agent]] = {}
    _agent_metadata: Dict[str, Dict[str, Any]] = {}
    _logger = logging.getLogger(__name__)
    
    @classmethod
    def register(cls, 
                name: Optional[str] = None,
                category: str = "general",
                description: str = "",
                parameters: Optional[Dict[str, Any]] = None):
        """
        Decorator for registering agent classes.
        
        Args:
            name: Registration name (defaults to class name without 'Agent' suffix)
            category: Agent category (e.g., "basic", "learning", "custom")
            description: Human-readable description
            parameters: Dictionary describing agent parameters
            
        Example:
            @AgentRegistry.register(
                name="TFT",
                category="basic",
                description="Tit-for-Tat strategy"
            )
            class TFTAgent(NPDAgent):
                pass
        """
        def decorator(agent_class: Type[Agent]) -> Type[Agent]:
            # Determine registration name
            reg_name = name
            if reg_name is None:
                class_name = agent_class.__name__
                if class_name.endswith('Agent'):
                    reg_name = class_name[:-5]
                else:
                    reg_name = class_name
            
            # Register the agent
            cls._agents[reg_name] = agent_class
            
            # Store metadata
            cls._agent_metadata[reg_name] = {
                'class': agent_class,
                'category': category,
                'description': description,
                'parameters': parameters or {},
                'module': agent_class.__module__,
                'is_npd': issubclass(agent_class, NPDAgent),
                'is_pairwise': issubclass(agent_class, PairwiseAgent)
            }
            
            cls._logger.info(f"Registered agent '{reg_name}' from {agent_class.__module__}")
            
            # Add registry info to the class
            agent_class._registry_name = reg_name
            agent_class._registry_metadata = cls._agent_metadata[reg_name]
            
            return agent_class
        
        return decorator
    
    @classmethod
    def create(cls, agent_type: str, agent_id: int, **kwargs) -> Agent:
        """
        Create an agent instance by type name.
        
        Args:
            agent_type: Registered agent type name
            agent_id: Unique agent identifier
            **kwargs: Agent-specific parameters
            
        Returns:
            Configured agent instance
            
        Raises:
            ValueError: If agent_type is not registered
        """
        if agent_type not in cls._agents:
            available = ', '.join(sorted(cls._agents.keys()))
            raise ValueError(
                f"Unknown agent type: {agent_type}. "
                f"Available types: {available}"
            )
        
        agent_class = cls._agents[agent_type]
        
        # Create instance with appropriate parameters
        try:
            agent = agent_class(agent_id, **kwargs)
            cls._logger.debug(f"Created {agent_type} agent with ID {agent_id}")
            return agent
        except Exception as e:
            cls._logger.error(f"Failed to create {agent_type} agent: {e}")
            raise
    
    @classmethod
    def list_agents(cls, category: Optional[str] = None) -> List[str]:
        """
        List all registered agent types.
        
        Args:
            category: Filter by category (optional)
            
        Returns:
            List of agent type names
        """
        if category:
            return [
                name for name, meta in cls._agent_metadata.items()
                if meta['category'] == category
            ]
        return sorted(cls._agents.keys())
    
    @classmethod
    def get_metadata(cls, agent_type: str) -> Dict[str, Any]:
        """
        Get metadata for a specific agent type.
        
        Args:
            agent_type: Registered agent type name
            
        Returns:
            Agent metadata dictionary
        """
        if agent_type not in cls._agent_metadata:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return cls._agent_metadata[agent_type].copy()
    
    @classmethod
    def get_categories(cls) -> List[str]:
        """Get all unique agent categories."""
        categories = set(
            meta['category'] for meta in cls._agent_metadata.values()
        )
        return sorted(categories)
    
    @classmethod
    def auto_discover(cls, package_path: Path):
        """
        Automatically discover and register agents in a package.
        
        Args:
            package_path: Path to package containing agent modules
        """
        cls._logger.info(f"Auto-discovering agents in {package_path}")
        
        # Find all Python files
        for py_file in package_path.glob("*.py"):
            if py_file.name.startswith('_'):
                continue
                
            # Convert to module name
            module_name = f"agents.{py_file.stem}"
            
            try:
                # Import the module
                module = import_module(module_name)
                
                # Find all Agent subclasses
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, Agent) and 
                        obj not in (Agent, NPDAgent, PairwiseAgent) and
                        not name.startswith('_')):
                        
                        # Check if already registered
                        if hasattr(obj, '_registry_name'):
                            continue
                        
                        # Auto-register with default metadata
                        cls.register()(obj)
                        
            except Exception as e:
                cls._logger.warning(f"Failed to import {module_name}: {e}")
    
    @classmethod
    def clear(cls):
        """Clear all registrations (mainly for testing)."""
        cls._agents.clear()
        cls._agent_metadata.clear()
    
    @classmethod
    def export_schema(cls) -> Dict[str, Any]:
        """
        Export registry schema for documentation/configuration.
        
        Returns:
            Dictionary with all agent types and their schemas
        """
        schema = {
            'agents': {},
            'categories': cls.get_categories()
        }
        
        for agent_type, metadata in cls._agent_metadata.items():
            schema['agents'][agent_type] = {
                'category': metadata['category'],
                'description': metadata['description'],
                'parameters': metadata['parameters'],
                'supports_npd': metadata['is_npd'],
                'supports_pairwise': metadata['is_pairwise']
            }
        
        return schema


# Convenience functions
def register_agent(name: Optional[str] = None, **kwargs):
    """Convenience function for agent registration."""
    return AgentRegistry.register(name, **kwargs)


def create_agent(agent_type: str, agent_id: int, **kwargs) -> Agent:
    """Convenience function for agent creation."""
    return AgentRegistry.create(agent_type, agent_id, **kwargs)


def list_agents(category: Optional[str] = None) -> List[str]:
    """Convenience function to list agents."""
    return AgentRegistry.list_agents(category)