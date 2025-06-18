"""
Unified Results Management System

Provides centralized organization, storage, and retrieval of experiment results.
"""

import json
import csv
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import logging
from collections import defaultdict


class ResultsManager:
    """
    Centralized management of experiment results.
    
    Features:
    - Consistent directory structure
    - Result indexing and search
    - Format conversion (JSON, CSV, DataFrame)
    - Result comparison tools
    - Historical data import
    """
    
    def __init__(self, base_dir: Union[str, Path] = "results"):
        """
        Initialize results manager.
        
        Args:
            base_dir: Base directory for all results
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Create subdirectories
        self.experiments_dir = self.base_dir / "experiments"
        self.comparisons_dir = self.base_dir / "comparisons"
        self.archives_dir = self.base_dir / "archives"
        
        for dir in [self.experiments_dir, self.comparisons_dir, self.archives_dir]:
            dir.mkdir(exist_ok=True)
        
        # Initialize index
        self.index_file = self.base_dir / "results_index.json"
        self.index = self._load_index()
    
    def _load_index(self) -> Dict[str, Any]:
        """Load or create results index."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {"experiments": {}, "last_updated": None}
    
    def _save_index(self):
        """Save results index."""
        self.index["last_updated"] = datetime.now().isoformat()
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def save_experiment(self, 
                       results: Dict[str, Any],
                       experiment_name: str,
                       category: Optional[str] = None) -> Path:
        """
        Save experiment results with consistent organization.
        
        Args:
            results: Experiment results dictionary
            experiment_name: Name for the experiment
            category: Optional category for organization
            
        Returns:
            Path to saved results directory
        """
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory structure
        if category:
            exp_dir = self.experiments_dir / category / f"{experiment_name}_{timestamp}"
        else:
            exp_dir = self.experiments_dir / f"{experiment_name}_{timestamp}"
        
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        results_file = exp_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Extract and save CSV data
        self._save_csv_data(results, exp_dir)
        
        # Update index
        index_entry = {
            "path": str(exp_dir.relative_to(self.base_dir)),
            "timestamp": timestamp,
            "name": experiment_name,
            "category": category,
            "num_agents": results.get("num_agents"),
            "num_rounds": results.get("num_rounds"),
            "agent_types": self._extract_agent_types(results)
        }
        
        index_key = f"{experiment_name}_{timestamp}"
        self.index["experiments"][index_key] = index_entry
        self._save_index()
        
        self.logger.info(f"Saved experiment results to {exp_dir}")
        return exp_dir
    
    def _save_csv_data(self, results: Dict, exp_dir: Path):
        """Extract and save CSV data from results."""
        csv_dir = exp_dir / "csv"
        csv_dir.mkdir(exist_ok=True)
        
        # Save history if available
        if "history" in results and results["history"]:
            history_file = csv_dir / "history.csv"
            self._save_history_csv(results["history"], history_file)
        
        # Save agent stats
        if "agent_stats" in results:
            stats_file = csv_dir / "agent_stats.csv"
            self._save_agent_stats_csv(results["agent_stats"], stats_file)
        
        # Save summary
        summary_file = csv_dir / "summary.csv"
        self._save_summary_csv(results, summary_file)
    
    def _save_history_csv(self, history: List[Dict], file_path: Path):
        """Save round-by-round history to CSV."""
        if not history:
            return
        
        # Flatten nested agent data
        rows = []
        for round_data in history:
            base_row = {
                "round": round_data.get("round"),
                "cooperation_rate": round_data.get("cooperation_rate")
            }
            
            # Add agent-specific data
            if "agents" in round_data:
                for agent_id, agent_data in round_data["agents"].items():
                    row = base_row.copy()
                    row["agent_id"] = agent_id
                    row.update(agent_data)
                    rows.append(row)
            else:
                rows.append(base_row)
        
        # Write to CSV
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(file_path, index=False)
    
    def _save_agent_stats_csv(self, agent_stats: List[Dict], file_path: Path):
        """Save agent statistics to CSV."""
        df = pd.DataFrame(agent_stats)
        df.to_csv(file_path, index=False)
    
    def _save_summary_csv(self, results: Dict, file_path: Path):
        """Save experiment summary to CSV."""
        summary = {
            "metric": [],
            "value": []
        }
        
        # Basic metrics
        for key in ["num_agents", "num_rounds", "average_cooperation"]:
            if key in results:
                summary["metric"].append(key)
                summary["value"].append(results[key])
        
        # Type-specific metrics
        if "type_cooperation_rates" in results:
            for agent_type, rate in results["type_cooperation_rates"].items():
                summary["metric"].append(f"{agent_type}_cooperation_rate")
                summary["value"].append(rate)
        
        df = pd.DataFrame(summary)
        df.to_csv(file_path, index=False)
    
    def _extract_agent_types(self, results: Dict) -> List[str]:
        """Extract unique agent types from results."""
        agent_types = set()
        
        if "agent_stats" in results:
            for stat in results["agent_stats"]:
                if "agent_type" in stat:
                    agent_types.add(stat["agent_type"])
        
        return sorted(list(agent_types))
    
    def load_experiment(self, experiment_key: str) -> Dict[str, Any]:
        """
        Load experiment results by key.
        
        Args:
            experiment_key: Key from the index or path
            
        Returns:
            Results dictionary
        """
        if experiment_key in self.index["experiments"]:
            entry = self.index["experiments"][experiment_key]
            results_path = self.base_dir / entry["path"] / "results.json"
        else:
            # Try as direct path
            results_path = Path(experiment_key)
            if not results_path.exists():
                results_path = self.base_dir / experiment_key / "results.json"
        
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def search_experiments(self,
                          name_pattern: Optional[str] = None,
                          category: Optional[str] = None,
                          agent_types: Optional[List[str]] = None,
                          date_from: Optional[str] = None,
                          date_to: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for experiments matching criteria.
        
        Args:
            name_pattern: Pattern to match in experiment name
            category: Category filter
            agent_types: Required agent types
            date_from: Start date (YYYYMMDD format)
            date_to: End date (YYYYMMDD format)
            
        Returns:
            List of matching experiment entries
        """
        matches = []
        
        for key, entry in self.index["experiments"].items():
            # Name pattern matching
            if name_pattern and name_pattern not in entry["name"]:
                continue
            
            # Category filter
            if category and entry.get("category") != category:
                continue
            
            # Agent type filter
            if agent_types:
                entry_types = set(entry.get("agent_types", []))
                if not all(t in entry_types for t in agent_types):
                    continue
            
            # Date filters
            if date_from and entry["timestamp"][:8] < date_from:
                continue
            if date_to and entry["timestamp"][:8] > date_to:
                continue
            
            matches.append({"key": key, **entry})
        
        return sorted(matches, key=lambda x: x["timestamp"], reverse=True)
    
    def compare_experiments(self,
                           experiment_keys: List[str],
                           metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare multiple experiments.
        
        Args:
            experiment_keys: List of experiment keys to compare
            metrics: Specific metrics to compare (None for all)
            
        Returns:
            DataFrame with comparison data
        """
        comparison_data = []
        
        for key in experiment_keys:
            results = self.load_experiment(key)
            
            row = {
                "experiment": key,
                "num_agents": results.get("num_agents"),
                "num_rounds": results.get("num_rounds"),
                "avg_cooperation": results.get("average_cooperation")
            }
            
            # Add agent stats
            if "agent_stats" in results:
                for stat in results["agent_stats"]:
                    agent_key = f"agent_{stat['agent_id']}_score"
                    row[agent_key] = stat.get("total_score")
            
            # Add type-specific metrics
            if "type_cooperation_rates" in results:
                for agent_type, rate in results["type_cooperation_rates"].items():
                    row[f"{agent_type}_coop_rate"] = rate
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Filter metrics if specified
        if metrics:
            available_cols = [c for c in metrics if c in df.columns]
            df = df[["experiment"] + available_cols]
        
        return df
    
    def import_legacy_results(self,
                            legacy_path: Path,
                            experiment_name: str,
                            format_type: str = "json") -> Path:
        """
        Import results from legacy format.
        
        Args:
            legacy_path: Path to legacy results
            experiment_name: Name for imported experiment
            format_type: Format of legacy data ('json', 'csv')
            
        Returns:
            Path to imported results
        """
        self.logger.info(f"Importing legacy results from {legacy_path}")
        
        if format_type == "json":
            with open(legacy_path, 'r') as f:
                results = json.load(f)
        elif format_type == "csv":
            # Convert CSV to results format
            df = pd.read_csv(legacy_path)
            results = self._csv_to_results(df)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        # Add import metadata
        results["_imported"] = {
            "from": str(legacy_path),
            "date": datetime.now().isoformat(),
            "format": format_type
        }
        
        return self.save_experiment(results, f"imported_{experiment_name}")
    
    def _csv_to_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Convert CSV data to results format."""
        results = {
            "num_rounds": len(df["round"].unique()) if "round" in df else None,
            "history": df.to_dict("records")
        }
        
        # Extract agent stats if possible
        if "agent_id" in df:
            agent_stats = []
            for agent_id in df["agent_id"].unique():
                agent_df = df[df["agent_id"] == agent_id]
                stats = {
                    "agent_id": agent_id,
                    "total_score": agent_df["score"].sum() if "score" in df else None,
                    "cooperation_rate": agent_df["action"].mean() if "action" in df else None
                }
                agent_stats.append(stats)
            results["agent_stats"] = agent_stats
        
        return results
    
    def archive_old_results(self, days_old: int = 30):
        """
        Archive results older than specified days.
        
        Args:
            days_old: Archive results older than this many days
        """
        cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        archived_count = 0
        
        for key, entry in list(self.index["experiments"].items()):
            # Parse timestamp
            timestamp_str = entry["timestamp"]
            entry_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S").timestamp()
            
            if entry_date < cutoff_date:
                # Move to archives
                old_path = self.base_dir / entry["path"]
                new_path = self.archives_dir / entry["path"]
                
                new_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old_path), str(new_path))
                
                # Update index
                entry["archived"] = True
                entry["archive_date"] = datetime.now().isoformat()
                
                archived_count += 1
        
        self._save_index()
        self.logger.info(f"Archived {archived_count} old results")
    
    def generate_report(self, output_path: Path):
        """
        Generate a summary report of all results.
        
        Args:
            output_path: Path to save report
        """
        report = {
            "generated": datetime.now().isoformat(),
            "total_experiments": len(self.index["experiments"]),
            "categories": defaultdict(int),
            "agent_types": defaultdict(int),
            "recent_experiments": []
        }
        
        # Analyze experiments
        for entry in self.index["experiments"].values():
            if entry.get("category"):
                report["categories"][entry["category"]] += 1
            
            for agent_type in entry.get("agent_types", []):
                report["agent_types"][agent_type] += 1
        
        # Get recent experiments
        sorted_experiments = sorted(
            self.index["experiments"].items(),
            key=lambda x: x[1]["timestamp"],
            reverse=True
        )
        
        report["recent_experiments"] = [
            {"key": k, **v} for k, v in sorted_experiments[:10]
        ]
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Generated report at {output_path}")