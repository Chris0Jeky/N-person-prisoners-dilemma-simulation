"""
Report generation for experiment results
"""

from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime
import pandas as pd


class ReportGenerator:
    """
    Generates comprehensive reports for experiment results.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_experiment_report(self, results: Dict[str, Any], report_name: str = "report"):
        """
        Generate a comprehensive HTML report for experiment results.
        
        Args:
            results: Experiment results
            report_name: Name for the report file
        """
        html_content = self._create_html_report(results)
        
        report_path = self.output_dir / f"{report_name}.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path
    
    def _create_html_report(self, results: Dict[str, Any]) -> str:
        """Create HTML content for the report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NPD Experiment Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1, h2, h3 {{
                    color: #333;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #4CAF50;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .metric {{
                    display: inline-block;
                    margin: 10px;
                    padding: 15px;
                    background-color: #f0f0f0;
                    border-radius: 5px;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #4CAF50;
                }}
                .plot {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .plot img {{
                    max-width: 100%;
                    height: auto;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>N-Person Prisoner's Dilemma Experiment Report</h1>
                <p><strong>Generated:</strong> {timestamp}</p>
                
                {self._create_summary_section(results)}
                {self._create_configuration_section(results)}
                {self._create_results_section(results)}
                {self._create_agent_details_section(results)}
                {self._create_plots_section()}
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_summary_section(self, results: Dict[str, Any]) -> str:
        """Create summary section of the report."""
        num_agents = results.get('num_agents', 'N/A')
        num_rounds = results.get('num_rounds', 'N/A')
        avg_coop = results.get('average_cooperation', 0)
        
        return f"""
        <h2>Experiment Summary</h2>
        <div class="metrics">
            <div class="metric">
                <div>Number of Agents</div>
                <div class="metric-value">{num_agents}</div>
            </div>
            <div class="metric">
                <div>Number of Rounds</div>
                <div class="metric-value">{num_rounds}</div>
            </div>
            <div class="metric">
                <div>Average Cooperation</div>
                <div class="metric-value">{avg_coop:.2%}</div>
            </div>
        </div>
        """
    
    def _create_configuration_section(self, results: Dict[str, Any]) -> str:
        """Create configuration section."""
        if 'config' not in results:
            return ""
            
        config = results['config']
        
        html = "<h2>Configuration</h2>"
        html += f"<p><strong>Experiment Name:</strong> {config.get('name', 'Unnamed')}</p>"
        
        if 'agents' in config:
            html += "<h3>Agent Configuration</h3>"
            html += "<table>"
            html += "<tr><th>ID</th><th>Type</th><th>Parameters</th></tr>"
            
            for agent in config['agents']:
                agent_id = agent['id']
                agent_type = agent['type']
                
                # Extract parameters
                params = {k: v for k, v in agent.items() if k not in ['id', 'type']}
                params_str = ', '.join(f"{k}={v}" for k, v in params.items())
                
                html += f"<tr><td>{agent_id}</td><td>{agent_type}</td><td>{params_str}</td></tr>"
            
            html += "</table>"
        
        return html
    
    def _create_results_section(self, results: Dict[str, Any]) -> str:
        """Create results summary section."""
        html = "<h2>Results Summary</h2>"
        
        if 'agent_stats' in results:
            # Calculate summary statistics
            stats = results['agent_stats']
            scores = [s['total_score'] for s in stats]
            coop_rates = [s['cooperation_rate'] for s in stats]
            
            html += f"""
            <div class="metrics">
                <div class="metric">
                    <div>Average Score</div>
                    <div class="metric-value">{sum(scores)/len(scores):.1f}</div>
                </div>
                <div class="metric">
                    <div>Score Range</div>
                    <div class="metric-value">{max(scores) - min(scores):.1f}</div>
                </div>
                <div class="metric">
                    <div>Cooperation Range</div>
                    <div class="metric-value">{max(coop_rates) - min(coop_rates):.2%}</div>
                </div>
            </div>
            """
        
        return html
    
    def _create_agent_details_section(self, results: Dict[str, Any]) -> str:
        """Create detailed agent results section."""
        if 'agent_stats' not in results:
            return ""
            
        html = "<h2>Agent Performance Details</h2>"
        html += "<table>"
        html += "<tr><th>Agent ID</th><th>Type</th><th>Total Score</th><th>Cooperation Rate</th><th>Cooperations</th><th>Defections</th></tr>"
        
        # Get agent types if available
        agent_types = {}
        if 'config' in results and 'agents' in results['config']:
            for agent in results['config']['agents']:
                agent_types[agent['id']] = agent['type']
        
        # Sort by score
        sorted_stats = sorted(results['agent_stats'], 
                            key=lambda x: x['total_score'], 
                            reverse=True)
        
        for stat in sorted_stats:
            agent_id = stat['agent_id']
            agent_type = agent_types.get(agent_id, 'Unknown')
            
            html += f"""
            <tr>
                <td>{agent_id}</td>
                <td>{agent_type}</td>
                <td>{stat['total_score']:.1f}</td>
                <td>{stat['cooperation_rate']:.2%}</td>
                <td>{stat['num_cooperations']}</td>
                <td>{stat['num_defections']}</td>
            </tr>
            """
        
        html += "</table>"
        
        return html
    
    def _create_plots_section(self) -> str:
        """Create plots section."""
        html = "<h2>Visualizations</h2>"
        
        # List of expected plot files
        plot_files = [
            ('cooperation_evolution.png', 'Cooperation Evolution'),
            ('score_distribution.png', 'Score Distribution'),
            ('agent_performance.png', 'Agent Performance'),
            ('cooperation_heatmap.png', 'Cooperation Patterns')
        ]
        
        for filename, title in plot_files:
            if (self.output_dir.parent / 'figures' / filename).exists():
                html += f"""
                <div class="plot">
                    <h3>{title}</h3>
                    <img src="../figures/{filename}" alt="{title}">
                </div>
                """
        
        return html
    
    def generate_batch_report(self, results: List[Dict[str, Any]], report_name: str = "batch_report"):
        """
        Generate report for batch experiments.
        
        Args:
            results: List of experiment results
            report_name: Name for the report file
        """
        # Create summary DataFrame
        summary_data = []
        
        for i, result in enumerate(results):
            exp_name = result.get('config', {}).get('name', f'Experiment_{i+1}')
            
            summary = {
                'Experiment': exp_name,
                'Agents': result.get('num_agents', 'N/A'),
                'Rounds': result.get('num_rounds', 'N/A'),
                'Avg Cooperation': result.get('average_cooperation', 0)
            }
            
            if 'agent_stats' in result:
                scores = [s['total_score'] for s in result['agent_stats']]
                summary['Avg Score'] = sum(scores) / len(scores) if scores else 0
                summary['Score Std'] = pd.Series(scores).std() if len(scores) > 1 else 0
            
            summary_data.append(summary)
        
        # Create DataFrame and save as CSV
        df = pd.DataFrame(summary_data)
        csv_path = self.output_dir / f"{report_name}.csv"
        df.to_csv(csv_path, index=False)
        
        # Also create HTML report
        html_path = self.output_dir / f"{report_name}.html"
        html_content = self._create_batch_html_report(df, results)
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return html_path, csv_path
    
    def _create_batch_html_report(self, df: pd.DataFrame, results: List[Dict[str, Any]]) -> str:
        """Create HTML report for batch results."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NPD Batch Experiment Report</title>
            <style>
                /* Same styles as single report */
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Batch Experiment Report</h1>
                <p><strong>Generated:</strong> {timestamp}</p>
                <p><strong>Total Experiments:</strong> {len(results)}</p>
                
                <h2>Summary Table</h2>
                {df.to_html(index=False, classes='summary-table')}
                
                <h2>Statistics</h2>
                <p><strong>Average Cooperation Across All Experiments:</strong> {df['Avg Cooperation'].mean():.2%}</p>
                <p><strong>Cooperation Range:</strong> {df['Avg Cooperation'].min():.2%} - {df['Avg Cooperation'].max():.2%}</p>
            </div>
        </body>
        </html>
        """
        
        return html