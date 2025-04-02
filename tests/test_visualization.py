"""
Tests for data processing and visualization components.
"""
import pytest
import pandas as pd
import networkx as nx
import numpy as np
import tempfile
import os
import json

# Import visualization components
from npdl.visualization.data_processor import (
    get_payoffs_by_strategy,
    get_strategy_scores,
    get_strategy_colors,
    prepare_network_data
)
from npdl.visualization.data_loader import (
    get_cooperation_rates,
    get_strategy_cooperation_rates
)
from npdl.visualization.network_viz import (
    generate_network_positions,
    create_network_figure
)


@pytest.fixture
def sample_rounds_df():
    """Sample round-by-round dataframe for testing."""
    return pd.DataFrame({
        'round': [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
        'agent_id': [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        'move': ['cooperate', 'cooperate', 'defect', 'defect',
                 'cooperate', 'defect', 'defect', 'cooperate',
                 'cooperate', 'cooperate', 'defect', 'cooperate'],
        'payoff': [1.0, 1.0, 3.0, 3.0, 1.5, 2.5, 2.5, 1.5, 2.0, 2.0, 3.5, 2.0],
        'strategy': ['tit_for_tat', 'tit_for_tat', 'always_defect', 'q_learning',
                     'tit_for_tat', 'tit_for_tat', 'always_defect', 'q_learning',
                     'tit_for_tat', 'tit_for_tat', 'always_defect', 'q_learning']
    })


@pytest.fixture
def sample_agents_df():
    """Sample agents dataframe for testing."""
    return pd.DataFrame({
        'scenario_name': ['test'] * 4,
        'run_number': [0] * 4,
        'agent_id': [0, 1, 2, 3],
        'strategy': ['tit_for_tat', 'tit_for_tat', 'always_defect', 'q_learning'],
        'final_score': [45.0, 42.0, 85.0, 62.0]
    })


@pytest.fixture
def sample_network():
    """Create a sample network for testing visualization."""
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3)])
    return G


class TestDataProcessing:
    """Test data processing functions."""
    
    def test_get_payoffs_by_strategy(self, sample_rounds_df):
        """Test calculating payoffs by strategy."""
        payoffs = get_payoffs_by_strategy(sample_rounds_df)
        
        # Check structure
        assert "round" in payoffs.columns
        assert "strategy" in payoffs.columns
        assert "payoff" in payoffs.columns
        
        # Check content
        # Round 0: tit_for_tat = 1.0, always_defect = 3.0, q_learning = 3.0
        round0 = payoffs[payoffs["round"] == 0]
        tft_r0 = round0[round0["strategy"] == "tit_for_tat"]
        assert tft_r0["payoff"].values[0] == 1.0
        
        ad_r0 = round0[round0["strategy"] == "always_defect"]
        assert ad_r0["payoff"].values[0] == 3.0
        
        # Round 2: tit_for_tat = 2.0, always_defect = 3.5, q_learning = 2.0
        round2 = payoffs[payoffs["round"] == 2]
        tft_r2 = round2[round2["strategy"] == "tit_for_tat"]
        assert tft_r2["payoff"].values[0] == 2.0
        
        ad_r2 = round2[round2["strategy"] == "always_defect"]
        assert ad_r2["payoff"].values[0] == 3.5
    
    def test_get_strategy_scores(self, sample_agents_df):
        """Test calculating strategy scores."""
        scores = get_strategy_scores(sample_agents_df)
        
        # Check structure
        assert "strategy" in scores.columns
        assert "mean" in scores.columns
        assert "std" in scores.columns
        assert "min" in scores.columns
        assert "max" in scores.columns
        
        # Check content
        tft_row = scores[scores["strategy"] == "tit_for_tat"]
        assert tft_row["mean"].values[0] == 43.5  # Average of 45.0 and 42.0
        assert tft_row["min"].values[0] == 42.0
        assert tft_row["max"].values[0] == 45.0
        
        ad_row = scores[scores["strategy"] == "always_defect"]
        assert ad_row["mean"].values[0] == 85.0
        
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        empty_result = get_strategy_scores(empty_df)
        assert empty_result.empty
    
    def test_get_strategy_colors(self):
        """Test retrieving strategy colors."""
        colors = get_strategy_colors()
        
        # Check that we have colors for all standard strategies
        assert "always_cooperate" in colors
        assert "always_defect" in colors
        assert "tit_for_tat" in colors
        assert "q_learning" in colors
        
        # Check color format (should be hex colors like #RRGGBB)
        for strategy, color in colors.items():
            assert color.startswith("#")
            assert len(color) == 7
    
    def test_prepare_network_data(self, sample_agents_df, sample_rounds_df):
        """Test preparing data for network visualization."""
        # Filter rounds data for a specific round
        round_data = sample_rounds_df[sample_rounds_df["round"] == 1]
        
        network_data = prepare_network_data(sample_agents_df, round_data, round_num=1)
        
        # Check structure
        assert "nodes" in network_data
        assert "edges" in network_data
        assert "round" in network_data
        
        # Check content
        assert network_data["round"] == 1
        assert len(network_data["nodes"]) == 4
        
        # Check node data
        for node in network_data["nodes"]:
            agent_id = node["id"]
            assert "strategy" in node
            assert "score" in node
            assert "move" in node
            assert "payoff" in node
            
            # Verify values match the input data
            agent_row = sample_agents_df[sample_agents_df["agent_id"] == agent_id].iloc[0]
            round_row = round_data[round_data["agent_id"] == agent_id].iloc[0]
            
            assert node["strategy"] == agent_row["strategy"]
            assert node["score"] == agent_row["final_score"]
            assert node["move"] == round_row["move"]
            assert node["payoff"] == round_row["payoff"]


class TestDataLoading:
    """Test data loading functions."""
    
    def test_get_cooperation_rates(self, sample_rounds_df):
        """Test calculating overall cooperation rates."""
        coop_rates = get_cooperation_rates(sample_rounds_df)
        
        # Check structure
        assert "round" in coop_rates.columns
        assert "cooperation_rate" in coop_rates.columns
        
        # Check content
        # Round 0: 2/4 = 0.5
        assert coop_rates[coop_rates["round"] == 0]["cooperation_rate"].values[0] == 0.5
        
        # Round 1: 2/4 = 0.5
        assert coop_rates[coop_rates["round"] == 1]["cooperation_rate"].values[0] == 0.5
        
        # Round 2: 3/4 = 0.75
        assert coop_rates[coop_rates["round"] == 2]["cooperation_rate"].values[0] == 0.75
    
    def test_get_strategy_cooperation_rates(self, sample_rounds_df):
        """Test calculating cooperation rates by strategy."""
        coop_rates = get_strategy_cooperation_rates(sample_rounds_df)
        
        # Check structure
        assert "round" in coop_rates.columns
        assert "strategy" in coop_rates.columns
        assert "cooperation_rate" in coop_rates.columns
        
        # Check content
        # Round 0, tit_for_tat: 2/2 = 1.0
        r0_tft = coop_rates[(coop_rates["round"] == 0) & (coop_rates["strategy"] == "tit_for_tat")]
        assert r0_tft["cooperation_rate"].values[0] == 1.0
        
        # Round 0, always_defect: 0/1 = 0.0
        r0_ad = coop_rates[(coop_rates["round"] == 0) & (coop_rates["strategy"] == "always_defect")]
        assert r0_ad["cooperation_rate"].values[0] == 0.0
        
        # Round 1, tit_for_tat: 1/2 = 0.5
        r1_tft = coop_rates[(coop_rates["round"] == 1) & (coop_rates["strategy"] == "tit_for_tat")]
        assert r1_tft["cooperation_rate"].values[0] == 0.5
        
        # Round 2, q_learning: 1/1 = 1.0
        r2_ql = coop_rates[(coop_rates["round"] == 2) & (coop_rates["strategy"] == "q_learning")]
        assert r2_ql["cooperation_rate"].values[0] == 1.0


class TestNetworkVisualization:
    """Test network visualization functions."""
    
    def test_generate_network_positions(self, sample_network):
        """Test generating positions for network nodes."""
        # Test with different layout types
        layouts = ["spring", "circular", "kamada_kawai", "spectral", "random"]
        
        for layout in layouts:
            positions = generate_network_positions(sample_network, layout_type=layout)
            
            # Check that we have positions for all nodes
            assert len(positions) == sample_network.number_of_nodes()
            
            # Check that positions are in the expected format
            for node, pos in positions.items():
                assert len(pos) == 2  # [x, y]
                # NetworkX may return numpy float types, which aren't instances of Python float
                # So we'll just check that the position values are numbers
                assert isinstance(pos[0], (float, int, np.number))
                assert isinstance(pos[1], (float, int, np.number))
    
    def test_create_network_figure(self, sample_network):
        """Test creating a network visualization figure."""
        # Create node data
        node_data = {
            0: {'strategy': 'tit_for_tat', 'score': 45.0, 'move': 'cooperate', 'payoff': 2.0},
            1: {'strategy': 'tit_for_tat', 'score': 42.0, 'move': 'cooperate', 'payoff': 2.0},
            2: {'strategy': 'always_defect', 'score': 85.0, 'move': 'defect', 'payoff': 3.5},
            3: {'strategy': 'q_learning', 'score': 62.0, 'move': 'cooperate', 'payoff': 2.0}
        }
        
        # Create figure
        fig = create_network_figure(
            sample_network,
            node_data=node_data,
            color_by='strategy',
            size_by='payoff',
            layout_type='spring'
        )
        
        # Check that figure is created
        assert fig is not None
        
        # Test with empty graph
        empty_graph = nx.Graph()
        fig = create_network_figure(empty_graph)
        assert fig is not None
        
        # Test with no node data
        fig = create_network_figure(sample_network, node_data=None)
        assert fig is not None
        
        # Test with different color/size attributes
        fig = create_network_figure(
            sample_network,
            node_data=node_data,
            color_by='move',
            size_by='score',
            layout_type='circular'
        )
        assert fig is not None


class TestFileIO:
    """Test file input/output operations."""
    
    def test_save_load_network_structure(self, sample_network):
        """Test saving and loading network structure."""
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w+', delete=False) as temp:
            # Create network data
            network_data = {
                "nodes": list(sample_network.nodes()),
                "edges": list(sample_network.edges()),
                "network_type": "test_network",
                "network_params": {"param1": 1, "param2": 2},
                "metrics": {"num_nodes": 4, "num_edges": 4}
            }
            
            # Save network data
            json.dump(network_data, temp)
        
        try:
            # Load network data
            with open(temp.name, 'r') as f:
                loaded_data = json.load(f)
            
            # Check that data is loaded correctly
            assert loaded_data["nodes"] == list(sample_network.nodes())
            assert loaded_data["edges"] == [list(edge) for edge in sample_network.edges()]
            assert loaded_data["network_type"] == "test_network"
            assert loaded_data["network_params"]["param1"] == 1
            assert loaded_data["network_params"]["param2"] == 2
            
            # Create a new graph from loaded data
            G = nx.Graph()
            G.add_nodes_from(loaded_data["nodes"])
            G.add_edges_from(loaded_data["edges"])
            
            # Check that graph structure is preserved
            assert G.number_of_nodes() == sample_network.number_of_nodes()
            assert G.number_of_edges() == sample_network.number_of_edges()
            assert set(G.edges()) == set(sample_network.edges())
        finally:
            # Clean up
            os.unlink(temp.name)
