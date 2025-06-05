"""
Tests for the interactive game module.

This module tests the interactive gameplay functionality including:
- Game initialization
- Player input handling
- AI opponent behavior
- Score calculation
- UI/display functions
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock, call
from io import StringIO

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from npdl.interactive.game import InteractiveGame, main, clear_screen
from npdl.core.agents import Agent


class TestInteractiveGame:
    """Test suite for the InteractiveGame class."""
    
    def test_game_initialization(self):
        """Test game initialization with default parameters."""
        game = InteractiveGame()
        
        assert game.num_agents == 5
        assert game.num_rounds == 10
        assert game.network_type == "fully_connected"
        assert len(game.agents) == 5
        assert game.human_id == 0
        assert game.agents[0].strategy_type == "always_cooperate"  # Human player
        
    def test_game_initialization_custom_params(self):
        """Test game initialization with custom parameters."""
        game = InteractiveGame(
            num_agents=7,
            num_rounds=20,
            network_type="small_world",
            opponents=["tit_for_tat", "pavlov"],
            payoff_type="exponential",
            payoff_params={"exponent": 3}
        )
        
        assert game.num_agents == 7
        assert game.num_rounds == 20
        assert game.network_type == "small_world"
        assert game.payoff_type == "exponential"
        assert game.payoff_params["exponent"] == 3
        assert len(game.agents) == 7
        
        # Check that opponents are from specified list
        for agent in game.agents[1:]:  # Skip human player
            assert agent.strategy_type in ["tit_for_tat", "pavlov"]
            
    def test_setup_game_agent_creation(self):
        """Test that agents are properly created during setup."""
        game = InteractiveGame(num_agents=4, opponents=["q_learning", "generous_tit_for_tat"])
        
        # Check human player
        assert game.agents[0].agent_id == 0
        assert game.agents[0].strategy_type == "always_cooperate"
        
        # Check AI agents have proper parameters
        q_learning_agents = [a for a in game.agents if a.strategy_type == "q_learning"]
        for agent in q_learning_agents:
            assert hasattr(agent, 'learning_rate')
            assert hasattr(agent, 'discount_factor')
            assert hasattr(agent, 'epsilon')
            
    @patch('builtins.input', side_effect=['c'])
    def test_get_human_move_cooperate(self, mock_input):
        """Test getting cooperate move from human player."""
        game = InteractiveGame()
        move = game._get_human_move()
        assert move == "cooperate"
        
    @patch('builtins.input', side_effect=['d'])
    def test_get_human_move_defect(self, mock_input):
        """Test getting defect move from human player."""
        game = InteractiveGame()
        move = game._get_human_move()
        assert move == "defect"
        
    @patch('builtins.input', side_effect=['x', 'invalid', 'c'])
    def test_get_human_move_invalid_then_valid(self, mock_input, capsys):
        """Test handling of invalid input before valid move."""
        game = InteractiveGame()
        move = game._get_human_move()
        assert move == "cooperate"
        
        captured = capsys.readouterr()
        assert "Invalid choice" in captured.out
        assert captured.out.count("Invalid choice") == 2  # Two invalid inputs
        
    @patch('builtins.input', return_value='')
    def test_display_round_results(self, mock_input, capsys):
        """Test display of round results."""
        game = InteractiveGame(num_agents=3)
        
        moves = {0: "cooperate", 1: "defect", 2: "cooperate"}
        payoffs = {0: 3.0, 1: 5.0, 2: 3.0}
        
        game._display_round_results(0, moves, payoffs)
        
        captured = capsys.readouterr()
        assert "Round 1 Results" in captured.out
        assert "Your move: COOPERATE" in captured.out
        assert "Your payoff: 3.00" in captured.out
        assert "Neighbor moves:" in captured.out
        assert "Current scores:" in captured.out
        assert "Cooperation rate:" in captured.out
        
    def test_display_game_summary(self, capsys):
        """Test display of game summary."""
        game = InteractiveGame(num_agents=3, num_rounds=2)
        
        # Simulate some game history
        game.agents[0].score = 10.0
        game.agents[1].score = 15.0
        game.agents[2].score = 8.0
        
        game.history = [
            {"round": 0, "moves": {0: "cooperate", 1: "defect", 2: "cooperate"}, "payoffs": {0: 3, 1: 5, 2: 3}},
            {"round": 1, "moves": {0: "defect", 1: "defect", 2: "cooperate"}, "payoffs": {0: 5, 1: 1, 2: 0}}
        ]
        
        with patch('npdl.interactive.game.clear_screen'):
            game._display_game_summary()
        
        captured = capsys.readouterr()
        assert "Game Summary" in captured.out
        assert "Final Scores:" in captured.out
        assert "YOU: 10.00" in captured.out
        assert "Cooperation Rates:" in captured.out
        assert "Cooperation Rate Trend:" in captured.out
        
    @patch('builtins.input')
    @patch('npdl.interactive.game.clear_screen')
    def test_run_game_flow(self, mock_clear, mock_input):
        """Test the overall game flow."""
        # Set up input sequence: Enter to start, 'c' for round 1, Enter to continue, 'd' for round 2, Enter to continue
        mock_input.side_effect = ['', 'c', '', 'd', '']
        
        game = InteractiveGame(num_agents=2, num_rounds=2)
        game.run_game()
        
        # Check that game completed
        assert len(game.history) == 2
        assert game.history[0]['moves'][0] == 'cooperate'
        assert game.history[1]['moves'][0] == 'defect'
        
        # Check that scores were updated
        assert game.agents[0].score > 0
        
    @patch('builtins.input')
    @patch('npdl.interactive.game.clear_screen')
    def test_run_game_shows_previous_moves(self, mock_clear, mock_input, capsys):
        """Test that previous moves are shown after first round."""
        mock_input.side_effect = ['', 'c', '', 'c', '']
        
        game = InteractiveGame(num_agents=2, num_rounds=2)
        
        # Mock the other agent to always defect
        game.agents[1].choose_move = Mock(return_value='defect')
        
        game.run_game()
        
        # Check that the output mentions previous moves
        captured = capsys.readouterr()
        assert "Neighbors' previous moves:" in captured.out


class TestInteractiveGameMain:
    """Test the main function and user interaction flow."""
    
    @patch('builtins.input')
    @patch('npdl.interactive.game.clear_screen')
    def test_main_default_choices(self, mock_clear, mock_input):
        """Test main function with all default choices."""
        # All empty inputs = use defaults
        mock_input.side_effect = ['', '', '', '', '', '', 'c', '', 'd', '']
        
        with patch('npdl.interactive.game.InteractiveGame.run_game') as mock_run:
            main()
            mock_run.assert_called_once()
            
    @patch('builtins.input')
    @patch('npdl.interactive.game.clear_screen')
    def test_main_custom_choices(self, mock_clear, mock_input):
        """Test main function with custom choices."""
        # Custom inputs: 7 agents, 15 rounds, small world network, all TFT, exponential payoff
        mock_input.side_effect = ['7', '15', '2', '2', '2']
        
        with patch('npdl.interactive.game.InteractiveGame') as mock_game_class:
            mock_game_instance = Mock()
            mock_game_class.return_value = mock_game_instance
            
            main()
            
            # Check that InteractiveGame was created with correct parameters
            mock_game_class.assert_called_once()
            call_args = mock_game_class.call_args
            
            assert call_args[1]['num_agents'] == 7
            assert call_args[1]['num_rounds'] == 15
            assert call_args[1]['network_type'] == 'small_world'
            assert call_args[1]['opponents'] == ['tit_for_tat']
            assert call_args[1]['payoff_type'] == 'exponential'
            
            mock_game_instance.run_game.assert_called_once()
            
    @patch('builtins.input')
    @patch('npdl.interactive.game.clear_screen')
    def test_main_invalid_network_choice(self, mock_clear, mock_input, capsys):
        """Test main function with invalid network choice."""
        mock_input.side_effect = ['5', '10', '99', '1', '1']  # 99 is invalid
        
        with patch('npdl.interactive.game.InteractiveGame.run_game'):
            main()
            
        captured = capsys.readouterr()
        assert "Invalid choice. Using fully connected network." in captured.out
        
    @patch('builtins.input')
    @patch('npdl.interactive.game.clear_screen')
    def test_main_keyboard_interrupt(self, mock_clear, mock_input, capsys):
        """Test handling of keyboard interrupt."""
        mock_input.side_effect = KeyboardInterrupt()
        
        main()  # Should not raise exception
        
        captured = capsys.readouterr()
        assert "Game aborted. Goodbye!" in captured.out
        
    @patch('builtins.input')
    @patch('npdl.interactive.game.clear_screen')
    def test_main_exception_handling(self, mock_clear, mock_input, capsys):
        """Test handling of general exceptions."""
        mock_input.side_effect = ValueError("Invalid input")
        
        main()  # Should not raise exception
        
        captured = capsys.readouterr()
        assert "Error:" in captured.out
        assert "Game aborted." in captured.out
        
    @patch('builtins.input')
    @patch('npdl.interactive.game.clear_screen') 
    def test_main_all_network_types(self, mock_clear, mock_input):
        """Test all network type selections."""
        network_choices = [
            ('1', 'fully_connected'),
            ('2', 'small_world'),
            ('3', 'scale_free'),
            ('4', 'random')
        ]
        
        for choice, expected_type in network_choices:
            mock_input.side_effect = ['5', '10', choice, '1', '1']
            
            with patch('npdl.interactive.game.InteractiveGame') as mock_game_class:
                mock_game_instance = Mock()
                mock_game_class.return_value = mock_game_instance
                
                main()
                
                call_args = mock_game_class.call_args
                assert call_args[1]['network_type'] == expected_type
                
    @patch('builtins.input')
    @patch('npdl.interactive.game.clear_screen')
    def test_main_all_opponent_types(self, mock_clear, mock_input):
        """Test all opponent type selections."""
        opponent_choices = [
            ('1', ["tit_for_tat", "q_learning", "always_defect", "random", "pavlov"]),
            ('2', ["tit_for_tat"]),
            ('3', ["q_learning"]),
            ('4', ["always_defect"])
        ]
        
        for choice, expected_opponents in opponent_choices:
            mock_input.side_effect = ['5', '10', '1', choice, '1']
            
            with patch('npdl.interactive.game.InteractiveGame') as mock_game_class:
                mock_game_instance = Mock()
                mock_game_class.return_value = mock_game_instance
                
                main()
                
                call_args = mock_game_class.call_args
                assert call_args[1]['opponents'] == expected_opponents


class TestUtilityFunctions:
    """Test utility functions in the game module."""
    
    @patch('os.system')
    def test_clear_screen_windows(self, mock_system):
        """Test clear_screen on Windows."""
        with patch('os.name', 'nt'):
            clear_screen()
            mock_system.assert_called_once_with('cls')
            
    @patch('os.system')
    def test_clear_screen_unix(self, mock_system):
        """Test clear_screen on Unix/Linux."""
        with patch('os.name', 'posix'):
            clear_screen()
            mock_system.assert_called_once_with('clear')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])