# game_2048/policies.py
import random
import numpy as np

def random_policy(env):
    """Selects a random action from the available legal actions."""
    legal_actions = env.get_legal_actions()
    if not legal_actions:
        # If no legal moves, game is likely over or stuck, return any action (e.g., 0)
        # The environment step function should handle the consequences
        return 0 # Or random.choice(env.action_space)
    return random.choice(legal_actions)

def simple_heuristic_policy(env):
    """
    A very simple heuristic: prioritize moves that result in the highest immediate score gain.
    Breaks ties randomly among the best-scoring moves, or randomly if no move increases score.
    """
    legal_actions = env.get_legal_actions()
    if not legal_actions:
        return 0 # No legal moves

    best_actions = []
    max_reward = -1
    original_board = np.copy(env.board)
    original_score = env.score

    for action in legal_actions:
        # Simulate the move to see the reward
        temp_board, reward = env._perform_move(action) # Use internal helper for simulation
        # Reset state after check - crucial!
        env.board = np.copy(original_board)
        env.score = original_score

        if reward > max_reward:
            max_reward = reward
            best_actions = [action]
        elif reward == max_reward:
            best_actions.append(action)

    if not best_actions: # Should not happen if legal_actions is not empty, but safeguard
         return random.choice(legal_actions)

    # If max_reward is still 0 (no move increased score), just pick a random legal move
    if max_reward <= 0:
        return random.choice(legal_actions)
    else:
        # Pick randomly among the best scoring actions
        return random.choice(best_actions)


# --- Placeholder for more advanced policies ---
# class DQNAgentPolicy:
#     def __init__(self, model, epsilon):
#         self.model = model
#         self.epsilon = epsilon
#
#     def __call__(self, state, legal_actions):
#         # Epsilon-greedy logic using the DQN model
#         pass