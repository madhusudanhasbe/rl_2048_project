# train.py
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import os
import json

from game_2048 import Game2048Env
from game_2048.agent import SarsaAgent, QLearningAgent, MonteCarloAgent, DQNAgent  # Assuming these are implemented
from game_2048.policies import random_policy, simple_heuristic_policy

def parse_args():
    parser = argparse.ArgumentParser(description="Train RL agents for 2048 game")
    parser.add_argument('--agent', type=str, choices=['random', 'heuristic', 'sarsa', 'qlearning', 'monte_carlo', 'dqn'], 
                        default='qlearning', help="Agent type")
    parser.add_argument('--episodes', type=int, default=10000, help="Number of episodes to train")
    parser.add_argument('--config', type=str, default='config.yaml', help="Configuration file")
    parser.add_argument('--save_dir', type=str, default='saved_models', help="Directory to save model")
    parser.add_argument('--eval_interval', type=int, default=500, help="Episodes between evaluations")
    parser.add_argument('--render', action='store_true', help="Render during evaluation")
    return parser.parse_args()

def create_agent(agent_type, env_config, agent_config):
    state_size = env_config.get('board_size', 4)
    action_size = 4  # Up, Down, Left, Right
    
    if agent_type == 'random':
        return BaseAgent(policy_fn=random_policy)
    elif agent_type == 'heuristic':
        return BaseAgent(policy_fn=simple_heuristic_policy)
    elif agent_type == 'sarsa':
        return SarsaAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=agent_config.get('learning_rate', 0.1),
            discount_factor=agent_config.get('gamma', 0.9),
            epsilon=agent_config.get('epsilon_start', 1.0),
            epsilon_decay=agent_config.get('epsilon_decay', 0.995),
            epsilon_min=agent_config.get('epsilon_min', 0.01)
        )
    elif agent_type == 'qlearning':
        return QLearningAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=agent_config.get('learning_rate', 0.1),
            discount_factor=agent_config.get('gamma', 0.9),
            epsilon=agent_config.get('epsilon_start', 1.0),
            epsilon_decay=agent_config.get('epsilon_decay', 0.995),
            epsilon_min=agent_config.get('epsilon_min', 0.01)
        )
    elif agent_type == 'monte_carlo':
        return MonteCarloAgent(
            state_size=state_size,
            action_size=action_size,
            discount_factor=agent_config.get('gamma', 0.9),
            epsilon=agent_config.get('epsilon_start', 1.0),
            epsilon_decay=agent_config.get('epsilon_decay', 0.995),
            epsilon_min=agent_config.get('epsilon_min', 0.01)
        )
    elif agent_type == 'dqn':
        return DQNAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=agent_config.get('learning_rate', 0.001),
            discount_factor=agent_config.get('gamma', 0.99),
            epsilon=agent_config.get('epsilon_start', 1.0),
            epsilon_decay=agent_config.get('epsilon_decay', 0.995),
            epsilon_min=agent_config.get('epsilon_min', 0.01),
            memory_size=agent_config.get('replay_buffer_size', 10000),
            batch_size=agent_config.get('batch_size', 32),
            target_update_frequency=agent_config.get('target_update_frequency', 10)
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def evaluate_agent(agent, env, num_episodes=10, render=False):
    scores = []
    max_tiles = []
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_score = 0
        
        while not done:
            action = agent.choose_action(env)
            next_state, reward, done, info = env.step(action)
            episode_score += reward
            state = next_state
            
            if render:
                env.render()
                import time
                time.sleep(0.1)
        
        scores.append(episode_score)
        max_tiles.append(np.max(state))
    
    return {
        'mean_score': np.mean(scores),
        'max_score': np.max(scores),
        'max_tile': np.max(max_tiles),
        'tile_counts': {
            '2048': sum(1 for t in max_tiles if t >= 2048),
            '1024': sum(1 for t in max_tiles if t >= 1024),
            '512': sum(1 for t in max_tiles if t >= 512),
            '256': sum(1 for t in max_tiles if t >= 256),
            '128': sum(1 for t in max_tiles if t >= 128)
        }
    }

def train(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment
    env_config = config.get('environment', {})
    env = Game2048Env(size=env_config.get('board_size', 4))
    
    # Create agent
    agent_config = config.get('agent', {})
    agent = create_agent(args.agent, env_config, agent_config)
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training loop
    evaluation_results = []
    
    for episode in tqdm(range(args.episodes)):
        state = env.reset()
        done = False
        total_reward = 0
        
        if args.agent == 'monte_carlo':
            agent.start_episode()
        
        while not done:
            action = agent.choose_action(env)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
        
        # Periodic evaluation
        if (episode + 1) % args.eval_interval == 0 or episode == 0:
            eval_results = evaluate_agent(agent, env, num_episodes=10, render=args.render and (episode + 1) == args.episodes)
            eval_results['episode'] = episode + 1
            evaluation_results.append(eval_results)
            
            print(f"Episode {episode + 1}/{args.episodes}")
            print(f"Mean Score: {eval_results['mean_score']:.2f}, Max Tile: {eval_results['max_tile']}")
            print(f"Tile Counts: {eval_results['tile_counts']}")
            
            # Save model
            model_path = os.path.join(args.save_dir, f"{args.agent}_episode_{episode+1}.model")
            agent.save_model(model_path)
    
    # Save evaluation results
    results_path = os.path.join(args.save_dir, f"{args.agent}_results.json")
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Plot learning curves
    if hasattr(agent, 'metrics') and agent.metrics["losses"]:
        plt.figure(figsize=(10, 5))
        plt.plot(agent.metrics["losses"])
        plt.title(f"Learning Curve for {args.agent}")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(args.save_dir, f"{args.agent}_learning_curve.png"))
    
    return evaluation_results

if __name__ == "__main__":
    args = parse_args()
    train(args)