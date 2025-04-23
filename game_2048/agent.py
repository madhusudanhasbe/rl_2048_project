# game_2048/agent.py

class BaseAgent:
    """Base class for RL agents."""
    def __init__(self, policy_fn):
        self.policy = policy_fn # Stores the function used to select actions

    def choose_action(self, env):
        """Chooses an action based on the environment state using the assigned policy."""
        # The policy function might need the environment to query legal actions or state
        return self.policy(env)

    def learn(self, state, action, reward, next_state, done):
        """Placeholder for the learning step. Specific agents will override this."""
        pass # Base agent doesn't learn

    def save_model(self, filepath):
         """Placeholder for saving learned parameters."""
         print(f"Save model not implemented for {self.__class__.__name__}")
         pass

    def load_model(self, filepath):
         """Placeholder for loading learned parameters."""
         print(f"Load model not implemented for {self.__class__.__name__}")
         pass

# --- Example of a specific Agent (e.g., for Q-Learning or DQN later) ---
# class DQNAgent(BaseAgent):
#     def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
#         super().__init__(policy_fn=self._dqn_policy) # Policy is internal method
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.memory = [] # Replay buffer
#         self.gamma = gamma       # discount rate
#         self.epsilon = epsilon   # exploration rate
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = epsilon_decay
#         self.learning_rate = learning_rate
#         self.model = self._build_model() # Keras/TensorFlow/PyTorch model
#         self.target_model = self._build_model()
#         self.update_target_model()

#     def _build_model(self):
#         # Define and compile the neural network model
#         pass

#     def update_target_model(self):
#         # Copy weights from model to target_model
#         pass

#     def remember(self, state, action, reward, next_state, done):
#         # Add experience to replay buffer
#         pass

#     def _dqn_policy(self, env): # Note: policy might need direct state access for NN
#         state = env.get_state() # Or a transformed state
#         # Epsilon-greedy action selection using self.model
#         pass

#     def choose_action(self, env): # Override base method if needed
#         return self._dqn_policy(env)

#     def learn(self, batch_size=32): # Learn from replay buffer
#         # Sample batch from memory
#         # Calculate target Q-values using target_model
#         # Train self.model on the batch
#         # Update epsilon
#         pass

#     def replay(self, batch_size):
#         self.learn(batch_size) # Call the learning logic

#     # Add save/load methods for the NN model weights

class SarsaAgent(BaseAgent):
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        super().__init__(policy_fn=self._sarsa_policy)
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}  # State-action value function
        self.metrics = {"losses": [], "rewards": [], "max_tiles": []}
        
    def _sarsa_policy(self, env):
        state = self._process_state(env.get_state())
        legal_actions = env.get_legal_actions()
        
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return np.random.choice(legal_actions)
        
        # Get Q values for current state
        q_values = [self.get_q_value(state, a) for a in legal_actions]
        best_action_idx = np.argmax(q_values)
        return legal_actions[best_action_idx]
    
    def _process_state(self, state):
        # Convert numpy array to tuple for dictionary key
        return tuple(map(tuple, state))
    
    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)
    
    def learn(self, state, action, reward, next_state, done):
        state = self._process_state(state)
        next_state = self._process_state(next_state)
        
        # Get current Q value
        current_q = self.get_q_value(state, action)
        
        if done:
            # Terminal state
            target_q = reward
        else:
            # SARSA: Use next action from policy (not max Q)
            next_action = self._sarsa_policy(next_state)
            next_q = self.get_q_value(next_state, next_action)
            target_q = reward + self.gamma * next_q
        
        # Update Q value
        new_q = current_q + self.lr * (target_q - current_q)
        self.q_table[(state, action)] = new_q
        
        # Track loss for metrics
        loss = (target_q - current_q) ** 2
        self.metrics["losses"].append(loss)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({'q_table': self.q_table, 'epsilon': self.epsilon}, f)
    
    def load_model(self, filepath):
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.epsilon = data['epsilon']

# ============================================

class QLearningAgent(BaseAgent):
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        super().__init__(policy_fn=self._q_policy)
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}
        self.metrics = {"losses": [], "rewards": [], "max_tiles": []}
        
    def _q_policy(self, env):
        state = self._process_state(env.get_state())
        legal_actions = env.get_legal_actions()
        
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return np.random.choice(legal_actions)
        
        # Get Q values for current state
        q_values = [self.get_q_value(state, a) for a in legal_actions]
        best_action_idx = np.argmax(q_values)
        return legal_actions[best_action_idx]
    
    def _process_state(self, state):
        # Convert numpy array to tuple for dictionary key
        return tuple(map(tuple, state))
    
    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)
    
    def learn(self, state, action, reward, next_state, done):
        state = self._process_state(state)
        next_state = self._process_state(next_state)
        
        # Get current Q value
        current_q = self.get_q_value(state, action)
        
        if done:
            # Terminal state
            target_q = reward
        else:
            # Q-Learning: Use max Q value for next state
            next_actions = self.get_legal_actions(next_state)
            next_q_values = [self.get_q_value(next_state, a) for a in next_actions]
            max_next_q = max(next_q_values) if next_q_values else 0
            target_q = reward + self.gamma * max_next_q
        
        # Update Q value
        new_q = current_q + self.lr * (target_q - current_q)
        self.q_table[(state, action)] = new_q
        
        # Track loss for metrics
        loss = (target_q - current_q) ** 2
        self.metrics["losses"].append(loss)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_legal_actions(self, state):
        # Since we can't directly call env.get_legal_actions for a state,
        # we would need to simulate legal actions for a given state
        # For now, return all actions as an approximation
        return [0, 1, 2, 3]
    
    def save_model(self, filepath):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({'q_table': self.q_table, 'epsilon': self.epsilon}, f)
    
    def load_model(self, filepath):
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.epsilon = data['epsilon']

# ============================================

class MonteCarloAgent(BaseAgent):
    def __init__(self, state_size, action_size, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        super().__init__(policy_fn=self._mc_policy)
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}  # State-action value function
        self.returns = {}  # Returns for each state-action pair
        self.episode_memory = []  # Store (state, action, reward) for current episode
        self.metrics = {"losses": [], "rewards": [], "max_tiles": []}
        
    def _mc_policy(self, env):
        state = self._process_state(env.get_state())
        legal_actions = env.get_legal_actions()
        
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return np.random.choice(legal_actions)
        
        # Get Q values for current state
        q_values = [self.get_q_value(state, a) for a in legal_actions]
        best_action_idx = np.argmax(q_values)
        return legal_actions[best_action_idx]
    
    def _process_state(self, state):
        # Convert numpy array to tuple for dictionary key
        return tuple(map(tuple, state))
    
    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)
    
    def start_episode(self):
        self.episode_memory = []
    
    def step(self, state, action, reward):
        # Store experience for the current episode
        state = self._process_state(state)
        self.episode_memory.append((state, action, reward))
    
    def end_episode(self):
        if not self.episode_memory:
            return  # No experiences to learn from
        
        # Calculate discounted returns for all steps
        G = 0
        loss_sum = 0
        
        # Process the episode in reverse to calculate returns
        for t in range(len(self.episode_memory) - 1, -1, -1):
            state, action, reward = self.episode_memory[t]
            G = reward + self.gamma * G
            
            # Every-visit MC approach
            # Update returns count
            sa_pair = (state, action)
            if sa_pair not in self.returns:
                self.returns[sa_pair] = []
            
            # Store return for this visit
            self.returns[sa_pair].append(G)
            
            # Update Q-value to average of returns
            old_q = self.get_q_value(state, action)
            new_q = sum(self.returns[sa_pair]) / len(self.returns[sa_pair])
            self.q_table[sa_pair] = new_q
            
            # Track loss for metrics
            loss = (new_q - old_q) ** 2
            loss_sum += loss
        
        # Add average loss for the episode
        if len(self.episode_memory) > 0:
            self.metrics["losses"].append(loss_sum / len(self.episode_memory))
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Clear episode memory
        self.episode_memory = []
    
    def learn(self, state, action, reward, next_state, done):
        # For MC, we store experiences and learn at the end of episode
        self.step(state, action, reward)
        if done:
            self.end_episode()
    
    def save_model(self, filepath):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table, 
                'returns': self.returns, 
                'epsilon': self.epsilon
            }, f)
    
    def load_model(self, filepath):
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.returns = data['returns']
            self.epsilon = data['epsilon']

# ============================================

import numpy as np
import tensorflow as tf
from collections import deque
import random

class DQNAgent(BaseAgent):
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, 
                 memory_size=10000, batch_size=32, target_update_frequency=10):
        super().__init__(policy_fn=self._dqn_policy)
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target_update_counter = 0
        self.target_update_frequency = target_update_frequency
        
        # Create two models - main and target
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        self.metrics = {"losses": [], "rewards": [], "max_tiles": []}
    
    def _build_model(self):
        # Neural Network model
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(self.state_size, self.state_size)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
    
    def _dqn_policy(self, env):
        state = env.get_state()
        legal_actions = env.get_legal_actions()
        
        if np.random.rand() <= self.epsilon:
            return np.random.choice(legal_actions)
        
        # Process state for NN input (normalize)
        state_normalized = self._normalize_state(state)
        state_tensor = np.expand_dims(state_normalized, axis=0)
        
        # Get action values
        q_values = self.model.predict(state_tensor, verbose=0)[0]
        
        # Filter to only legal actions and select best
        legal_q_values = [(action, q_values[action]) for action in legal_actions]
        best_action = max(legal_q_values, key=lambda x: x[1])[0]
        
        return best_action
    
    def _normalize_state(self, state):
        # Log transform and normalize board state
        state = np.copy(state).astype(float)
        state[state > 0] = np.log2(state[state > 0])
        # Normalize to [0, 1]
        if np.max(state) > 0:
            state = state / 11.0  # Max log2 value is 2048 -> log2(2048) = 11
        return state
    
    def remember(self, state, action, reward, next_state, done):
        # Store experience in replay memory
        state = self._normalize_state(state)
        next_state = self._normalize_state(next_state)
        self.memory.append((state, action, reward, next_state, done))
    
    def learn(self, state, action, reward, next_state, done):
        # Add experience to memory
        self.remember(state, action, reward, next_state, done)
        
        # Learn if enough samples in memory
        if len(self.memory) >= self.batch_size:
            self._replay()
        
        # Update target network
        if done:
            self.target_update_counter += 1
            if self.target_update_counter >= self.target_update_frequency:
                self.update_target_model()
                self.target_update_counter = 0
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
    
    def _replay(self):
        # Sample batch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Get current Q values
        targets = self.model.predict(states, verbose=0)
        
        # Get next Q values from target model
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Update Q values
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the model
        history = self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)
        
        # Record loss for metrics
        self.metrics["losses"].append(history.history['loss'][0])
    
    def save_model(self, filepath):
        self.model.save(filepath)
    
    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)
        self.update_target_model()