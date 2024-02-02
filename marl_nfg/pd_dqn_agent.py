import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random

# Used as keys for the agent's memory
STATE = 0
Q_VALS = 1
ACTION = 2
REWARD = 3
TERMINAL = 4

class PDDQNAgent(object):
    """ Agent that uses the SARSA update rule to learn Q(s,a) estimates,
        using an MLP as a function approximator. """
    def __init__(self, num_states, num_actions, func_approx=None):

        self.observation_type = 'discrete'
        self.action_type = 'discrete'
        self.learning = True

        # input: num_states + num_agents
        # output: Q(s,a)
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_agents = None

        # set up agent
        self.n_inputs = None
        self.controls = list(range(self.num_actions))
        self.training = True
        self.eval_mode = False
    
        self.func_approx = None
        if func_approx is not None:
            self.func_approx = func_approx
        
        # Hyperparameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.9985
        self.epsilon_min = 0.1

        self.gamma = 0.2
        self.learning_rate = 0.01

        self.max_history = 1000
        self.batch_size = 10

        # memory and bookkeeping
        self.history = []
        self.saved_model_name = ""
        self.name = "DQN"

        # initialize memory for now 
        # TODO: merge functionality with self.history later
        self.prev_states = None
        self.prev_actions = None
        self.rewards = None

    def update_n_agents(self, n_agents):
        """ Update the number of agents in the environment """
        self.num_agents = n_agents
        # bookkeeping
        self.prev_states = [None]*self.num_agents
        self.prev_actions = [None]*self.num_agents
        self.rewards = [None]*self.num_agents
        # set up function approximator
        self.n_inputs = self.num_states + self.num_agents
        self.n_outputs = self.num_actions
        if self.func_approx is None:
            self.func_approx = self.init_mlp(self.n_inputs, self.num_actions)

    def init_mlp(self, input_len, output_len):
        """ Initialize an MLP as a function approximator """

        input_layer = keras.Input(shape=input_len, name='input_layer')
        dense_1 = keras.layers.Dense(11, name='dense_1', activation='relu')(input_layer)
        output_layer = keras.layers.Dense(output_len, name = 'otuput_layer', activation='linear')(dense_1)

        model = keras.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='MSE', metrics=['mse'])
        model.summary()

        return model

    def calc_action_values(self, inputs):
        """ Return Q(s,a) for a given state s:=inputs,
            for an MLP, forward pass the inputs and get a prediction"""
        return self.func_approx.predict(inputs, verbose=0)[0]

    def random_action(self, probs=None):
        """ Choose a random action (random arm to pull), with probabilities probs
            If probs is None, then each choice is equally likely. """
        n_actions = len(probs)
        return np.random.choice(range(n_actions), p=probs)

    def e_greedy(self, Q_values):
        """ Greedily choose the action with highest est. action-value,
            or choose another random one with probability epsilon """
        n_actions = len(Q_values)
        probs = np.full(n_actions, self.epsilon/n_actions)
        probs[np.argmax(Q_values)] += 1-self.epsilon
        return self.random_action(probs)

    def act(self, states):
        """ Choose an action for each agent, given the current state """
        if self.num_agents is None:
            raise Exception("Number of agents not set. Call update_n_agents() first.")
            
        actions = []
        for agent in range(self.num_agents):
            actions.append(self.act_single(states, agent))

        self.prev_actions = actions
        self.prev_states = states
        
        return actions

    def act_single(self, states, agent):
        """ Choose an action for a single agent, given the current state """
        # shape of states: (num_agents, num_states)
        # expected input to MLP: (num_agents + num_states, 1)
        # expected output of MLP: (num_actions, 1)
        if states[agent] is None:
            state = 0
        else:
            state = states[agent]

        state_one_hot = np.zeros(self.num_states)
        state_one_hot[state] = 1    
        agent_one_hot = np.zeros(self.num_agents)
        agent_one_hot[agent] = 1

        state = np.concatenate((state_one_hot, agent_one_hot))
        action = self.get_decision(state)

        return action

    def get_decision(self, state):
        """ Wrapper function for returning an action for a given state,
            and also includes necessary updates to memory and epsilon """
        Q_values = self.calc_action_values(np.array([state]))
        action = self.e_greedy(Q_values)

        if self.training:
            # update epsilon
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

            # states, q_vals, action, reward, terminal
            self.history.append([state, Q_values, action, None, None])

            # forget the oldest memories to make room for new ones
            if len(self.history) > self.max_history:
                self.history.pop(0)

        return self.controls[action]

    def update_reward(self, rewards, terminal=False):
        """ Record the current reward and whether or not the state is terminal. """
        if self.training:
            # update it for each agent in order
            for agent_idx, reward in enumerate(rewards):
                placing_index = len(self.history) - self.num_agents + agent_idx
                self.history[placing_index][REWARD] = reward
                self.history[placing_index][TERMINAL] = terminal
        
        self.rewards = rewards

        if self.prev_actions is not None and not self.eval_mode:
            self.train_model()
    
    def train_model(self):
        """ Train the MLP function approximator using experience replay """

        if not self.training:
            return

        if len(self.history) <= self.batch_size:
            return

        # uniformly sample from memory
        current_batch = random.sample(range(len(self.history)-1), self.batch_size)
        
        # generate training samples to be used with usual backprop learning methods

        X_train, y_train = [], []
        for batch_idx in current_batch:
            
            cur_state = self.history[batch_idx][STATE]
            cur_action = self.history[batch_idx][ACTION]
            cur_Q_vals = self.history[batch_idx][Q_VALS]
            cur_reward = self.history[batch_idx][REWARD]
            terminal = self.history[batch_idx][TERMINAL]

            X_train.append(cur_state)

            # define target using SARSA update rule
            target = cur_Q_vals
            if terminal:
                target[cur_action] = cur_reward
            else:
                next_Q_vals = self.history[batch_idx+1][Q_VALS]
                # print(next_action, next_Q_vals)
                self.update_rule = "cur_reward + self.gamma*(np.max(next_Q_vals))"
                target[cur_action] = cur_reward + self.gamma*(np.max(next_Q_vals))

            y_train.append(target)
            
        X_train = np.array(X_train) 
        y_train = np.array(y_train)

        self.func_approx.fit(X_train, y_train, verbose=0)

    def save_model(self, filename):
        """ Save a copy of the current model to file """
        self.func_approx.save(filename)
        self.saved_model_name = filename

    def save_details(self, filename):
        """ Save a text copy of the hyperparams and other details of this agent object """
        with open(filename, 'w') as f:
            deets =  { k:v for k,v in vars(self).items() if k not in ['history','func_approx'] }
            f.write(str(deets))

    def load_model(self, filename, exploit=False):
        """ Load model for training or optional exploitative deployment """
        self.func_approx = tf.keras.models.load_model(filename)
        if exploit:
            self.epsilon = 0
            self.epsilon_min = 0
            self.training = False

    def begin_episode(self):
        self.prev_states = [None]*self.num_agents
        self.prev_actions = [None]*self.num_agents
        self.rewards = [None]*self.num_agents
        # self.epsilon = self.epsilon_start
        pass
    
    def get_params(self):
        return self.func_approx
    
    def set_params(self, params):   
        self.func_approx = params