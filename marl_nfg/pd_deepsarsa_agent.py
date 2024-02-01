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

class DSARSAAgent:
	""" Agent that uses the SARSA update rule to learn Q(s,a) estimates,
		using an MLP as a function approximator. """
	def __init__(self, n_inputs, controls, func_approx=None):

		# set up agent
		self.n_inputs = n_inputs
		self.controls = controls
		self.training = True

		if func_approx is not None:
			self.func_approx = func_approx
		else:
			self.func_approx = self.init_mlp(n_inputs, len(controls))

		# Hyperparameters
		self.epsilon = 1.0
		self.epsilon_decay = 0.9999985
		self.epsilon_min = 0.1

		self.gamma = 0.99
		self.learning_rate = 0.001

		self.max_history = 100000
		self.batch_size = 45

		# memory and bookkeeping
		self.history = []
		self.saved_model_name = ""

	def init_mlp(self,input_len, output_len):
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
		return self.func_approx.predict(inputs)[0]

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

	def record_reward(self, reward, terminal):
		""" Record the current reward and whether or not the state is terminal. """
		if self.training:
			self.history[-1][REWARD] = reward
			self.history[-1][TERMINAL] = terminal
	
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
				next_action = self.history[batch_idx+1][ACTION]
				next_Q_vals = self.history[batch_idx+1][Q_VALS]
				self.update_rule = "cur_reward + self.gamma*(next_Q_vals[next_action])"
				target[cur_action] = cur_reward + self.gamma*(next_Q_vals[next_action])

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