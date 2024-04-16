import numpy as np
import random

# from playground.marl_nfg.pd_dqn_agent import PDDQNAgent
from pls.shields.shields import Shield
from torch.distributions import Categorical
import torch

# Used as keys for the agent's memory
STATE = 0
Q_VALS = 1
ACTION = 2
REWARD = 3
TERMINAL = 4
ACTION_DIST = 5
# SAFETY_DIST = 6

class PDDQNShieldedAgent(object):
    """ Agent that uses the SARSA update rule to learn Q(s,a) estimates,
        using an MLP as a function approximator. """
    def __init__(self, num_states, num_actions, func_approx=None, shield_params=None, shield=None, get_sensor_value_ground_truth=None):

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
        self.optimizer = None
        self.base_loss_fn = None
        if func_approx is not None:
            self.func_approx = func_approx
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.1

        self.gamma = 0.2
        self.learning_rate = 0.01

        self.max_history = 1000
        self.batch_size = 64
        self.epochs = 5

        # memory and bookkeeping
        self.history = []
        self.saved_model_name = ""
        self.name = "DQNShielded"

        # initialize memory for now 
        # TODO: merge functionality with self.history later
        self.prev_states = None
        self.prev_actions = None
        self.rewards = None

        # shield
        if get_sensor_value_ground_truth is not None:
            self.get_sensor_value_ground_truth = get_sensor_value_ground_truth
        else:
            self.get_sensor_value_ground_truth = lambda x: x
            print("No sensor value function provided. Asumming ground truth.")


        # agents can share a shield if a shield is passed in
        assert shield_params is None or shield is None, "Cannot pass in both shield_params and shield"
        self.shield = None
        if shield_params is not None:
            self.shield = Shield(get_sensor_value_ground_truth=self.get_sensor_value_ground_truth, **shield_params)
        elif shield is not None:
            self.shield = shield

        self.alpha = 1

        self.debug_info_history = []
        self.loss_info = []
        self.save_debug_info = True

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

        model = torch.nn.Sequential(
            torch.nn.Linear(input_len, 11),
            torch.nn.ReLU(),
            torch.nn.Linear(11, output_len)
        )

        model.to(self.device)
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.base_loss_fn = torch.nn.MSELoss().to(self.device)

        return model

    def calc_action_values(self, inputs):
        """ Return Q(s,a) for a given state s:=inputs,
            for an MLP, forward pass the inputs and get a prediction"""
        self.func_approx.eval()
        return self.func_approx(inputs) # [0] to get rid of batch dimension=1

    def random_action(self, probs=None):
        """ Choose a random action (random arm to pull), with probabilities probs
            If probs is None, then each choice is equally likely. """
        n_actions = len(probs)
        return np.random.choice(range(n_actions), p=probs)

    def e_greedy(self, Q_values):
        """ Greedily choose the action with highest est. action-value,
            or choose another random one with probability epsilon """
        n_actions = len(Q_values)

        if random.random() < self.epsilon:
            return np.random.choice(range(n_actions))
        else:
            return np.argmax(Q_values.detach().numpy())

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

        state_one_hot = torch.zeros(self.num_states)
        state_one_hot[state] = 1    
        agent_one_hot = torch.zeros(self.num_agents)
        agent_one_hot[agent] = 1

        state = torch.concatenate((state_one_hot, agent_one_hot))
        action = self.get_decision(state)

        return action

    def get_shielded_action(self, x, base_actions):
        if self.shield is None:
            sensor_values = self.get_sensor_value_ground_truth(x)
        else:
            sensor_values = self.shield.get_sensor_values(x)

        self.debug_info = {"sensor_value": sensor_values, "base_policy": base_actions}

        actions = None
        safety = -1
        if self.shield is None:
            # actions = distribution.get_actions(deterministic=deterministic)
            # log_prob = distribution.log_prob(actions)
            self.debug_info["shielded_policy"] = base_actions

            # return (actions, values, log_prob)
            actions = base_actions

        elif self.shield.differentiable:  # PLPG
            # compute the shielded policy
            # print(base_actions.shape, sensor_values.shape)
            actions = self.shield.get_shielded_policy(base_actions.unsqueeze(0), sensor_values.unsqueeze(0))
            # shielded_policy = Categorical(probs=actions)
            safety = self.shield.get_policy_safety(sensor_values.unsqueeze(0), base_actions.unsqueeze(0))
            # print(safety)
            # get the most probable action of the shielded policy if we want to use a deterministic policy,
            # otherwuse, sample an action
            # if deterministic:
            #     actions = torch.argmax(shielded_policy.probs, dim=1)
            # else:
            #     actions = shielded_policy.sample()

            # log_prob = shielded_policy.log_prob(actions)
            self.debug_info["shielded_policy"] = actions
            self.debug_info["safety"] = safety
            self.debug_info["action_safety"] = self.shield.get_action_safeties(sensor_values.unsqueeze(0))

        else:  # VSRL
            with torch.no_grad():
                actions = self.shield.get_shielded_policy_vsrl(
                    base_actions.unsqueeze(0), sensor_values.unsqueeze(0)
                )
                shielded_policy = Categorical(probs=actions)

                safety = self.shield.get_policy_safety(sensor_values.unsqueeze(0), base_actions.unsqueeze(0))
                
                # get the most probable action of the shielded policy if we want to use a deterministic policy,
                # otherwuse, sample an action
                # if deterministic:
                #     actions = torch.argmax(shielded_policy.probs, dim=1)
                # else:
                #     actions = shielded_policy.sample()

                # log_prob = distribution.log_prob(actions)
                self.debug_info["shielded_policy"] = shielded_policy.probs
                self.debug_info["safety"] = safety
                self.debug_info["action_safety"] = self.shield.get_action_safeties(sensor_values.unsqueeze(0))

                # return (actions, values, log_prob)
        
        self.debug_info_history.append(self.debug_info)
        return actions

    def get_decision(self, state):
        """ Wrapper function for returning an action for a given state,
            and also includes necessary updates to memory and epsilon """
        
        state = torch.Tensor(state).to(self.device)

        Q_values = self.calc_action_values(state.unsqueeze(0)).squeeze(0).detach()

        # softmax q vals
        Q_values_norm = torch.exp(Q_values) / torch.sum(torch.exp(Q_values))
        shielded_policy = self.get_shielded_action(state, Q_values_norm).squeeze(0)

        # self.debug_info =
        action = self.e_greedy(shielded_policy)
        # print("State: ", state)
        # print("Q_values: ", Q_values)
        # print("Q_values_norm: ", Q_values_norm)
        # print("Shielded policy: ", shielded_policy)

        if self.training:
            # update epsilon
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

            # states, q_vals, action, reward, terminal
            self.history.append([state, Q_values, action, None, None, shielded_policy])

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
            for _ in range(self.epochs):
                self.train_model()
    
    def get_safety_loss(self, sensor_values, base_actions):
        ####### Safety loss ###########################################
        if self.shield is None:
            # no shield
            safety_loss = torch.Tensor([0])
        else:
            policy_safeties = self.shield.get_policy_safety(sensor_values.unsqueeze(0), base_actions.unsqueeze(0))
            policy_safeties = policy_safeties.flatten()
            safety_loss = -torch.log(policy_safeties)
            safety_loss = torch.mean(safety_loss)

        loss = self.alpha * safety_loss
        return loss
        ###############################################################


    def train_model(self):
        """ Train the MLP function approximator using experience replay """

        if not self.training:
            return

        if len(self.history) <= self.batch_size:
            return

        self.func_approx.train()

        # uniformly sample from memory
        current_batch = random.sample(range(len(self.history)-1), self.batch_size)
        
        # generate training samples to be used with usual backprop learning methods

        X_train, y_train = [], []
        y_pred = []
        safety_augmentations = []
        for batch_idx in current_batch:
            
            cur_state = self.history[batch_idx][STATE]
            cur_action = self.history[batch_idx][ACTION]
            cur_Q_vals = self.history[batch_idx][Q_VALS]
            cur_reward = self.history[batch_idx][REWARD]
            terminal = self.history[batch_idx][TERMINAL]
            action_dist = self.history[batch_idx][ACTION_DIST]

            X_train.append(cur_state)

            # define target using DQN update rule
            target = cur_Q_vals
            if terminal:
                target[cur_action] = cur_reward
            else:
                next_Q_vals = self.history[batch_idx+1][Q_VALS]
                
                self.update_rule = "cur_reward + self.gamma*(torch.max(next_Q_vals))"
                target[cur_action] = cur_reward + self.gamma*(torch.max(next_Q_vals))

            y_train.append(target)
            # y_pred.append(cur_Q_vals)
            # TODO: need to backpropagate through action_dist or safety_dist?
            safety_augmentations.append(self.get_safety_loss(cur_state, action_dist))

        # convert to torch tensors
        X_train = torch.stack(X_train, dim=0).to(self.device)   
        y_train = torch.stack(y_train, dim=0).to(self.device)
        safety_loss = torch.mean(torch.stack(safety_augmentations, dim=0)).to(self.device)
        
        # forward pass
        # TODO: Create target network for DQN
        y_pred = self.func_approx(X_train)

        # calculate loss
        base_loss = self.base_loss_fn(y_pred, y_train)
        loss = base_loss + safety_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        losses = {"base_loss": base_loss, "safety_loss": safety_loss}
        # print(losses)
        self.loss_info.append(losses)

    def save_model(self, filename):
        """ Save a copy of the current model to file """
        torch.save(self.func_approx.state_dict(), filename)
        self.saved_model_name = filename

    def save_details(self, filename):
        """ Save a text copy of the hyperparams and other details of this agent object """
        with open(filename+".txt", 'w') as f:
            deets =  { k:v for k,v in vars(self).items() if k not in ['history','func_approx'] }
            f.write(str(deets))

        if self.save_debug_info:
            with open(filename + "_debug_info.txt", 'w') as f:
                f.write(str(self.debug_info_history))

    def load_model(self, filename=None, exploit=False):
        """ Load model for training or optional exploitative deployment """
        if filename is None:
            self.saved_model_name = filename
        self.func_approx = self.init_mlp(self.n_inputs, self.num_actions)
        self.func_approx.load_state_dict(torch.load(self.saved_model_name))
        self.func_approx.to(self.device)
        if exploit:
            self.epsilon = 0
            self.epsilon_min = 0
            self.training = False

    def begin_episode(self):
        # self.prev_states = [None]*self.num_agents
        # self.prev_actions = [None]*self.num_agents
        # self.rewards = [None]*self.num_agents
        # self.epsilon = self.epsilon_start
        pass
    
    def get_params(self):
        return self.func_approx
    
    def set_params(self, params):   
        self.func_approx = params