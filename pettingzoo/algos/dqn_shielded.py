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

class DQNShielded(object):
    """ Agent that uses the SARSA update rule to learn Q(s,a) estimates,
        using an MLP as a function approximator. """
    def __init__(self, num_states, num_actions, func_approx=None, shield_params=None, shield=None, alpha=1.0):

        self.observation_type = 'discrete'
        self.action_type = 'discrete'
        self.learning = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # input: num_states
        # output: Q(s,a)
        self.num_states = num_states
        self.num_actions = num_actions

        # set up agent
        self.n_inputs = None
        self.controls = list(range(self.num_actions))
        self.training = True
        self.eval_mode = False
    
        self.func_approx = None
        self.optimizer = None
        self.base_loss_fn = None
        if func_approx is not None:
            self.func_approx = func_approx.to(self.device)

        # Hyperparameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.1

        self.gamma = 0.999
        self.learning_rate = 0.01

        self.max_history = 1000
        self.batch_size = 128
        self.epochs = 4

        # memory and bookkeeping
        self.history = []
        self._temp_history = []
        self.saved_model_name = ""
        self.name = "DQNShielded"

        # initialize memory for now 
        # TODO: merge functionality with self.history later
        self.prev_state = None
        self.prev_action = None
        self.reward = None

        # shield
        # agents can share a shield if a shield is passed in
        assert shield_params is None or shield is None, "Cannot pass in both shield_params and shield"
        self.shield = None
        if shield_params is not None:
            self.shield = Shield(**shield_params)
        elif shield is not None:
            self.shield = shield
        elif shield_params is None and shield is None:
            self.shield = None

        # TODO: add shield to gpu
        # if self.shield is not None:
        #     self.shield.shield_layer.to(self.device)

        self.alpha = alpha

        self.debug_info_history = []
        self.loss_info = []
        self.save_debug_info = False

        self._setup()

    def _setup(self):
        """ Set up the agent for training or evaluation """

        # bookkeeping
        self.prev_state = None
        self.prev_action = None
        self.reward = None
        # set up function approximator
        self.n_inputs = self.num_states
        self.n_outputs = self.num_actions
        if self.func_approx is None:
            self.func_approx = self.init_mlp(self.n_inputs, self.num_actions)
        # set up training stuff
        self.optimizer = torch.optim.Adam(self.func_approx.parameters(), lr=self.learning_rate)
        self.base_loss_fn = torch.nn.MSELoss().to(self.device)

    def init_mlp(self, input_len, output_len):
        """ Initialize an MLP as a function approximator """

        model = torch.nn.Sequential(
            torch.nn.Linear(input_len, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            # torch.nn.ReLU(),
            # torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_len)
        )

        model.to(self.device)
        
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

        if random.random() < self.epsilon and not self.eval_mode:
            return np.random.choice(range(n_actions))
        else:
            return np.argmax(Q_values.detach().cpu().numpy())

    def act(self, states):
        """ Choose an action for each agent, given the current state """
        # if sum(states[90:120]) < 30:
        #     print([i for i, s in enumerate(states[90:120]) if s < 1])

        action = self.get_decision(states)

        self.prev_action = action
        self.prev_state = states
        
        return action

    def get_shielded_action(self, x, base_actions):
        sensor_values = x
        if self.shield is not None:
            sensor_values = self.shield.get_sensor_values(x)

        sensor_values = sensor_values.to("cpu")
        base_actions = base_actions.to("cpu")

        self.debug_info = {"sensor_value": sensor_values, "base_policy": base_actions}

        actions = None
        safety = -1
        if self.shield is None:
            self.debug_info["shielded_policy"] = base_actions
            actions = base_actions

        elif self.shield.differentiable:  # PLPG
            # compute the shielded policy
            actions = self.shield.get_shielded_policy(base_actions.unsqueeze(0), sensor_values.unsqueeze(0))
            # shielded_policy = Categorical(probs=actions)
            safety = self.shield.get_policy_safety(sensor_values.unsqueeze(0), base_actions.unsqueeze(0))

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
        return actions.to(self.device)

    def get_decision(self, state):
        """ Wrapper function for returning an action for a given state,
            and also includes necessary updates to memory and epsilon """
        
        state = torch.Tensor(state).to(self.device)

        Q_values = self.calc_action_values(state.unsqueeze(0)).squeeze(0).detach()

        # softmax q vals
        Q_values_norm = torch.exp(Q_values) / torch.sum(torch.exp(Q_values))
        shielded_policy = self.get_shielded_action(state, Q_values_norm).squeeze(0).to(self.device)

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

            # states, q_vals, action, reward, terminal, action_dist
            self._temp_history = [state, Q_values, action, None, None, shielded_policy]

        return self.controls[action]

    def update_reward(self, reward, terminal=False):
        """ Record the current reward and whether or not the state is terminal. """
        if self.training:
            self.history.append([None]*6)
            self.history[-1][STATE] = self._temp_history[STATE]
            self.history[-1][Q_VALS] = self._temp_history[Q_VALS]
            self.history[-1][ACTION] = self._temp_history[ACTION] 
            self.history[-1][REWARD] = reward
            self.history[-1][TERMINAL] = terminal
            self.history[-1][ACTION_DIST] = self._temp_history[ACTION_DIST]
            # forget the oldest memories to make room for new ones
            if len(self.history) > self.max_history:
                self.history.pop(0)        
        
        self.reward = reward

        if not self.eval_mode and self.training and self.prev_state is not None:
            for _ in range(self.epochs):
                self.train_model()
    
    def get_safety_loss(self, sensor_values_batch, base_actions_batch):
        ####### Safety loss ###########################################
        if self.shield is None:
            # no shield
            safety_loss = torch.Tensor([0]*len(sensor_values_batch))
        else:
            policy_safeties = self.shield.get_policy_safety(sensor_values_batch, base_actions_batch)
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
        current_batch = torch.Tensor(random.sample(range(len(self.history)-1), self.batch_size)).to(self.device)
        next_batch = current_batch + 1

        cur_states = torch.stack([self.history[int(i)][STATE] for i in current_batch]).to(self.device).float()
        cur_q_vals = torch.stack([self.history[int(i)][Q_VALS] for i in current_batch]).to(self.device).float()
        next_q_vals = torch.stack([self.history[int(i)][Q_VALS] for i in next_batch]).to(self.device).float()
        cur_actions = torch.stack([torch.tensor(self.history[int(i)][ACTION]) for i in current_batch]).to(self.device)
        cur_rewards = torch.stack([torch.tensor(self.history[int(i)][REWARD]) for i in current_batch]).to(self.device).float()
        cur_terminal = torch.stack([torch.tensor(self.history[int(i)][TERMINAL]) for i in current_batch]).to(self.device).bool()
        cur_action_dist = torch.stack([self.history[int(i)][ACTION_DIST] for i in current_batch]).to(self.device).float()

        X_train = cur_states
        y_train = cur_q_vals.float().clone().detach()
                
        # if it is not a terminal state, update the q value with respect to future reward
        # if it is a terminal state, the q value is just the reward
        targets = torch.mul(self.gamma, torch.max(next_q_vals, axis=1)[0])
        y_train[:, cur_actions] = cur_rewards + torch.mul(1-cur_terminal.int().float(), targets)
        

        safety_augmentations =  self.get_safety_loss(cur_states, cur_action_dist)
        safety_loss = torch.mean(safety_augmentations).to(self.device)

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

    def set_eval_mode(self, bool_val):
        """ Set the agent to evaluation mode """
        self.eval_mode = bool_val
        self.training = not bool_val

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
            self.eval_mode = True

    def begin_episode(self):
        self.prev_states = None
        # self.prev_actions = None
        # self.rewards = None
        # self.epsilon = self.epsilon_start
        pass
    
    def get_params(self):
        return self.func_approx
    
    def set_params(self, params):   
        self.func_approx = params