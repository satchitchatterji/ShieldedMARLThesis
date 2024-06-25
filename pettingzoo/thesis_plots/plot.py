import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import os

def load_data(base, file, which = "all"):
    data = []
    if which == "all" or which == "train":
        with open(f"{base}/{file}_train.pk", "rb") as f:
            train = pk.load(f)
            data.append(train)

    if which == "all" or which == "eval":
        with open(f"{base}/{file}_eval.pk", "rb") as f:
            eval = pk.load(f)
            data.append(eval)

    if which == "all" or which == "safety":
        with open(f"{base}/{file}_safety.pk", "rb") as f:
            safeties = pk.load(f)
            data.append(safeties)

    if which == "all" or which == "eval_eps":    
        with open(f"{base}/{file}_eval_eps.pk", "rb") as f:
            eval_eps = pk.load(f)   
            data.append(eval_eps)

    return tuple(data)

def cumulative_mean(data):

    if len(data) == 0:
        return np.array([])
    
    cumulative_sum = 0
    cumulative_means = []
    
    for i, value in enumerate(data):
        cumulative_sum += value
        cumulative_means.append(cumulative_sum / (i + 1))
    
    return np.array(cumulative_means)

def rolling_average(data, window_size=100):
    if type(window_size) ==str:
        return cumulative_mean(data)
    
    if window_size < 1:
        raise ValueError("Window size must be at least 1")
    elif window_size == 1:
        return data

    cumsum = np.cumsum(np.insert(data, 0, 0)) 
    rolling_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    
    return rolling_avg


def plot_rewards(reward_hists, ax, label="", rolling_window=1):
    # plot mean rewards per episode
    agent_rewards = [[np.mean(reward_hist[agent]) for reward_hist in reward_hists] for agent in reward_hists[0].keys()]
    for a, agent in enumerate(reward_hists[0].keys()):
        if label == "": 
            label=agent
        ax.plot(rolling_average(agent_rewards[a],rolling_window), label=label)

    if len(reward_hists[0].keys()) > 1:
        mean = [np.mean([np.mean(reward_hist[agent]) for agent in reward_hist.keys()]) for reward_hist in reward_hists]
        ax.plot(rolling_average(mean, rolling_window), label="mean", color="black", linestyle="--")

    return ax

############################################ PLOT SUM REWARDS ############################################

def plot_rewards_sum(reward_hists, ax, label="", rolling_window=1):
    # plot sum of rewards per episode
    agent_rewards = [[np.sum(reward_hist[agent]) for reward_hist in reward_hists] for agent in reward_hists[0].keys()]
    for a, agent in enumerate(reward_hists[0].keys()):
        if label == "": 
            label=agent
        ax.plot(rolling_average(agent_rewards[a],rolling_window), label=label)


    if len(reward_hists[0].keys()) > 1:
        mean = [np.mean([np.sum(reward_hist[agent]) for agent in reward_hist.keys()]) for reward_hist in reward_hists]
        ax.plot(rolling_average(mean, rolling_window), label="mean", color="black", linestyle="--")

    return ax


############################################ PLOT EVAL MEAN REWARDS ############################################

def plot_eval_mean(eval_hists, eval_episodes, ax, label="", rolling_window=1):
    # compute eval means and stds
    eval_means = {}
    eval_stds = {}

    for a, agent in enumerate(eval_hists[0][0].keys()):
        eval_means[agent] = []
        eval_stds[agent] = []
        for eval_hist in eval_hists:
            eval_means[agent].append(np.mean([np.mean(eval_reward_hist[agent]) for eval_reward_hist in eval_hist]))
            eval_stds[agent].append(np.std([np.mean(eval_reward_hist[agent]) for eval_reward_hist in eval_hist]))


    # plot eval info
    for a, agent in enumerate(eval_hists[0][0].keys()):
        ax.errorbar(eval_episodes, rolling_average(eval_means[agent], rolling_window), yerr=rolling_average(eval_stds[agent], rolling_window), label=label, capsize=5, marker="x")

    if len(eval_hists[0][0].keys()) > 1:
        # compute overall mean and std
        eval_means["mean"] = []
        eval_stds["mean"] = []
        for eval_hist in eval_hists:
            eval_means["mean"].append(np.mean([np.mean([np.mean(eval_reward_hist[agent]) for agent in eval_reward_hist.keys()]) for eval_reward_hist in eval_hist]))
            eval_stds["mean"].append(np.std([np.mean([np.mean(eval_reward_hist[agent]) for agent in eval_reward_hist.keys()]) for eval_reward_hist in eval_hist]))
        ax.errorbar(eval_episodes, rolling_average(eval_means["mean"], rolling_window), yerr=rolling_average(eval_stds["mean"], rolling_window), color="black", linestyle="--", capsize=5, marker="x")

    return ax


############################################ PLOT EVAL SUM REWARDS ############################################

def plot_eval_sum(eval_hists, eval_episodes, ax, label="", rolling_window=1):
    # compute eval sums and stds
    eval_sums = {}
    eval_stds = {}

    for a, agent in enumerate(eval_hists[0][0].keys()):
        eval_sums[agent] = []
        eval_stds[agent] = []
        for eval_hist in eval_hists:
            eval_sums[agent].append(np.mean([np.sum(eval_reward_hist[agent]) for eval_reward_hist in eval_hist]))
            eval_stds[agent].append(np.std([np.sum(eval_reward_hist[agent]) for eval_reward_hist in eval_hist]))

    # plot eval info
    for a, agent in enumerate(eval_hists[0][0].keys()):
        ax.errorbar(eval_episodes, rolling_average(eval_sums[agent], rolling_window), yerr=rolling_average(eval_stds[agent], rolling_window), label=label, capsize=5, marker="x")
    

    if len(eval_hists[0][0].keys()) > 1:        # compute overall mean and std
        eval_sums["mean"] = []
        eval_stds["mean"] = []
        for eval_hist in eval_hists:
            eval_sums["mean"].append(np.mean([np.mean([np.sum(eval_reward_hist[agent]) for agent in eval_reward_hist.keys()]) for eval_reward_hist in eval_hist]))
            eval_stds["mean"].append(np.std([np.mean([np.sum(eval_reward_hist[agent]) for agent in eval_reward_hist.keys()]) for eval_reward_hist in eval_hist]))

        ax.errorbar(eval_episodes, rolling_average(eval_sums["mean"], rolling_window), yerr=rolling_average(eval_stds["mean"], rolling_window), label=label, color="black", linestyle="--", capsize=5, marker="x")

    return ax

############################################ PLOT EVAL MEAN SAFETY ############################################

def plot_eval_safety(eval_safeties, eval_episodes, ax, label=""):
   
    # plot safety info
    # compute eval means and stds
    eval_means = {}
    eval_stds = {}

    for a, agent in enumerate(eval_safeties[0][0].keys()):
        eval_means[agent] = []
        eval_stds[agent] = []
        for eval_hist in eval_safeties:
            eval_means[agent].append(np.mean([np.mean(eval_reward_hist[agent]) for eval_reward_hist in eval_hist]))
            eval_stds[agent].append(np.std([np.mean(eval_reward_hist[agent]) for eval_reward_hist in eval_hist]))

    # plot eval info
    for a, agent in enumerate(eval_safeties[0][0].keys()):
        ax.errorbar(eval_episodes, eval_means[agent], yerr=eval_stds[agent], label=label, capsize=5, marker="x")
    
    if len(eval_safeties[0][0].keys()) > 1:
        # compute overall mean and std
        eval_means["mean"] = []
        eval_stds["mean"] = []
        for eval_hist in eval_safeties:
            eval_means["mean"].append(np.mean([np.mean([np.mean(eval_reward_hist[agent]) for agent in eval_reward_hist.keys()]) for eval_reward_hist in eval_hist]))
            eval_stds["mean"].append(np.std([np.mean([np.mean(eval_reward_hist[agent]) for agent in eval_reward_hist.keys()]) for eval_reward_hist in eval_hist]))

        ax.errorbar(eval_episodes, eval_means["mean"], yerr=eval_stds["mean"], label=label, color="black", linestyle="--", capsize=5, marker="x")

    return ax


if __name__ == "__main__":
    base = "../histories/CartSafe-v0"
    base_configs = "../configs/CartSafe-v0"
    files = {"IPPO": "2024-06-18_153958", "SIPPO": "2024-06-18_154455"}

    fig, axs = plt.subplots(3, 2, figsize=(20, 15))
    axs = axs.flatten()

    for i, (algo_name, file) in enumerate(files.items()):
        train, eval, safety, eval_eps = load_data(algo_name+"/"+file, "all")
        # plot rewards
        plot_rewards(train, axs[i], label="train")
        plot_rewards_sum(train, axs[i+2], label="train")
        plot_eval_mean(eval, eval_eps, axs[i+4])
        plot_eval_sum(eval, eval_eps, axs[i+4])
        plot_eval_safety(safety, eval_eps, axs[i+4])