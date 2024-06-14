import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import os

def load_data(file, which = "all"):
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

def plot_rewards(reward_hists, ax, label=""):
    # plot mean rewards per episode
    agent_rewards = [[np.mean(reward_hist[agent]) for reward_hist in reward_hists] for agent in reward_hists[0].keys()]
    for a, agent in enumerate(reward_hists[0].keys()):
        if label == "": 
            label=agent
        ax.plot(agent_rewards[a], label=label)

    if len(reward_hists[0].keys()) > 1:
        ax.plot([np.mean([np.mean(reward_hist[agent]) for agent in reward_hist.keys()]) for reward_hist in reward_hists], label="mean", color="black", linestyle="--")

    # ax.set_xlabel("Episode")
    # ax.set_ylabel("Mean Reward")
    # ax.set_title(f"{algo_name} on {env_name} (training)")

    # plt.legend()

    # plt.grid(True)
    # # plt.savefig(f"plots/{env_name}/{algo_name}/{cur_time}_train_mean.png")
    # plt.show()
    # plt.clf()
    return ax

############################################ PLOT SUM REWARDS ############################################

def plot_rewards_sum(reward_hists, ax, label=""):
    # plot sum of rewards per episode
    agent_rewards = [[np.sum(reward_hist[agent]) for reward_hist in reward_hists] for agent in reward_hists[0].keys()]
    for a, agent in enumerate(reward_hists[0].keys()):
        if label == "": 
            label=agent
        ax.plot(agent_rewards[a], label=label)


    if len(reward_hists[0].keys()) > 1:
        ax.plot([np.mean([np.sum(reward_hist[agent]) for agent in reward_hist.keys()]) for reward_hist in reward_hists], label="mean", color="black", linestyle="--")

    return ax
    # plt.xlabel("Episode")
    # plt.ylabel("Total Reward")
    # plt.title(f"{algo_name} on {env_name} (training)")

    # plt.legend()
    # plt.grid(True)
    # # plt.savefig(f"plots/{env_name}/{algo_name}/{cur_time}_train_total.png")
    # plt.show()
    # plt.clf()

############################################ PLOT EVAL MEAN REWARDS ############################################

def plot_eval_mean(eval_hists, eval_episodes, ax):
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
        ax.errorbar(eval_episodes, eval_means[agent], yerr=eval_stds[agent], label=f"{agent} mean", capsize=5, marker="x")

    if len(eval_hists[0][0].keys()) > 1:
        # compute overall mean and std
        eval_means["mean"] = []
        eval_stds["mean"] = []
        for eval_hist in eval_hists:
            eval_means["mean"].append(np.mean([np.mean([np.mean(eval_reward_hist[agent]) for agent in eval_reward_hist.keys()]) for eval_reward_hist in eval_hist]))
            eval_stds["mean"].append(np.std([np.mean([np.mean(eval_reward_hist[agent]) for agent in eval_reward_hist.keys()]) for eval_reward_hist in eval_hist]))
        ax.errorbar(eval_episodes, eval_means["mean"], yerr=eval_stds["mean"], label="mean", color="black", linestyle="--", capsize=5, marker="x")

    # plt.xticks(eval_episodes)
    # plt.xlabel("Episode")
    # plt.ylabel("Mean Reward")
    # plt.title(f"{algo_name} on {env_name} (evaluation)")
    # plt.grid(True)
    # plt.legend()

    # plt.show()
    return ax


############################################ PLOT EVAL SUM REWARDS ############################################

def plot_eval_sum(eval_hists, eval_episodes, ax):
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
        ax.errorbar(eval_episodes, eval_sums[agent], yerr=eval_stds[agent], label=f"{agent} total", capsize=5, marker="x")
    

    if len(eval_hists[0][0].keys()) > 1:        # compute overall mean and std
        eval_sums["mean"] = []
        eval_stds["mean"] = []
        for eval_hist in eval_hists:
            eval_sums["mean"].append(np.mean([np.mean([np.sum(eval_reward_hist[agent]) for agent in eval_reward_hist.keys()]) for eval_reward_hist in eval_hist]))
            eval_stds["mean"].append(np.std([np.mean([np.sum(eval_reward_hist[agent]) for agent in eval_reward_hist.keys()]) for eval_reward_hist in eval_hist]))

        ax.errorbar(eval_episodes, eval_sums["mean"], yerr=eval_stds["mean"], label="mean", color="black", linestyle="--", capsize=5, marker="x")

    # # plt.xticks(eval_episodes)
    # plt.xlabel("Episode")
    # plt.ylabel("Total Reward")
    # plt.title(f"{algo_name} on {env_name} (evaluation)")
    # plt.grid(True)
    # plt.legend()
    # # plt.savefig(f"plots/{env_name}/{algo_name}/{cur_time}_eval_total.png")
    # plt.show()
    return ax

############################################ PLOT EVAL MEAN SAFETY ############################################

def plot_eval_safety(eval_safeties, eval_episodes, ax):
   
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
        ax.errorbar(eval_episodes, eval_means[agent], yerr=eval_stds[agent], label=f"{agent} mean", capsize=5, marker="x")
    
    if len(eval_safeties[0][0].keys()) > 1:
        # compute overall mean and std
        eval_means["mean"] = []
        eval_stds["mean"] = []
        for eval_hist in eval_safeties:
            eval_means["mean"].append(np.mean([np.mean([np.mean(eval_reward_hist[agent]) for agent in eval_reward_hist.keys()]) for eval_reward_hist in eval_hist]))
            eval_stds["mean"].append(np.std([np.mean([np.mean(eval_reward_hist[agent]) for agent in eval_reward_hist.keys()]) for eval_reward_hist in eval_hist]))

        ax.errorbar(eval_episodes, eval_means["mean"], yerr=eval_stds["mean"], label="mean", color="black", linestyle="--", capsize=5, marker="x")

    # plt.xticks(eval_episodes)
    # plt.xlabel("Episode")
    # plt.ylabel("Mean Safety")
    # plt.title(f"{algo_name} on {env_name} (safety evaluations)")
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    return ax


if __name__ == "__main__":
    base = "../../../histories/CartSafe-v0"
    base_configs = "../../../configs/CartSafe-v0"
    files = {"IPPO": "2024-06-14_195159", "SIPPO": "2024-06-11_190327"}