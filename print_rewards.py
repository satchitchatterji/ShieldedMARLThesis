import matplotlib.pyplot as plt
import pandas as pd

def plot_rewards(history):
    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set(xlabel='Episode', ylabel='Reward', title='Reward per episode')
    ax.grid()
    return fig, ax

def plot_cumulative_mean_rewards(history):
    fig, ax = plt.subplots()
    ax.plot(pd.Series(history).expanding().mean())
    ax.set(xlabel='Episode', ylabel='Cumulative Mean Reward', title='Cumulative Mean Reward per episode')
    ax.grid()
    return fig, ax

def plot_all(history):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(history)
    ax[0].set(xlabel='Episode', ylabel='Reward', title='Reward per episode')
    ax[0].grid()
    ax[1].plot(pd.Series(history).expanding().mean())
    ax[1].set(xlabel='Episode', ylabel='Cumulative Mean Reward', title='Cumulative Mean Reward per episode')
    ax[1].grid()
    plt.show()

def plot_all_multi(history, labels):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for i in range(len(history)):
        ax[0].plot(history[i], label=labels[i])
        ax[1].plot(pd.Series(history[i]).expanding().mean(), label=labels[i])
    ax[0].set(xlabel='Episode', ylabel='Reward', title='Reward per episode')
    ax[0].grid()
    ax[0].legend()
    ax[1].set(xlabel='Episode', ylabel='Cumulative Mean Reward', title='Cumulative Mean Reward per episode')
    ax[1].grid()
    ax[1].legend()
    plt.show()

def plot_all_multi_with_mean(history, labels):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for i in range(len(history)):
        ax[0].plot(history[i], label=labels[i])
        ax[1].plot(pd.Series(history[i]).expanding().mean(), label=labels[i])
    ax[0].set(xlabel='Episode', ylabel='Reward', title='Reward per episode')
    ax[0].grid()
    ax[0].legend()
    ax[1].set(xlabel='Episode', ylabel='Cumulative Mean Reward', title='Cumulative Mean Reward per episode')
    ax[1].grid()
    ax[1].legend()
    plt.show()