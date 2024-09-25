import pandas as pd
import matplotlib.pyplot as plt
percent_rolling = 0.1

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# df = pd.read_csv('wandb_export_2024-06-27T15_09_01.463+02_00.csv'); topic = "total_reward_mean"
df = pd.read_csv('wandb_export_2024-06-27T15_09_07.687+02_00.csv'); topic = "eval_total_reward_mean"
# df = pd.read_csv('wandb_export_2024-06-27T15_09_14.977+02_00.csv'); topic = "eval_mean_safety"
cols = [x for x in df.columns if x.endswith(topic)]
df = df[cols]
firstunderscore = [x.find("_") for x in cols] 

algos = ["IPPO", "SIPPO"]

# get default colour wheel for matplotlib
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# red, light blue, blue, dark blue
colors = [colors[1], "darkblue", "lightblue", "blue"]

ippo_cols = [cols[i] for i in range(len(cols)) if cols[i][:firstunderscore[i]] == "IPPO"]
# sippo_cols = [cols[i] for i in range(len(cols)) if cols[i][:firstunderscore[i]] == "SIPPO"]
print(ippo_cols)

sippo_2_cols = ["SIPPO_2024-06-26_235639", "SIPPO_2024-06-26_235628", "SIPPO_2024-06-26_235618", "SIPPO_2024-06-26_235604"]
sippo_05_cols = ["SIPPO_2024-06-26_235333", "SIPPO_2024-06-26_235317", "SIPPO_2024-06-26_235237", "SIPPO_2024-06-26_235229", "SIPPO_2024-06-26_235230"]
sippo_1_cols = ["SIPPO_2024-06-26_224116", "SIPPO_2024-06-26_222807", "SIPPO_2024-06-26_222743", "SIPPO_2024-06-26_222740", "SIPPO_2024-06-26_222648"]

window = int(len(df)*percent_rolling)

ippo_df = df[ippo_cols]
sippo_05_df = df[[x+f" - {topic}" for x in sippo_05_cols]]
sippo_1_df = df[[x+f" - {topic}" for x in sippo_1_cols]]
sippo_2_df = df[[x+f" - {topic}" for x in sippo_2_cols]]

# take mean and std of each algo
ippo_df["mean"] = ippo_df.mean(axis=1).rolling(window=window).mean()
sippo_05_df["mean"] = sippo_05_df.mean(axis=1).rolling(window=window).mean()
sippo_1_df["mean"] = sippo_1_df.mean(axis=1).rolling(window=window).mean()
sippo_2_df["mean"] = sippo_2_df.mean(axis=1).rolling(window=window).mean()

ippo_df["std"] = ippo_df.std(axis=1).rolling(window=window).mean()
sippo_05_df["std"] = sippo_05_df.std(axis=1).rolling(window=window).mean()
sippo_1_df["std"] = sippo_1_df.std(axis=1).rolling(window=window).mean()
sippo_2_df["std"] = sippo_2_df.std(axis=1).rolling(window=window).mean()

# plot
fig, ax = plt.subplots()
ax.plot(ippo_df["mean"], label="PPO", color=colors[0])
ax.fill_between(ippo_df.index, ippo_df["mean"] - ippo_df["std"], ippo_df["mean"] + ippo_df["std"], alpha=0.2, color=colors[0])

ax.plot(sippo_05_df["mean"], label=r"PLPG ($\alpha=0.5$)", color=colors[1], linestyle="dotted")
ax.fill_between(sippo_05_df.index, sippo_05_df["mean"] - sippo_05_df["std"], sippo_05_df["mean"] + sippo_05_df["std"], alpha=0.2, color=colors[1])

ax.plot(sippo_1_df["mean"], label=r"PLPG ($\alpha=1.0$)", color=colors[2], linestyle="dashed")
ax.fill_between(sippo_1_df.index, sippo_1_df["mean"] - sippo_1_df["std"], sippo_1_df["mean"] + sippo_1_df["std"], alpha=0.2, color=colors[2])

ax.plot(sippo_2_df["mean"], label=r"PLPG ($\alpha=2.0$)", color=colors[3], linestyle="solid")
ax.fill_between(sippo_2_df.index, sippo_2_df["mean"] - sippo_2_df["std"], sippo_2_df["mean"] + sippo_2_df["std"], alpha=0.2, color=colors[3])

ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
ax.set_title("Mean Reward per Episode (Evaluation)")
ax.grid()

# ax.legend()
plt.tight_layout()
plt.savefig("eval_CartSafe-v0_ppo.png", dpi=300, bbox_inches="tight")
# plt.show()
