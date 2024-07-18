import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv('wandb_export_2024-07-01T16_53_49.427+02_00.csv'); topic = "mean_reward"
df = pd.read_csv('wandb_export_2024-07-01T16_52_55.611+02_00.csv'); topic = "eval_mean_safety"
# df = pd.read_csv('wandb_export_2024-07-01T16_53_24.971+02_00.csv'); topic = "eval_mean_reward"

cols = [x for x in df.columns if x.endswith(topic)]
df = df[cols]
firstunderscore = [x.find("_") for x in cols] 
firstspace = [x.find(" ") for x in cols] 

algos = ["IPPO", "SIPPO"]

# get default colour wheel for matplotlib
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# red, light blue, blue, dark blue
colors = [colors[1], "darkblue", "lightblue", "blue"]

ippo_cols = [cols[i] for i in range(len(cols)) if cols[i][:firstunderscore[i]] == "IPPO"][:5]
sippo_cols = [cols[i] for i in range(len(cols)) if cols[i][:firstunderscore[i]] == "SIPPO"]
print(ippo_cols)


ippo_df = df[ippo_cols]
sippo_df = df[[x for x in sippo_cols]]

# take mean and std of each algo
ippo_df["mean"] = ippo_df.mean(axis=1)
sippo_df["mean"] = sippo_df.mean(axis=1)

ippo_df["std"] = ippo_df.std(axis=1)
sippo_df["std"] = sippo_df.std(axis=1)

# plot
fig, ax = plt.subplots()
x = ippo_df.index if topic == "mean_reward" else ippo_df.index*10
ax.plot(x, ippo_df["mean"], label="IPPO", color=colors[0])
ax.fill_between(x, ippo_df["mean"] - ippo_df["std"], ippo_df["mean"] + ippo_df["std"], alpha=0.2, color=colors[0])

x = sippo_df.index if topic == "mean_reward" else sippo_df.index*10
ax.plot(x, sippo_df["mean"], label=r"SIPPO", color=colors[1])
ax.fill_between(x, sippo_df["mean"] - sippo_df["std"], sippo_df["mean"] + sippo_df["std"], alpha=0.2, color=colors[1])

ax.set_xlabel("Episode")
ax.legend()
ax.grid()
plt.tight_layout()

if topic == "mean_reward":
    ax.set_ylabel("Mean Reward")
    ax.set_title("Mean Reward per Episode (Training)")
    plt.savefig("training_pd.png", dpi=300, bbox_inches="tight")

if topic == "eval_mean_safety":
    ax.set_ylabel("Mean Action==Stag")
    ax.set_title("Mean Stag (Safety) per Episode")
    plt.savefig("safety_pd.png", dpi=300, bbox_inches="tight")

if topic == "eval_mean_reward":
    ax.set_ylabel("Mean Reward")
    ax.set_title("Mean Reward per Episode (Evaluation)")
    plt.savefig("eval_pd.png", dpi=300, bbox_inches="tight")


plt.show()
