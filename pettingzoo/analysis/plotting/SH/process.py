import pandas as pd
import matplotlib.pyplot as plt


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

df = pd.read_csv('wandb_export_2024-07-19T21_58_03.895+02_00.csv'); topic = "mean_reward"
# df = pd.read_csv('wandb_export_2024-07-19T21_57_52.672+02_00.csv'); topic = "eval_mean_safety"
# df = pd.read_csv('wandb_export_2024-07-19T21_57_38.021+02_00.csv'); topic = "eval_mean_reward"

percent_rolling = 0.1

cols = [x for x in df.columns if x.endswith(topic)]
df = df[cols]
firstunderscore = [x.find("_") for x in cols] 
firstspace = [x.find(" ") for x in cols] 

algos = ["IPPO", "SIPPO"]

# get default colour wheel for matplotlib
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# red, light blue, blue, dark blue
# colors = [colors[1], "darkblue", "lightblue", "blue"]

# colors = ["darkred", "darkcyan", "purple"]
# print(colors)
# ippo_cols = [cols[i] for i in range(len(cols)) if cols[i][:firstunderscore[i]] == "IPPO"]
# sippo_cols = [cols[i] for i in range(len(cols)) if cols[i][:firstunderscore[i]] == "SIPPO"]

ippo_runs = ["IPPO_2024-07-01_164539", "IPPO_2024-07-01_164532", "IPPO_2024-07-01_164520", "IPPO_2024-07-01_164514", "IPPO_2024-07-01_164509"]
sippo_runs_v1 = ["SIPPO_2024-07-01_200634", "SIPPO_2024-07-01_200623", "SIPPO_2024-07-01_200617", "SIPPO_2024-07-01_200610", "SIPPO_2024-07-01_195959"]
sippo_runs_v0 = ["SIPPO_2024-07-01_164115", "SIPPO_2024-07-01_164110", "SIPPO_2024-07-01_163655", "SIPPO_2024-07-01_163649", "SIPPO_2024-07-01_163643"]


ippo_df = df[[x+" - "+topic for x in ippo_runs]]
sippo_df_v0 = df[[x+" - "+topic for x in sippo_runs_v0]]
sippo_df_v1 = df[[x+" - "+topic for x in sippo_runs_v1]]

# take mean and std of each algo
ippo_df["mean"] = ippo_df.mean(axis=1)
sippo_df_v0["mean"] = sippo_df_v0.mean(axis=1)
sippo_df_v1["mean"] = sippo_df_v1.mean(axis=1)

ippo_df["std"] = ippo_df.std(axis=1)
sippo_df_v0["std"] = sippo_df_v0.std(axis=1)
sippo_df_v1["std"] = sippo_df_v1.std(axis=1)

print(len(ippo_df))
window = int(len(ippo_df)*percent_rolling)
print("Rolling Window:", window)

# plot
fig, ax = plt.subplots()
x = ippo_df.rolling(window).mean().index if topic == "mean_reward" else ippo_df.rolling(window).mean().index*10
ax.plot(x, ippo_df["mean"].rolling(window).mean(), label="IPPO", color=colors[0])
ax.fill_between(x, (ippo_df["mean"] - ippo_df["std"]).rolling(window).mean(), (ippo_df["mean"] + ippo_df["std"]).rolling(window).mean(), alpha=0.2, color=colors[0])

x = sippo_df_v0.rolling(window).mean().index if topic == "mean_reward" else sippo_df_v0.rolling(window).mean().index*10
ax.plot(x, sippo_df_v0["mean"].rolling(window).mean(), label=r"SIPPO (pure strategy)", color=colors[1])
ax.fill_between(x, (sippo_df_v0["mean"] - sippo_df_v0["std"]).rolling(window).mean(), (sippo_df_v0["mean"] + sippo_df_v0["std"]).rolling(window).mean(), alpha=0.2, color=colors[1])

x = sippo_df_v1.rolling(window).mean().index if topic == "mean_reward" else sippo_df_v1.rolling(window).mean().index*10
ax.plot(x, sippo_df_v1["mean"].rolling(window).mean(), label=r"SIPPO (mixed strategy)", color=colors[2])
ax.fill_between(x, (sippo_df_v1["mean"] - sippo_df_v1["std"]).rolling(window).mean(), (sippo_df_v1["mean"] + sippo_df_v1["std"]).rolling(window).mean(), alpha=0.2, color=colors[2])

ax.set_xlabel("Episode")

ax.grid()
plt.tight_layout()

if topic == "mean_reward":
    ax.set_ylabel("Mean Reward")
    ax.set_title("Mean Reward per Episode Step (Training)")
    plt.legend()
    plt.savefig("training_sh.png", dpi=300, bbox_inches="tight")

if topic == "eval_mean_safety":
    ax.set_ylabel("Mean Action==Stag")
    ax.set_title("Mean Stag (Safety) per Episode")
    plt.savefig("safety_sh.png", dpi=300, bbox_inches="tight")

if topic == "eval_mean_reward":
    ax.set_ylabel("Mean Reward")
    ax.set_title("Mean Reward per Episode Step (Evaluation)")
    plt.savefig("eval_sh.png", dpi=300, bbox_inches="tight")


plt.show()
