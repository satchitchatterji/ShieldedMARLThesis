import pandas as pd
import pprint

df = pd.read_csv("project.csv")
# extract last 35 rows
df = df.iloc[-35:].reset_index(drop=True)
# make df from config column
config_df = pd.DataFrame.from_dict([eval(x) for x in df["config"].to_list()])
# print(config_df)

config_df["name"] = df["name"]

group_fields = ["algo", "eval_policy", "on_policy", "shield_alpha"]
# get unique values for each field
def name(a,e,o,s):
    return f"{a}_{e}_{o}_{s}"

runs = {}
# Iterate through each unique combination of field values
for a in config_df["algo"].unique():
    for e in config_df["eval_policy"].unique():
        for o in config_df["on_policy"].unique():
            for s in config_df["shield_alpha"].unique():
                # Filter the DataFrame for the current combination
                filtered_df = config_df[
                    (config_df["algo"] == a) &
                    (config_df["eval_policy"] == e) &
                    (config_df["on_policy"] == o) &
                    (config_df["shield_alpha"] == s)
                ]
                # Store the filtered DataFrame in the dictionary
                runs[name(a, e, o, s)] = filtered_df["name"].tolist()


pprint.pp(runs)