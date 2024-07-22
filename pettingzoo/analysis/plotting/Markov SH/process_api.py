import pandas as pd
import pprint


runs_too_short = ["SIPPO_2024-07-05_165559", "SIPPO_2024-07-05_165554", "SIPPO_2024-07-05_165551", 'SIPPO_2024-07-04_194104']
killed_runs = ["IPPO_2024-07-05_171759", "SIPPO_2024-07-04_210601", "SIPPO_2024-07-04_210658"]

exclude_runs = runs_too_short + killed_runs

# get unique values for each field
def name(r,a,v):
    return f"{r}_{a}_{v}"

def get_non_empty_groups():

    df = pd.read_csv("project.csv")
    # make df from config column
    config_df = pd.DataFrame.from_dict([eval(x) for x in df["config"].to_list()])
    # print(config_df)

    config_df["name"] = df["name"]

    group_fields = ["shielded_ratio", "algo", "shield_version"]

    runs = {}
    # Iterate through each unique combination of field values
    for r in config_df["shielded_ratio"].unique():
        for a in config_df["algo"].unique():
            for v in config_df["shield_version"].unique():
                filtered_df = config_df[
                    (config_df["shielded_ratio"] == r) &
                    (config_df["algo"] == a) &
                    (config_df["shield_version"] == v)
                ]
                filtered_df = filtered_df[~filtered_df["name"].isin(exclude_runs)]
                # Store the filtered DataFrame in the dictionary
                runs[name(r,a,v)] = filtered_df["name"].tolist()

    # delete empty groups
    runs = {k:v for k,v in runs.items() if v}
    pprint.pp(runs)
    return runs

if __name__ == "__main__":
    get_non_empty_groups()