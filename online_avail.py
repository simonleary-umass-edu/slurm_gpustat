import argparse
import pandas as pd
from tabulate import tabulate
from slurm_gpustat import parse_all_gpus, node_states, available, summary

"""
this is copy pasted out of slurm_gpustat.py
this does just the **first half** of `slurm_gpustat.py --action=current`
minus the "all GPU's" column
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="slurm_gpus tool")
    parser.add_argument("-p", "--partition", default=None,
                        help=("the partition/queue (or multiple, comma separated) of"
                              " interest. By default set to all available partitions."))
    args = parser.parse_args()

    resources = parse_all_gpus(partition=args.partition)
    states = node_states(partition=args.partition)

    online_table = summary(mode="online", resources=resources, states=states)
    avail_table = available(resources=resources, states=states, verbose=False)

    online_df = pd.DataFrame(online_table, columns=["GPU model", "online"])
    online_df = online_df.set_index(["GPU model"])

    avail_df = pd.DataFrame(avail_table, columns=["GPU model", "available", "notes"])
    avail_df = avail_df.set_index(["GPU model"])

    big_df = pd.DataFrame()
    for df in [online_df, avail_df]:
        big_df = big_df.merge(df, how='outer', left_index=True, right_index=True)
    big_df = big_df.sort_values(by="online", ascending=False)
    print(tabulate(big_df, headers=(["GPU model", "online", "available", "notes"])))
