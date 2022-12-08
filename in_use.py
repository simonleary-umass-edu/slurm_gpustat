import argparse
from tabulate import tabulate
from slurm_gpustat import parse_all_gpus, in_use

"""
this is copy pasted out of slurm_gpustat.py
this does just the **second half** of `slurm_gpustat.py --action=current`
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="slurm_gpus tool")
    parser.add_argument("-p", "--partition", default=None,
                        help=("the partition/queue (or multiple, comma separated) of"
                              " interest. By default set to all available partitions."))
    parser.add_argument("--verbose", action="store_true",
                        help="provide a more detailed breakdown of resources")
    args = parser.parse_args()

    resources = parse_all_gpus(partition=args.partition)
    in_use_table = in_use(resources, partition=args.partition,verbose=args.verbose)
    print(tabulate(in_use_table, showindex=False, headers="firstrow"))