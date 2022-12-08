import argparse
from slurm_gpustat import parse_all_gpus, in_use

def main():
    parser = argparse.ArgumentParser(description="slurm_gpus tool")
    parser.add_argument("-p", "--partition", default=None,
                        help=("the partition/queue (or multiple, comma separated) of"
                              " interest. By default set to all available partitions."))
    args = parser.parse_args()

    resources = parse_all_gpus(partition=args.partition)
    in_use(resources, partition=args.partition)


if __name__ == "__main__":
    main()
