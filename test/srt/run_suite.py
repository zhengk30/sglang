import argparse
import glob
from dataclasses import dataclass

from sglang.test.test_utils import run_unittest_files


@dataclass
class TestFile:
    name: str
    estimated_time: float = 60


suites = {
    "per-commit": [
    ],
    "per-commit-2-gpu": [
        TestFile("test_disaggregation.py", 499),
        TestFile("test_dp_attention.py", 277),
    ],
    "per-commit-4-gpu": [
    ],
    "per-commit-8-gpu": [
    ],
    "per-commit-4-gpu-b200": [
        # TestFile("test_gpt_oss_4gpu.py", 600),
        # TestFile("test_deepseek_v3_fp4_4gpu.py", 3600),
    ],
    "per-commit-4-gpu-deepep": [
    ],
    "per-commit-8-gpu-deepep": [
    ],
    "per-commit-8-gpu-h20": [
    ],
    "vllm_dependency_test": [
    ],
}

# Add AMD tests
suite_amd = {
    "per-commit-amd": [
    ],
    "per-commit-amd-mi35x": [
    ],
    "per-commit-2-gpu-amd": [
    ],
    "per-commit-4-gpu-amd": [
    ],
    "per-commit-8-gpu-amd": [
    ],
    "nightly-amd": [
    ],
}

# Add Intel Xeon tests
suite_xeon = {
    "per-commit-cpu": [
    ],
}

# Add Ascend NPU tests
suite_ascend = {
    "per-commit-1-ascend-npu": [
    ],
    "per-commit-2-ascend-npu": [
    ],
    "per-commit-4-ascend-npu": [
    ],
    "per-commit-16-ascend-a3": [
    ],
}

suites.update(suite_amd)
suites.update(suite_xeon)
suites.update(suite_ascend)


def auto_partition(files, rank, size):
    """
    Partition files into size sublists with approximately equal sums of estimated times
    using stable sorting, and return the partition for the specified rank.

    Args:
        files (list): List of file objects with estimated_time attribute
        rank (int): Index of the partition to return (0 to size-1)
        size (int): Number of partitions

    Returns:
        list: List of file objects in the specified rank's partition
    """
    weights = [f.estimated_time for f in files]

    if not weights or size <= 0 or size > len(weights):
        return []

    # Create list of (weight, original_index) tuples
    # Using negative index as secondary key to maintain original order for equal weights
    indexed_weights = [(w, -i) for i, w in enumerate(weights)]
    # Stable sort in descending order by weight
    # If weights are equal, larger (negative) index comes first (i.e., earlier original position)
    indexed_weights = sorted(indexed_weights, reverse=True)

    # Extract original indices (negate back to positive)
    indexed_weights = [(w, -i) for w, i in indexed_weights]

    # Initialize partitions and their sums
    partitions = [[] for _ in range(size)]
    sums = [0.0] * size

    # Greedy approach: assign each weight to partition with smallest current sum
    for weight, idx in indexed_weights:
        # Find partition with minimum sum
        min_sum_idx = sums.index(min(sums))
        partitions[min_sum_idx].append(idx)
        sums[min_sum_idx] += weight

    # Return the files corresponding to the indices in the specified rank's partition
    indices = partitions[rank]
    return [files[i] for i in indices]


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--timeout-per-file",
        type=int,
        default=1200,
        help="The time limit for running one file in seconds.",
    )
    arg_parser.add_argument(
        "--suite",
        type=str,
        default=list(suites.keys())[0],
        choices=list(suites.keys()) + ["all"],
        help="The suite to run",
    )
    arg_parser.add_argument(
        "--range-begin",
        type=int,
        default=0,
        help="The begin index of the range of the files to run.",
    )
    arg_parser.add_argument(
        "--range-end",
        type=int,
        default=None,
        help="The end index of the range of the files to run.",
    )
    arg_parser.add_argument(
        "--auto-partition-id",
        type=int,
        help="Use auto load balancing. The part id.",
    )
    arg_parser.add_argument(
        "--auto-partition-size",
        type=int,
        help="Use auto load balancing. The number of parts.",
    )
    args = arg_parser.parse_args()
    print(f"{args=}")

    if args.suite == "all":
        files = glob.glob("**/test_*.py", recursive=True)
    else:
        files = suites[args.suite]

    if args.auto_partition_size:
        files = auto_partition(files, args.auto_partition_id, args.auto_partition_size)
    else:
        files = files[args.range_begin : args.range_end]

    print("The running tests are ", [f.name for f in files])

    exit_code = run_unittest_files(files, args.timeout_per_file)
    exit(exit_code)
