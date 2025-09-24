import matplotlib.pyplot as plt

import os

import sys

import numpy as np


def parse_blocky_csv(path):
    """
    Parses a file where each block looks like:
      "Method, p, q, bs, T"
      step0,step1,step2,...
      metric_0_line_for_method
      metric_1_line_for_method
      ...
    Blocks repeat. Number of metric lines per block can vary.
    Returns: 
      data: dict: {
        method_label: {
            "steps": [floats],
            "metrics": [[floats], [floats], ...]   # list of metric series
        }, ...
      }
      grouped: dict: {
        method_label: [
            {"steps": [...], "metrics": [...]},
            {"steps": [...], "metrics": [...]},
            ...
        ]
      }
    """
    def parse_nums(line):
        return [float(x.strip()) for x in line.split(",") if x.strip()]

    data = {}
    grouped = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    i = 0
    while i < len(lines):
        # header
        if lines[i].startswith('"') and lines[i].endswith('"'):
            label = lines[i].strip('"')
            i += 1
            # steps
            steps = parse_nums(lines[i]); i += 1
            # metrics until next header or EOF
            metrics = []
            while i < len(lines) and not (lines[i].startswith('"') and lines[i].endswith('"')):
                metrics.append(parse_nums(lines[i]))
                i += 1
            data[label] = {"steps": steps, "metrics": metrics}
            if label not in grouped:
                grouped[label] = []
            grouped[label].append({"steps": steps, "metrics": metrics})
        else:
            # skip unexpected lines safely
            i += 1

    for key, method in grouped.items():
        # aggregate by averaging metrics across runs
        if len(method) == 1:
            continue
        num_runs = len(method)
        metric_lists = []
        for run in method:
            metric_lists.append(run["metrics"])
        # Now metric_lists is a list of lists of metrics
        means, std = calculate_mean_and_std_across_lists(metric_lists)
        if key == 'Scaffnew, 0.5, 0.0, 128, 4000' or key == 'Scaffnew, 0.5, 0.0, 128, 2000' :
            steps = list(range(11, 11 * len(means[0]) + 1, 11))
        else:
            steps = list(range(10, 10 * len(means[0]) + 1, 10))
        method.append({"steps": steps, "means": means})
        method.append({"steps": steps, "std": std})

    return data, grouped

def calculate_mean_and_std_across_lists(lists):
    """
    Calculate the mean and standard deviation for corresponding sublists across lists at the same index.
    """
    num_sublists = len(lists[0])  # Assume all lists have the same number of sublists
    means = []
    stds = []

    for sublist_index in range(num_sublists):
        # Collect sublists at the same index from all lists
        sublists = [lst[sublist_index] for lst in lists]

        # Find the maximum length of the sublists
        min_length = min(len(sublist) for sublist in sublists)

        # Pad sublists to the same length with None
        padded_sublists = [
            sublist + [None] * (min_length - len(sublist)) for sublist in sublists
        ]

        # Compute mean and std for each position across sublists
        sublist_means = []
        sublist_stds = []
        for values in zip(*padded_sublists):
            valid_values = [v for v in values if v is not None]
            if valid_values:
                sublist_means.append(np.mean(valid_values))
                sublist_stds.append(np.std(valid_values))
            else:
                sublist_means.append(None)
                sublist_stds.append(None)

        means.append(sublist_means)
        stds.append(sublist_stds)

    return means, stds

def plot_train_acc_across_methods(grouped, metric_index=0, title=None, xlabel="Step", ylabel="Metric", save_as=None):
    """
    Plots the selected metric_index for all methods on one chart.
    metric_index=0 means the first metric in each block.
    """
    plt.figure()
    colors = ['black', 'red', 'orange', 'blue']
    for i, (label, methods) in enumerate(grouped.items()):
        for method in methods:
            # if "steps" in method and "metrics" in method:
            #     steps = method["steps"]
            #     metrics = method["metrics"]
            #     if metric_index < len(metrics):
            #         y = metrics[metric_index]
            #         plt.plot(steps, y, label=label.split(',')[0], color=colors[i % len(colors)])
            if "steps" in method and "means" in method:
                steps = method["steps"]
                means = method["means"][metric_index]
            if "steps" in method and "std" in method:
                stds = method["std"][metric_index]
                plt.plot(steps, means, label=','.join(label.split(',')[:1]), color=colors[i % len(colors)])
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    # if title:
    #     plt.title(title)
    plt.legend(fontsize=16)
    plt.tight_layout()
    if save_as:
        os.makedirs(os.path.dirname(save_as), exist_ok=True)
        plt.savefig(save_as)
    plt.show()


def plot_loss_across_methods(grouped, metric_index=2, title="Loss Over Rounds", xlabel="Communication Rounds", ylabel="Loss", save_as=None):
    """
    Plots the loss for all methods on one chart in a separate figure.
    Assumes loss is the third metric (metric_index=2).
    """
    plt.figure()
    colors = ['black', 'red', 'orange', 'blue']
    for i, (label, methods) in enumerate(grouped.items()):
        for method in methods:
            if "steps" in method and "means" in method:
                steps = method["steps"]
                means = method["means"][metric_index]
            if "steps" in method and "std" in method:
                stds = method["std"][metric_index]
                plt.plot(steps, means, label=','.join(label.split(',')[:1]), color=colors[i % len(colors)])
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    # if title:
    #     plt.title(title)
    plt.legend(fontsize=16)
    plt.tight_layout()
    if save_as:
        os.makedirs(os.path.dirname(save_as), exist_ok=True)
        plt.savefig(save_as)
    plt.show()


def plot_test_acc_across_methods(grouped, metric_indx=1,title="Test Accuracy Over Rounds", xlabel="Communication Rounds", ylabel="Test Accuracy", save_as=None):
    """
    Plots the test accuracy for all methods on one chart in a separate figure.
    Assumes test accuracy is the second metric (metric_index=1).
    """
    plt.figure()
    colors = ['black', 'red', 'orange', 'blue']
    for i, (label, methods) in enumerate(grouped.items()):
        for method in methods:
            if "steps" in method and "means" in method:
                steps = method["steps"]
                means = method["means"][metric_indx]
            if "steps" in method and "std" in method:
                stds = method["std"][metric_indx]
                plt.plot(steps, means, label=label.split(',')[0], color=colors[i % len(colors)])
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    # if title:
    #     plt.title(title)
    plt.legend(fontsize=16)
    plt.tight_layout()
    if save_as:
        os.makedirs(os.path.dirname(save_as), exist_ok=True)
        plt.savefig(save_as)
    plt.show()


if __name__ == "__main__":
    # 1) Point to your file (save your pasted content as e.g. results.csv)
    # absolute path of the current script
    current_file = os.path.abspath(__file__)

    # directory that contains the current script
    current_dir = os.path.dirname(current_file)

    csv_filename = "cifar10_e2000_homFalse_0_L_2_dir_10.csv"
    csv_dirname = os.path.splitext(csv_filename)[0]
    path = os.path.join(current_dir, "results", csv_dirname, csv_filename)

    data = parse_blocky_csv(path)

    # 2) Choose which metric line to plot:
    #    - 0: first metric after steps (e.g., accuracy@1 or train acc)

    # Define the subdirectory to save plots and CSV using the CSV file name (without extension)
    csv_name = os.path.splitext(os.path.basename(path))[0]
    save_dir = os.path.join(current_dir, "results", csv_name)
    os.makedirs(save_dir, exist_ok=True)

    # Extract dataset and dir value from CSV filename for plot naming
    # Example: cifar100_e3000_homFalse_0_L_2_dir_1.csv -> dataset=cifar100, dir_val=1
    filename_parts = csv_name.split('_')
    dataset = filename_parts[0]  # e.g., cifar100
    dir_val = None
    for i, part in enumerate(filename_parts):
        if part == 'dir' and i + 1 < len(filename_parts):
            dir_val = filename_parts[i + 1]
            break
    
    # Create descriptive filename suffix
    if dir_val:
        filename_suffix = f"{dataset}_dir{dir_val}"
    else:
        filename_suffix = dataset

    # Move the CSV file to the save directory
    csv_destination = os.path.join(save_dir, os.path.basename(path))
    os.rename(path, csv_destination)

    # Parse the CSV file
    _, data = parse_blocky_csv(csv_destination)

    # Plot metric 0 (e.g., accuracy@1 or train acc) in a separate figure
    plot_train_acc_across_methods(
        data,
        metric_index=0,  # First metric
        title="Train Accuracy Over Rounds",
        xlabel="Communication Rounds",
        ylabel="Train Accuracy",
        save_as=os.path.join(save_dir, f"fedrcu_train_accuracy_comparison_{filename_suffix}.eps")
    )

    # Plot loss in a separate figure
    plot_loss_across_methods(
        data,
        title="Loss Over Rounds",
        xlabel="Communication Rounds",
        ylabel="Loss",
        save_as=os.path.join(save_dir, f"fedrcu_loss_comparison_{filename_suffix}.eps")
    )

    # Plot test accuracy in a separate figure
    plot_test_acc_across_methods(
        data,
        title="Test Accuracy Over Rounds",
        xlabel="Communication Rounds",
        ylabel="Test Accuracy",
        save_as=os.path.join(save_dir, f"fedrcu_test_accuracy_comparison_{filename_suffix}.eps")
    )
