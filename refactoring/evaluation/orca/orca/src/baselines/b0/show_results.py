from codecarbon import EmissionsTracker
import psutil
import csv
import os
import json
from pathlib import Path
from accuracy import calculate_exact_match, calculate_statement_coverage, calculate_prefix_match, calculate_symbol_table_accuracy

# Start CodeCarbon tracker
tracker = EmissionsTracker(project_name=r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\orca\src\baselines\b0\show_results.py")
tracker.start()

mem_start = psutil.virtual_memory().used / (1024**2)
cpu_start = psutil.cpu_percent(interval=None)

script_directory = Path(__file__).resolve()
base_directory = script_directory.parents[3]

def calculate_metrics(predictions, dataset, complete):
    """Calculates accuracy metrics based on predictions and ground truth."""
    em = 0
    cov_r = 0
    cov_p = 0
    pre_r = 0
    pre_p = 0
    st = 0
    buggy_count = 0
    non_buggy_count = 0

    for id in predictions:
        try:
            exception_info = dataset[id]['exception_info']
            if exception_info:
                buggy_count += 1
            else:
                non_buggy_count += 1

            pred_symbol_table = predictions[id]["symbol_table"]
            pred_exe = predictions[id]['execution_order']
            gt_exe_symbol_table = dataset[id]['final_trace']
            gt_exe = dataset[id]['ground_truth_execution_order']

            em += calculate_exact_match(pred_exe, gt_exe)
            statement_recall, statement_precision = calculate_statement_coverage(pred_exe, gt_exe)
            cov_r += statement_recall
            cov_p += statement_precision
            prefix_recall, prefix_precision = calculate_prefix_match(pred_exe, gt_exe)
            pre_r += prefix_recall
            pre_p += prefix_precision

            if complete:
                st += calculate_symbol_table_accuracy(pred_symbol_table, gt_exe_symbol_table)
        except:
            continue

    total_instances = buggy_count + non_buggy_count

    results = {
        "Total Instances": len(predictions),
        "Buggy Instances": buggy_count,
        "Non-Buggy Instances": non_buggy_count,
    }

    if total_instances > 0:
        results["Exact Match"] = 100 * (em / total_instances)
        results["Prefix Match Recall"] = 100 * (pre_r / total_instances)
        results["Prefix Match Precision"] = 100 * (pre_p / total_instances)
        results["Statement Coverage Recall"] = 100 * (cov_r / total_instances)
        results["Statement Coverage Precision"] = 100 * (cov_p / total_instances)
        if complete:
            results["Symbol Table Accuracy"] = 100 * (st / total_instances)
    else:
        results["Exact Match"] = 0.0
        results["Prefix Match Recall"] = 0.0
        results["Prefix Match Precision"] = 0.0
        results["Statement Coverage Recall"] = 0.0
        results["Statement Coverage Precision"] = 0.0
        if complete:
            results["Symbol Table Accuracy"] = 0.0


    return results

def process_data(complete, dataset, predictions):
    """Processes data to calculate and return accuracy metrics."""
    return calculate_metrics(predictions, dataset, complete)

def load_dataset(dataset_path):
    """Loads a JSON dataset from the specified file path."""
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    return dataset

def print_results(results, title):
    """Prints the results in a formatted manner."""
    print(f"\n===================== {title} =====================")
    for key, value in results.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    print()

if __name__ == "__main__":
    complete_code_dataset_dir = base_directory / 'dataset' / 'baseline' / 'fixeval_cfg_b0.json'
    complete_code_pred_dir = base_directory / 'output' / 'baseline' / 'b0' / 'codeExe_fixeval.json'
    incomplete_code_dataset_dir = base_directory / 'dataset' / 'baseline' / 'fixeval_incom_cfg_b0.json'
    incomplete_code_pred_dir = base_directory / 'output' / 'baseline' / 'b0' / 'codeExe_incom_fixeval.json'

    complete_dataset = load_dataset(complete_code_dataset_dir)
    pred_complete_codeExe = load_dataset(complete_code_pred_dir)
    incomplete_dataset = load_dataset(incomplete_code_dataset_dir)
    pred_incom_codeExe = load_dataset(incomplete_code_pred_dir)

    results_complete = process_data(True, complete_dataset, pred_complete_codeExe)
    results_incomplete = process_data(False, incomplete_dataset, pred_incom_codeExe)

    print_results(results_complete, "RQ3 & RQ4: Complete Code Results")
    print_results(results_incomplete, "RQ3: Incomplete Code Results")

mem_end = psutil.virtual_memory().used / (1024**2)
cpu_end = psutil.cpu_percent(interval=None)

csv_file = "psutil_data.csv"
file_exists = os.path.isfile(csv_file)
with open(csv_file, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(["file", "mem_start_MB", "mem_end_MB", "mem_diff_MB", "cpu_start_percent", "cpu_end_percent"])
    writer.writerow([
        __file__,
        f"{mem_start:.2f}",
        f"{mem_end:.2f}",
        f"{mem_end - mem_start:.2f}",
        f"{cpu_start:.2f}",
        f"{cpu_end:.2f}"
    ])

tracker.stop()
