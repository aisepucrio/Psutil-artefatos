from codecarbon import EmissionsTracker
import psutil
import csv
import os
import json
from pathlib import Path

# Start CodeCarbon tracker
tracker = EmissionsTracker(project_name=r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\orca\src\baselines\b2\show_results.py")
tracker.start()

mem_start = psutil.virtual_memory().used / (1024**2)
cpu_start = psutil.cpu_percent(interval=None)

script_directory = Path(__file__).resolve()
base_directory = script_directory.parents[3]

def calculate_rates(buggy_count, non_buggy_count, tp, fp, fn, tn):
    """Calculates and returns various rates."""
    return (
        100 * (tp / buggy_count) if buggy_count else 0,
        100 * (fp / non_buggy_count) if non_buggy_count else 0,
        100 * (fn / buggy_count) if buggy_count else 0,
        100 * (tn / non_buggy_count) if non_buggy_count else 0,
    )

def print_error_detection_results(buggy_count, non_buggy_count, tp, fp, fn, tn):
    """Prints error detection results."""
    print("\nTable 2 - Error Detection")
    print(f"True Positive Instances: {tp}")
    print(f"False Positive Instances: {fp}")
    print(f"False Negative Instances: {fn}")
    print(f"True Negative Instances: {tn}")

    tp_rate, fp_rate, fn_rate, tn_rate = calculate_rates(buggy_count, non_buggy_count, tp, fp, fn, tn)
    print(f"\nTrue Positive Rate: {tp_rate:.2f}")
    print(f"False Positive Rate: {fp_rate:.2f}")
    print(f"False Negative Rate: {fn_rate:.2f}")
    print(f"True Negative Rate: {tn_rate:.2f}")

    accuracy = 100 * ((tp + tn) / (buggy_count + non_buggy_count)) if (buggy_count + non_buggy_count) else 0
    print(f"\nAccuracy: {accuracy:.2f}")

def print_accuracy_metrics(total, buggy_count, non_buggy_count, ErrorLocation, EM, PRE_R, PRE_P, COV_R, COV_P):
    """Prints the accuracy metrics."""
    print(f"Total Instances: {total}")
    print(f"Buggy Instances: {buggy_count}")
    print(f"Non-Buggy Instances: {non_buggy_count}")
    print("\nTable 1 - Error Localization")
    print(f"Error Localization: {100 * (ErrorLocation / buggy_count):.2f}" if buggy_count else "Error Localization: 0.00")
    print_error_detection_results(buggy_count, non_buggy_count, tp, fp, fn, tn)
    print("\n============== RQ3 ==============")
    print(f"Exact Match: {100 * (EM / total):.2f}")
    print(f"\nPrefix Recall: {100 * (PRE_R / total):.2f}")
    print(f"Prefix Precision: {100 * (PRE_P / total):.2f}")
    print(f"\nStatement Cov. Recall: {100 * (COV_R / total):.2f}")
    print(f"Statement Cov. Precision: {100 * (COV_P / total):.2f}")

def extract_accuracy_data(dataset, response_cache):
    """Extracts and calculates accuracy metrics."""
    ErrorLocation = 0
    tp, fp, fn, tn = 0, 0, 0, 0
    EM, PRE_R, PRE_P, COV_R, COV_P = 0, 0, 0, 0, 0
    buggy_count, non_buggy_count = 0, 0

    for probID in response_cache:
        for subID in response_cache[probID]:
            ground_truth_exception_info = dataset[probID][subID]['exception_info']
            obj = response_cache[probID][subID]

            if ground_truth_exception_info:
                buggy_count += 1
            else:
                non_buggy_count += 1

            if not obj or not obj['accuracy']:
                continue

            accuracy = obj['accuracy']

            if accuracy['Is_Error'] is not None:
                if ground_truth_exception_info and accuracy['Is_Error']:
                    tp += 1
                elif not ground_truth_exception_info and not accuracy['Is_Error']:
                    tn += 1
                elif not ground_truth_exception_info and accuracy['Is_Error']:
                    fp += 1
                elif ground_truth_exception_info and not accuracy['Is_Error']:
                    fn += 1

            if ground_truth_exception_info and accuracy['ErrorLocation'] and accuracy['Is_Error']:
                ErrorLocation += accuracy['ErrorLocation']

            if accuracy['EM'] is not None:
                EM += accuracy['EM']
            if accuracy['PRE'][0] is not None and accuracy['PRE'][1] is not None:
                PRE_R += accuracy['PRE'][0]
                PRE_P += accuracy['PRE'][1]
            if accuracy['COV'][0] is not None and accuracy['COV'][1] is not None:
                COV_R += accuracy['COV'][0]
                COV_P += accuracy['COV'][1]

    return buggy_count, non_buggy_count, ErrorLocation, tp, fp, fn, tn, EM, PRE_R, PRE_P, COV_R, COV_P

def process_data(is_complete, dataset, response_cache):
    """Processes the data and prints the results."""
    buggy_count, non_buggy_count, ErrorLocation, tp, fp, fn, tn, EM, PRE_R, PRE_P, COV_R, COV_P = extract_accuracy_data(dataset, response_cache)
    total = buggy_count + non_buggy_count

    if is_complete:
        print("\n============== RQ1 ==============")
    else:
        print("\n============== RQ2 ==============")

    print_accuracy_metrics(total, buggy_count, non_buggy_count, ErrorLocation, EM, PRE_R, PRE_P, COV_R, COV_P)


def load_dataset(dataset_path):
    """Loads a JSON dataset from the specified path."""
    with open(dataset_path, 'r') as f:
        return json.load(f)

# Main execution
if __name__ == "__main__":
    complete_code_dataset = base_directory / 'dataset' / 'fixeval_merged_cfg.json'
    incomplete_code_dataset = base_directory / 'dataset' / 'fixeval_incom_merged_cfg.json'

    response_save_dir = base_directory / 'output' / 'baseline' / 'b2'
    complete_code_res_dir = response_save_dir / 'b2_complete_fixeval.json'
    incomplete_code_res_dir = response_save_dir / 'b2_incomplete_fixeval.json'

    print("Loading the dataset...")
    complete_code_data = load_dataset(complete_code_dataset)
    incomplete_code_data = load_dataset(incomplete_code_dataset)
    print("Loading Results...")
    complete_b2_res = load_dataset(complete_code_res_dir)
    incomplete_b2_res = load_dataset(incomplete_code_res_dir)

    print("\n===================== Complete Code Results =====================")
    process_data(True, complete_code_data, complete_b2_res)
    print("\n===================== Incomplete Code Results =====================")
    process_data(False, incomplete_code_data, incomplete_b2_res)
    print()

# Collect final system metrics and stop tracker
mem_end = psutil.virtual_memory().used / (1024**2)
cpu_end = psutil.cpu_percent(interval=None)

# Save psutil data to psutil_data.csv
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
