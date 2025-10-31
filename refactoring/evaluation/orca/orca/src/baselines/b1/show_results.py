from codecarbon import EmissionsTracker
import psutil
import csv
import os
import json
from pathlib import Path

# Start CodeCarbon tracker
tracker = EmissionsTracker(project_name=r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\orca\src\baselines\b1\show_results.py")
tracker.start()

mem_start = psutil.virtual_memory().used / (1024**2)
cpu_start = psutil.cpu_percent(interval=None)

script_directory = Path(__file__).resolve()
base_directory = script_directory.parents[3]

def calculate_rates(tp, fp, fn, tn, buggy_count, non_buggy_count):
    """Calculates various rates based on the given metrics."""
    total = buggy_count + non_buggy_count
    true_positive_rate = (tp / buggy_count) * 100 if buggy_count else 0
    false_positive_rate = (fp / non_buggy_count) * 100 if non_buggy_count else 0
    false_negative_rate = (fn / buggy_count) * 100 if buggy_count else 0
    true_negative_rate = (tn / non_buggy_count) * 100 if non_buggy_count else 0
    accuracy = ((tp + tn) / total) * 100 if total else 0
    return true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate, accuracy

def print_results(is_complete, total, buggy_count, non_buggy_count, error_location, tp, fp, fn, tn, em, pre_r, pre_p, cov_r, cov_p):
    """Prints the accuracy results in a formatted table."""
    print(f"Total Instances: {total}")
    print(f"Buggy Instances: {buggy_count}")
    print(f"Non-Buggy Instances: {non_buggy_count}")
    
    table_prefix = "============== RQ1 ==============" if is_complete else "============== RQ2 =============="
    print(f"\n{table_prefix}")
    print("Table 1 - Error Localization")
    print(f"Error Localization: {100 * (error_location/buggy_count):.2f}" if buggy_count else "Error Localization: 0.00")
    
    print("\nTable 2 - Error Detection")
    print(f"True Positive Instances: {tp}")
    print(f"False Positive Instances: {fp}")
    print(f"False Negative Instances: {fn}")
    print(f"True Negative Instances: {tn}")

    tpr, fpr, fnr, tnr, accuracy = calculate_rates(tp, fp, fn, tn, buggy_count, non_buggy_count)
    print(f"\nTrue Positive Rate: {tpr:.2f}")
    print(f"False Positive Rate: {fpr:.2f}")
    print(f"False Negative Rate: {fnr:.2f}")
    print(f"True Negative Rate: {tnr:.2f}")
    print(f"\nAccuracy: {accuracy:.2f}")

    print("\n============== RQ3 ==============")
    print(f"Exact Match: {100 * (em/total):.2f}" if total else "Exact Match: 0.00")
    print(f"\nPrefix Recall: {100 * (pre_r/total):.2f}" if total else "Prefix Recall: 0.00")
    print(f"Prefix Precision: {100 * (pre_p/total):.2f}" if total else "Prefix Precision: 0.00")
    print(f"\nStatement Cov. Recall: {100 * (cov_r/total):.2f}" if total else "Statement Cov. Recall: 0.00")
    print(f"Statement Cov. Precision: {100 * (cov_p/total):.2f}" if total else "Statement Cov. Precision: 0.00")

def extract_accuracy_metrics(obj, ground_truth_exception_info):
    """Extracts and returns the accuracy metrics from a single instance."""
    accuracy = obj.get('accuracy', {})
    error_location = 0
    tp, fp, fn, tn = 0, 0, 0, 0
    em, pre_r, pre_p, cov_r, cov_p = 0, 0, 0, 0, 0

    if accuracy.get('Is_Error') is not None:
        if ground_truth_exception_info and accuracy['Is_Error'] == True:
            tp += 1
        elif not ground_truth_exception_info and accuracy['Is_Error'] == False:
            tn += 1
        elif not ground_truth_exception_info and accuracy['Is_Error'] == True:
            fp += 1
        elif ground_truth_exception_info and accuracy['Is_Error'] == False:
            fn += 1
    
    if ground_truth_exception_info and accuracy.get('ErrorLocation') and accuracy['Is_Error'] == True:
        error_location += accuracy['ErrorLocation']
    
    if accuracy.get('EM') is not None:
        em += accuracy['EM']
    
    if accuracy.get('PRE') and all(val is not None for val in accuracy['PRE']):
        pre_r += accuracy['PRE'][0]
        pre_p += accuracy['PRE'][1]
    
    if accuracy.get('COV') and all(val is not None for val in accuracy['COV']):
        cov_r += accuracy['COV'][0]
        cov_p += accuracy['COV'][1]

    return error_location, tp, fp, fn, tn, em, pre_r, pre_p, cov_r, cov_p

def process_data(is_complete, dataset, response_cache):
    """Processes the data to calculate and print accuracy metrics."""
    error_location = 0
    tp, fp, fn, tn = 0, 0, 0, 0
    em, pre_r, pre_p, cov_r, cov_p = 0, 0, 0, 0, 0
    buggy_count, non_buggy_count = 0, 0

    for prob_id in response_cache:
        for sub_id in response_cache[prob_id]:
            try:
                ground_truth_exception_info = dataset[prob_id].get(sub_id, {}).get('exception_info', '')
                obj = response_cache[prob_id].get(sub_id, {})
            except:
                continue

            if not obj or not obj.get('accuracy'):
                continue

            if ground_truth_exception_info:
                buggy_count += 1
            else:
                non_buggy_count += 1
            
            extracted_metrics = extract_accuracy_metrics(obj, ground_truth_exception_info)
            error_location += extracted_metrics[0]
            tp += extracted_metrics[1]
            fp += extracted_metrics[2]
            fn += extracted_metrics[3]
            tn += extracted_metrics[4]
            em += extracted_metrics[5]
            pre_r += extracted_metrics[6]
            pre_p += extracted_metrics[7]
            cov_r += extracted_metrics[8]
            cov_p += extracted_metrics[9]

    total = buggy_count + non_buggy_count
    print_results(is_complete, total, buggy_count, non_buggy_count, error_location, tp, fp, fn, tn, em, pre_r, pre_p, cov_r, cov_p)

def load_dataset(dataset_path):
    """Loads a JSON dataset from the specified file path."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    complete_code_dataset = base_directory / 'dataset' / 'fixeval_merged_cfg.json'
    incomplete_code_dataset = base_directory / 'dataset' / 'fixeval_incom_merged_cfg.json'
    response_save_dir = base_directory / 'output' / 'baseline' / 'b1'
    complete_code_res_dir = response_save_dir / 'b1_complete_fixeval.json'
    incomplete_code_res_dir = response_save_dir / 'b1_incomplete_fixeval.json'

    print("Loading the dataset...")
    complete_code_data = load_dataset(complete_code_dataset)
    incomplete_code_data = load_dataset(incomplete_code_dataset)
    print("Loading Results...")
    complete_b1_res = load_dataset(complete_code_res_dir)
    incomplete_b1_res = load_dataset(incomplete_code_res_dir)

    print("\n===================== Complete Code Results =====================")
    process_data(True, complete_code_data, complete_b1_res)
    print("\n===================== Incomplete Code Results =====================")
    process_data(False, incomplete_code_data, incomplete_b1_res)
    print()

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
