from codecarbon import EmissionsTracker
import psutil
import csv
import os
import json
from pathlib import Path
from utils import get_statements_from_blocks, get_scope
from accuracy import get_statement_coverage, get_statement_prefix

def calculate_accuracy_metrics(dataset, response_cache, buggy_count_func, non_buggy_count_func, accuracy_metrics_func):
    """
    Calculates accuracy metrics based on the provided dataset and response cache.

    Args:
        dataset (dict): The dataset containing ground truth data.
        response_cache (dict): The predictions made by ORCA.
        buggy_count_func (function): Function to update buggy count.
        non_buggy_count_func (function): Function to update non-buggy count.
        accuracy_metrics_func (function): Function to calculate and update accuracy metrics.

    Returns:
        tuple: A tuple containing the updated buggy count and non_buggy count.
    """
    buggy_count = 0
    non_buggy_count = 0

    for probID in response_cache:
        for subID in response_cache[probID]:
            try:
                obj = dataset[probID][subID]
                res_obj = response_cache[probID][subID]

                buggy_count, non_buggy_count = buggy_count_func(obj, buggy_count, non_buggy_count)

                if res_obj == {} or res_obj['accuracy'] == {}:
                    continue

                accuracy_metrics_func(obj, res_obj)

            except:
                continue
    return buggy_count, non_buggy_count


def calculate_rq1_2_metrics(obj, buggy_count, non_buggy_count):
    """
    Calculates buggy and non-buggy counts for RQ1 and RQ2.
    """
    exception_info = obj['exception_info']
    if exception_info:
        buggy_count += 1
    else:
        non_buggy_count += 1
    return buggy_count, non_buggy_count


def calculate_rq1_2_accuracy(obj, res_obj, error_block_match, true_positive, true_negative, false_positive, false_negative):
    """
    Calculates accuracy metrics for RQ1 and RQ2.
    """
    accuracy = res_obj['accuracy']
    exception_info = obj['exception_info']

    if accuracy['EB']:
        error_block_match += accuracy['EB']

    if exception_info and accuracy['is_error'] == True:
        true_positive += 1
    elif not exception_info and accuracy['is_error'] == False:
        true_negative += 1
    elif not exception_info and accuracy['is_error'] == True:
        false_positive += 1
    elif exception_info and accuracy['is_error'] == False:
        false_negative += 1
    return error_block_match, true_positive, true_negative, false_positive, false_negative


def print_rq1_2_results(RQ_no, table_number, buggy_count, non_buggy_count, error_block_match, true_positive, true_negative, false_positive, false_negative):
    """
    Prints the results for RQ1 and RQ2.
    """
    print(f"\n========================================= RQ{RQ_no} =========================================")
    if RQ_no == 1:
        print(f"Complete Code Results for Table {table_number} and Table {table_number + 1}\n")
    else:
        print(f"Incomplete Code Results for Table {table_number} and Table {table_number + 1}\n")

    total_instances = buggy_count + non_buggy_count
    print(f"Total Instances: {total_instances}")
    print("Buggy Instances: ", buggy_count)
    print("Non-Buggy Instances: ", non_buggy_count)
    print(f"\n---- Table {table_number}: BLOCK-LEVEL FAULT LOCALIZATION ----")
    print(f"Error Block Match: {100 * (error_block_match / buggy_count):.2f}%")
    print(f"\n---- Table {table_number + 1}: INSTANCE-LEVEL RUNTIME-ERROR DETECTION ----")
    print("True Positive Count: ", true_positive)
    print("False Positive Count: ", false_positive)
    print("False Negative Count: ", false_negative)
    print("True Negative Count: ", true_negative)
    print(f"\nTrue Positive Rate: {100 * (true_positive / buggy_count):.2f}%")
    print(f"False Positive Rate: {100 * (false_positive / non_buggy_count):.2f}%")
    print(f"False Negative Rate: {100 * (false_negative / buggy_count):.2f}%")
    print(f"True Negative Rate: {100 * (true_negative / non_buggy_count):.2f}%")
    print(f"\nAccuracy: {100 * ((true_positive + true_negative) / total_instances):.2f}%\n")


def calculate_rq1_2(RQ_no, table_number, dataset, response_cache):
    """
    Calculates and prints the results for RQ1 and RQ2.
    """
    error_block_match = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    buggy_count, non_buggy_count = calculate_accuracy_metrics(
        dataset,
        response_cache,
        calculate_rq1_2_metrics,
        calculate_rq1_2_metrics,
        lambda obj, res_obj: calculate_rq1_2_accuracy(obj, res_obj, error_block_match, true_positive, true_negative, false_positive, false_negative),
    )

    error_block_match = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for probID in response_cache:
        for subID in response_cache[probID]:
            try:
                obj = dataset[probID][subID]
                res_obj = response_cache[probID][subID]

                exception_info = obj['exception_info']
                if exception_info:
                    buggy_count += 1
                else:
                    non_buggy_count += 1

                if res_obj == {}:
                    continue
                if res_obj['accuracy'] == {}:
                    continue

                accuracy = res_obj['accuracy']

                error_block_match, true_positive, true_negative, false_positive, false_negative = calculate_rq1_2_accuracy(
                    obj, res_obj, error_block_match, true_positive, true_negative, false_positive, false_negative
                )

            except Exception as e:
                continue

    print_rq1_2_results(RQ_no, table_number, buggy_count, non_buggy_count, error_block_match, true_positive, true_negative, false_positive, false_negative)


def calculate_rq3_metrics(obj, res_obj, exact_match_execution, block_exe_pre_r, block_exe_pre_p, statement_exe_pre_r, statement_exe_pre_p, block_cov_r, block_cov_p, statement_cov_r, statement_cov_p, block_transition_r, block_transition_p):
    """
    Calculates metrics for RQ3.
    """
    gt_blocks = res_obj['gt']
    pred_blocks = res_obj['pred']
    accuracy = res_obj['accuracy']
    block_range = obj['cfg_block_range']

    pd_statement, gt_statement = get_statements_from_blocks(block_range, pred_blocks, gt_blocks)

    if accuracy['EM']:
        exact_match_execution += accuracy['EM']

    prefix_recall = accuracy['PF'][0]
    prefix_precision = accuracy['PF'][1]
    if prefix_recall and prefix_precision:
        block_exe_pre_r += prefix_recall
        block_exe_pre_p += prefix_precision

    statement_prefix_recall, statement_prefix_precision = get_statement_prefix(pd_statement, gt_statement)
    if statement_prefix_recall and statement_prefix_precision:
        statement_exe_pre_r += statement_prefix_recall
        statement_exe_pre_p += statement_prefix_precision

    block_coverage_recall = accuracy['BM'][0]
    block_coverage_precision = accuracy['BM'][1]
    if block_coverage_recall and block_coverage_precision:
        block_cov_r += block_coverage_recall
        block_cov_p += block_coverage_precision

    statement_cov_recall, statement_cov_precision = get_statement_coverage(pd_statement, gt_statement)
    if statement_cov_recall and statement_cov_precision:
        statement_cov_r += statement_cov_recall
        statement_cov_p += statement_cov_precision

    control_flow_recall = accuracy['CF'][0]
    control_flow_precision = accuracy['CF'][1]
    if control_flow_recall and control_flow_precision:
        block_transition_r += control_flow_recall
        block_transition_p += control_flow_precision
    return exact_match_execution, block_exe_pre_r, block_exe_pre_p, statement_exe_pre_r, statement_exe_pre_p, block_cov_r, block_cov_p, statement_cov_r, statement_cov_p, block_transition_r, block_transition_p


def print_rq3_results(table_1, table_2, is_complete, buggy_count, non_buggy_count, exact_match_execution, block_exe_pre_r, block_exe_pre_p, statement_exe_pre_r, statement_exe_pre_p, block_cov_r, block_cov_p, statement_cov_r, statement_cov_p, block_transition_r, block_transition_p):
    """
    Prints the results for RQ3.
    """
    count = buggy_count + non_buggy_count

    print(f"\n========================================= RQ3 =========================================")
    if is_complete:
        print(f"Complete Code Results for Table {table_1} and Table {table_2}\n")
    else:
        print(f"Incomplete Code Results for Table {table_1} and Table {table_2}\n")
    print(f"Total Instances: {count}")
    print("Buggy Instances: ", buggy_count)
    print("Non-Buggy Instances: ", non_buggy_count)
    print(f"\n---- Table {table_1}: EXECUTION TRACES AT STATEMENT LEVEL ----")
    print("\nExact Match Execution: ", f"{100 * (exact_match_execution / count):.2f}%")
    print("\nPrefix Match:")
    print(f"Recall: {100 * (statement_exe_pre_r / count):.2f}%")
    print(f"Precision: {100 * (statement_exe_pre_p / count):.2f}%")
    print("\nCoverage Match:")
    print(f"Recall: {100 * (statement_cov_r / count):.2f}%")
    print(f"Precision: {100 * (statement_cov_p / count):.2f}%\n")

    print(f"\n---- Table {table_2}: EXECUTION TRACES AT BLOCK LEVEL ----")
    print("\nExact Match Execution: ", f"{100 * (exact_match_execution / count):.2f}%")
    print("\nPrefix Match:")
    print(f"Recall: {100 * (block_exe_pre_r / count):.2f}%")
    print(f"Precision: {100 * (block_exe_pre_p / count):.2f}%")
    print("\nCoverage Match:")
    print(f"Recall: {100 * (block_cov_r / count):.2f}%")
    print(f"Precision: {100 * (block_cov_p / count):.2f}%")
    print("\nBlock Transition Match:")
    print(f"Recall: {100 * (block_transition_r / count):.2f}%")
    print(f"Precision: {100 * (block_transition_p / count):.2f}%")


def calculate_rq3(table_1, table_2, is_complete, dataset, response_cache):
    """
    Calculates and prints the results for RQ3.
    """
    buggy_count = 0
    non_buggy_count = 0

    exact_match_execution = 0
    block_exe_pre_r = 0
    block_exe_pre_p = 0
    statement_exe_pre_r = 0
    statement_exe_pre_p = 0
    block_cov_r = 0
    block_cov_p = 0
    statement_cov_r = 0
    statement_cov_p = 0
    block_transition_r = 0
    block_transition_p = 0

    buggy_count, non_buggy_count = calculate_accuracy_metrics(
        dataset,
        response_cache,
        lambda obj, buggy_count, non_buggy_count: (buggy_count + 1 if obj['exception_info'] else buggy_count, non_buggy_count + 1 if not obj['exception_info'] else non_buggy_count),
        lambda obj, buggy_count, non_buggy_count: (buggy_count, non_buggy_count),
        lambda obj, res_obj: calculate_rq3_metrics(
            obj,
            res_obj,
            exact_match_execution,
            block_exe_pre_r,
            block_exe_pre_p,
            statement_exe_pre_r,
            statement_exe_pre_p,
            block_cov_r,
            block_cov_p,
            statement_cov_r,
            statement_cov_p,
            block_transition_r,
            block_transition_p,
        ),
    )
    exact_match_execution = 0
    block_exe_pre_r = 0
    block_exe_pre_p = 0
    statement_exe_pre_r = 0
    statement_exe_pre_p = 0
    block_cov_r = 0
    block_cov_p = 0
    statement_cov_r = 0
    statement_cov_p = 0
    block_transition_r = 0
    block_transition_p = 0

    for probID in response_cache:
        for subID in response_cache[probID]:
            try:
                obj = dataset[probID][subID]
                res_obj = response_cache[probID][subID]

                if res_obj == {} or res_obj['accuracy'] == {}:
                    continue

                exact_match_execution, block_exe_pre_r, block_exe_pre_p, statement_exe_pre_r, statement_exe_pre_p, block_cov_r, block_cov_p, statement_cov_r, statement_cov_p, block_transition_r, block_transition_p = calculate_rq3_metrics(
                    obj,
                    res_obj,
                    exact_match_execution,
                    block_exe_pre_r,
                    block_exe_pre_p,
                    statement_exe_pre_r,
                    statement_exe_pre_p,
                    block_cov_r,
                    block_cov_p,
                    statement_cov_r,
                    statement_cov_p,
                    block_transition_r,
                    block_transition_p,
                )
            except:
                continue

    print_rq3_results(
        table_1,
        table_2,
        is_complete,
        buggy_count,
        non_buggy_count,
        exact_match_execution,
        block_exe_pre_r,
        block_exe_pre_p,
        statement_exe_pre_r,
        statement_exe_pre_p,
        block_cov_r,
        block_cov_p,
        statement_cov_r,
        statement_cov_p,
        block_transition_r,
        block_transition_p,
    )


def calculate_rq4_metrics(obj, res_obj, symbol_table):
    """
    Calculates metrics for RQ4.
    """
    accuracy = res_obj['accuracy']
    if accuracy['ST']:
        symbol_table += accuracy['ST']
    return symbol_table


def print_rq4_results(table_number, buggy_count, non_buggy_count, symbol_table):
    """
    Prints the results for RQ4.
    """
    count = buggy_count + non_buggy_count

    print(f"\n========================================= RQ4 =========================================")
    print(f"Complete Code Results for Table {table_number}\n")
    print(f"Total Instances: {count}")
    print("Buggy Instances: ", buggy_count)
    print("Non-Buggy Instances: ", non_buggy_count)
    print(f"\n---- Table {table_number}: VARIABLE VALUE ACCURACY ----")
    print("\nVarible Value Accuracy: ", f"{100 * (symbol_table / count):.2f}%\n")


def calculate_rq4(table_number, dataset, response_cache):
    """
    Calculates and prints the results for RQ4.
    """
    buggy_count = 0
    non_buggy_count = 0
    symbol_table = 0

    buggy_count, non_buggy_count = calculate_accuracy_metrics(
        dataset,
        response_cache,
        lambda obj, buggy_count, non_buggy_count: (buggy_count + 1 if obj['exception_info'] else buggy_count, non_buggy_count + 1 if not obj['exception_info'] else non_buggy_count),
        lambda obj, buggy_count, non_buggy_count: (buggy_count, non_buggy_count),
        lambda obj, res_obj: calculate_rq4_metrics(obj, res_obj, symbol_table),
    )

    symbol_table = 0

    for probID in response_cache:
        for subID in response_cache[probID]:
            try:
                obj = dataset[probID][subID]
                res_obj = response_cache[probID][subID]

                if res_obj == {} or res_obj['accuracy'] == {}:
                    continue

                symbol_table = calculate_rq4_metrics(obj, res_obj, symbol_table)
            except:
                continue

    print_rq4_results(table_number, buggy_count, non_buggy_count, symbol_table)


def calculate_rq5_metrics(obj, res_obj, type_statements):
    """
    Calculates metrics for RQ5.
    """
    if not obj['exception_info']:
        return type_statements

    if res_obj == {} or res_obj['accuracy'] == {}:
        return type_statements

    EB = res_obj['accuracy']['EB']

    code = obj['code']
    gt_execution = obj['ground_truth_execution_order']

    for_loop, while_loop, if_statement, simple_statement = get_scope(code)
    line_number = gt_execution[-1]
    type_statement = ""

    for scope in for_loop:
        if scope[0] <= line_number <= scope[1]:
            type_statement = "for"
            break

    for scope in while_loop:
        if scope[0] <= line_number <= scope[1]:
            type_statement = "while"
            break

    for scope in if_statement:
        if scope[0] <= line_number <= scope[1]:
            type_statement = "if"
            break

    for scope in simple_statement:
        if scope[0] <= line_number <= scope[1]:
            type_statement = "simple"
            break

    if EB == 1:
        type_statements[type_statement]["correct"] += 1
        type_statements[type_statement]["total"] += 1
    else:
        type_statements[type_statement]["incorrect"] += 1
        type_statements[type_statement]["total"] += 1

    return type_statements


def print_rq5_results(table_number, buggy_count, type_statements):
    """
    Prints the results for RQ5.
    """
    for_pre = f"{100 * ((type_statements['for']['correct'] + type_statements['while']['correct']) / (type_statements['for']['total']+ type_statements['while']['total'])):.2f}"
    if_pre = f"{100 * (type_statements['if']['correct'] / type_statements['if']['total']):.2f}"
    simple_pre = f"{100 * (type_statements['simple']['correct'] / type_statements['simple']['total']):.2f}"

    print(f"\n========================================= RQ5 =========================================")
    print(f"Complete Code Results for Table {table_number}\n")
    print("Total: ", buggy_count)
    print(f"\n---- Table {table_number}: CRASH LOCATION PROFILING ----")
    print("Error within Simple Statement: ", type_statements['simple']['total'])
    print("Detected Crashes: ", type_statements['simple']['correct'])
    print(f"Accuracy: {simple_pre}%")

    print("\nError within Branch Statement: ", type_statements['if']['total'])
    print("Detected Crashes: ", type_statements['if']['correct'])
    print(f"Accuracy: {if_pre}%")

    print("\nError within Loop Statement: ", type_statements['for']['total'] + type_statements['while']['total'])
    print("Detected Crashes: ", type_statements['for']['correct'] + type_statements['while']['correct'])
    print(f"Accuracy: {for_pre}%")


def calculate_rq5(table_number, dataset, response_cache):
    """
    Calculates and prints the results for RQ5.
    """
    buggy_count = 0
    type_statements = {
        'for': {"correct": 0, "incorrect": 0, "total": 0},
        'while': {"correct": 0, "incorrect": 0, "total": 0},
        'if': {"correct": 0, "incorrect": 0, "total": 0},
        'simple': {"correct": 0, "incorrect": 0, "total": 0},
    }

    for probID in response_cache:
        for subID in response_cache[probID]:
            try:
                obj = dataset[probID][subID]
                res_obj = response_cache[probID][subID]

                if not obj['exception_info']:
                    continue

                if res_obj == {} or res_obj['accuracy'] == {}:
                    continue

                buggy_count += 1
                type_statements = calculate_rq5_metrics(obj, res_obj, type_statements)
            except:
                continue

    print_rq5_results(table_number, buggy_count, type_statements)


def process_data(complete_dataset, incomplete_dataset, complete_response_cache, incomplete_response_cache):
    """
    Processes the output data and shows the accuracy results for the specified RQs.

    Args:
        complete_dataset (dict): Dataset for complete code.
        incomplete_dataset (dict): Dataset for incomplete code.
        complete_response_cache (dict): Response cache for complete code.
        incomplete_response_cache (dict): Response cache for incomplete code.
    """
    calculate_rq1_2(1, 1, complete_dataset, complete_response_cache)
    calculate_rq1_2(2, 3, incomplete_dataset, incomplete_response_cache)
    calculate_rq3(5, 7, True, complete_dataset, complete_response_cache)
    calculate_rq3(6, 7, False, incomplete_dataset, incomplete_response_cache)
    calculate_rq4(8, complete_dataset, complete_response_cache)
    calculate_rq5(9, complete_dataset, complete_response_cache)


def load_file(dataset_path):
    """
    Loads data from a JSON file.

    Args:
        dataset_path (str): The path to the JSON file.

    Returns:
        dict: The loaded data.
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    script_directory = Path(__file__).resolve()
    base_directory = script_directory.parents[2]

    complete_dataset_path = base_directory / "dataset" / "fixeval_merged_cfg.json"
    incomplete_dataset_path = base_directory / "dataset" / "fixeval_incom_merged_cfg.json"

    response_save_dir = base_directory / 'output' / 'orca'
    complete_dataset_response_path = response_save_dir / 'output_cfg_merged.json'
    incomplete_dataset_response_path = response_save_dir / 'output_incom_cfg_merged.json'

    complete_dataset = load_file(complete_dataset_path)
    incomplete_dataset = load_file(incomplete_dataset_path)

    complete_results = load_file(complete_dataset_response_path)
    incomplete_results = load_file(incomplete_dataset_response_path)

    process_data(complete_dataset, incomplete_dataset, complete_results, incomplete_results)
    print()

    mem_start = psutil.virtual_memory().used / (1024**2)
    cpu_start = psutil.cpu_percent(interval=None)

    csv_file = "psutil_data.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["file", "mem_start_MB", "mem_end_MB", "mem_diff_MB", "cpu_start_percent", "cpu_end_percent"])
        writer.writerow(
            [
                __file__,
                f"{mem_start:.2f}",
                f"{psutil.virtual_memory().used / (1024 ** 2):.2f}",
                f"{psutil.virtual_memory().used / (1024 ** 2) - mem_start:.2f}",
                f"{cpu_start:.2f}",
                f"{psutil.cpu_percent(interval=None):.2f}",
            ]
        )

    tracker = EmissionsTracker(project_name=r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\orca\src\orca\show_results.py")
    tracker.start()
    tracker.stop()
