from codecarbon import EmissionsTracker
import psutil
import csv
import os
import json
import re
import time
import pandas as pd
from tqdm import tqdm
from unleash.evaluation.utils.common import correct_templates_and_update_files
from unleash.evaluation.utils.GA_calculator import evaluate
from unleash.evaluation.utils.template_level_analysis import evaluate_template_level, evaluate_template_level_lstm
from unleash.evaluation.utils.PA_calculator import calculate_parsing_accuracy, calculate_parsing_accuracy_lstm
from .post_process import correct_single_template
from multiprocessing import Process

TIMEOUT = 3600 * 48

def initialize_tracker():
    tracker = EmissionsTracker(project_name=r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\UNLEASH\unleash\evaluation\utils\evaluator_main.py")
    tracker.start()
    return tracker

def get_initial_system_metrics():
    return psutil.virtual_memory().used / (1024**2), psutil.cpu_percent(interval=None)

def prepare_output_directory(output_dir):
    os.makedirs(output_dir, exist_ok=True)

def prepare_result_file(output_dir):
    result_file = 'parsing_accuracy.csv'
    result_file_path = os.path.join(output_dir, result_file)
    if not os.path.exists(result_file_path):
        with open(result_file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Dataset', 'traning_time', 'parsing_time', 'identified_templates',
                             'ground_templates', 'GA', 'PA', 'FGA', 'PTA', 'RTA', 'FTA'])
    return result_file

def correct_template_general(template):
    while True:
        prev = template
        template = re.sub(r'<\*>\.<\*>', '<*>', template)
        if prev == template:
            break

    while True:
        prev = template
        template = re.sub(r'<\*><\*>', '<*>', template)
        template = re.sub(r'<\*>\:<\*>', '<*>', template)
        template = re.sub(r'<\*> <\*>', '<*>', template)
        if prev == template:
            break
    return template

def align_with_null_values(groudtruth_row):
    log = groudtruth_row['Content']
    template = groudtruth_row['EventTemplate']
    pattern_parts = template.split("<*>")
    pattern_parts_escaped = [re.escape(part) for part in pattern_parts]
    regex_pattern = "(.*?)".join(pattern_parts_escaped)
    regex = "^" + regex_pattern + "$"
    matches = re.search(regex, log)

    if matches is None:
        return template

    parts = []
    for index, part in enumerate(template.split("<*>")):
        parts.append(part)
        if index < len(matches.groups()):
            parts.append('') if matches.groups()[index] == '' else parts.append('<*>')
    return ''.join(parts)

def is_file_empty(file_path):
    try:
        return os.path.getsize(file_path) == 0
    except OSError:
        return True

def calculate_accuracies(groundtruth, parsedresult, lstm):
    filter_templates = None
    GA, FGA = evaluate(groundtruth, parsedresult, filter_templates)
    if lstm:
        PA = calculate_parsing_accuracy_lstm(groundtruth, parsedresult, filter_templates)
        tool_templates, ground_templates, FTA, PTA, RTA = evaluate_template_level_lstm(
            "", groundtruth, parsedresult, filter_templates
        )
    else:
        PA = calculate_parsing_accuracy(groundtruth, parsedresult, filter_templates)
        tool_templates, ground_templates, FTA, PTA, RTA = evaluate_template_level(
            "", groundtruth, parsedresult, filter_templates
        )
    return GA, PA, FGA, PTA, RTA, FTA, tool_templates, ground_templates

def load_time_costs(output_dir, dataset):
    time_cost_file = os.path.join(output_dir, 'time_cost.json')
    training_time, parsing_time = 0, 0
    if os.path.exists(time_cost_file):
        with open(time_cost_file, 'r') as file:
            time_table = json.load(file)
            training_time = time_table.get(dataset, {}).get('TrainingTime', 0)
            parsing_time = time_table.get(dataset, {}).get('ParsingTime', 0)
    return training_time, parsing_time

def write_results_to_file(output_dir, result_file, dataset, training_time, parsing_time, tool_templates, ground_templates, GA, PA, FGA, PTA, RTA, FTA):
    result = (
        f"{dataset},{training_time:.3f},{parsing_time:.3f},"
        f"{tool_templates},{ground_templates},{GA:.3f},{PA:.3f},"
        f"{FGA:.3f},{PTA:.3f},{RTA:.3f},{FTA:.3f}\n"
    )
    with open(os.path.join(output_dir, result_file), 'a') as summary_file:
        summary_file.write(result)

def process_evaluation(dataset, input_dir, output_dir, log_file, result_file, lstm):
    print(f"\n=== Evaluation on {dataset} ===")
    indir = os.path.join(input_dir, os.path.dirname(log_file))
    log_file_basename = os.path.basename(log_file)
    groundtruth_path = os.path.join(indir, log_file_basename + '_structured.csv')
    parsedresult_path = os.path.join(output_dir, log_file_basename + '_structured.csv')

    if not os.path.exists(parsedresult_path) or is_file_empty(parsedresult_path):
        print("No output file generated.")
        write_results_to_file(output_dir, result_file, dataset, 0, 0, "None", "None", 0, 0, 0, 0, 0, 0)
        return

    try:
        parsedresult = pd.read_csv(parsedresult_path, dtype=str)
        parsedresult.fillna("", inplace=True)
        groundtruth = pd.read_csv(groundtruth_path, dtype=str)
    except pd.errors.EmptyDataError:
        print("Empty CSV file encountered.")
        write_results_to_file(output_dir, result_file, dataset, 0, 0, "None", "None", 0, 0, 0, 0, 0, 0)
        return

    tqdm.pandas()
    groundtruth['EventTemplate'] = groundtruth.progress_apply(align_with_null_values, axis=1)
    groundtruth['EventTemplate'] = groundtruth['EventTemplate'].map(correct_template_general)
    parsedresult['EventTemplate'] = parsedresult.progress_apply(align_with_null_values, axis=1)

    GA, PA, FGA, PTA, RTA, FTA, tool_templates, ground_templates = calculate_accuracies(groundtruth, parsedresult, lstm)

    training_time, parsing_time = load_time_costs(output_dir, dataset)

    write_results_to_file(output_dir, result_file, dataset, training_time, parsing_time, tool_templates, ground_templates, GA, PA, FGA, PTA, RTA, FTA)


def evaluator(dataset, input_dir, output_dir, log_file, result_file, lstm=False):
    process_evaluation(dataset, input_dir, output_dir, log_file, result_file, lstm)

def save_psutil_data(mem_start, mem_end, cpu_start, cpu_end):
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

if __name__ == '__main__':
    tracker = initialize_tracker()
    mem_start, cpu_start = get_initial_system_metrics()
    
    # Example usage (replace with your actual parameters)
    # evaluator(dataset, input_dir, output_dir, log_file, result_file, lstm=False)
    
    mem_end = psutil.virtual_memory().used / (1024**2)
    cpu_end = psutil.cpu_percent(interval=None)
    save_psutil_data(mem_start, mem_end, cpu_start, cpu_end)
    tracker.stop()
