from codecarbon import EmissionsTracker
import psutil
import csv
import os
import argparse
import json
import logging
import random
import statistics
from pathlib import Path
from tqdm import tqdm
import cvc5
from utils import (
    build_smt2_formula_from_string_constraints,
    set_cvc5_options_for_unsat,
    parse_input_formula
)
from intrinsic_search import init_solver, check_subset_unsat

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class UNSATVerifier:
    def __init__(self):
        self.solver = cvc5.Solver()
        set_cvc5_options_for_unsat(self.solver)
        self.statistics = None

    def reset(self):
        self.solver.resetAssertions()

    def check(self, constraints, all_constraints, all_smt2_constraints, placeholder):
        smt2_formula = build_smt2_formula_from_string_constraints(
            constraints, all_constraints, all_smt2_constraints, placeholder
        )
        if 'set-logic' not in smt2_formula:
            try:
                self.solver.setLogic("QF_SLIA")
            except:
                pass
        parse_input_formula(self.solver, smt2_formula, "smt_formula")
        result = self.solver.checkSat().isUnsat()
        statistics_dict = self.solver.getStatistics().get()
        setattr(self, "statistics", statistics_dict)
        return result


def load_data(path_to_data, split):
    path_to_benchmark = Path(path_to_data) / f"unsat.Leetcode.{split}.json"
    logger.info(f'Loading data from {path_to_benchmark}')
    with open(path_to_benchmark, 'r') as f:
        data_instances = json.load(f)
    return data_instances


def run_instance_checks(instance, n, solver, unsat_verifier):
    all_clauses = instance['constraints']
    all_smt2_clauses = instance['smt2_constraints']
    check_subset_unsat(solver, all_smt2_clauses, instance['smt2_formula_placeholder'])
    mus = solver.getUnsatCore()
    is_unsat = False
    for _ in range(n):
        subset = [element for element in all_clauses if random.choice([True, False])]
        is_unsat = unsat_verifier.check(
            subset, all_clauses, all_smt2_clauses, instance['smt2_formula_placeholder']
        )
        if is_unsat:
            break
    try:
        ratio = (len(all_clauses) - len(subset)) / (len(all_clauses) - len(mus))
    except ZeroDivisionError:
        ratio = 0.0
    return is_unsat, ratio, all_clauses, subset


def update_stratified_results(stratified_results, is_unsat, ratio, all_clauses, subset):
    reduction = (len(all_clauses) - len(subset)) / len(all_clauses)
    lower_lim = len(all_clauses) - (len(all_clauses) % 10)
    key = f'{lower_lim}-{lower_lim + 10}'
    stratified_results = update_results_for_key(stratified_results, key, is_unsat, ratio)
    stratified_results = update_results_for_key(stratified_results, 'Total', is_unsat, ratio)
    return reduction, stratified_results


def update_results_for_key(stratified_results, key, is_unsat, ratio):
    if key in stratified_results:
        [correct, total], existing_ratio = stratified_results[key]
        if is_unsat:
            correct += 1
        total += 1
        stratified_results[key] = [[correct, total], ratio + existing_ratio]
    else:
        if is_unsat:
            stratified_results[key] = [[1, 1], ratio]
        else:
            stratified_results[key] = [[0, 1], ratio]
    return stratified_results


def calculate_and_print_results(stratified_results, pct, pct_corr):
    print(f"Mean constraint reduction: {pct:.2%}")
    print(f"Mean constraint reduction, corrected: {pct_corr:.2%}")
    print("Stratification based on intervals of total constraints:")
    print("Interval\tC/N\tAdjusted r\tAbsolute r")
    for key, ratio_counter in stratified_results.items():
        correct, total = ratio_counter[0]
        try:
            adjusted_ratio = ratio_counter[1] / correct
        except ZeroDivisionError:
            adjusted_ratio = 0.0
        absolute_ratio = ratio_counter[1] / total
        print(f'{key}\t\t{correct}/{total}\t{adjusted_ratio:.3f}\t\t{absolute_ratio:.3f}')
    return


def stratify_by_interval(path_to_data, split, n, seed=None):
    if seed:
        random.seed(seed)
    data_instances = load_data(path_to_data, split)
    solver = init_solver()
    stratified_results = {}
    pct, pct_corr = [], []
    unsat_verifier = UNSATVerifier()
    for instance in tqdm(data_instances, total=len(data_instances)):
        is_unsat, ratio, all_clauses, subset = run_instance_checks(instance, n, solver, unsat_verifier)
        reduction, stratified_results = update_stratified_results(stratified_results, is_unsat, ratio, all_clauses, subset)
        pct.append(reduction if is_unsat else 0)
        pct_corr.append(reduction)
    stratified_results = dict(sorted(stratified_results.items()))
    return stratified_results, statistics.mean(pct), statistics.mean(pct_corr)


if __name__ == '__main__':
    tracker = EmissionsTracker(project_name=r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\intellismt\experiments\intrinsic_naive.py")
    tracker.start()
    mem_start = psutil.virtual_memory().used / (1024**2)
    cpu_start = psutil.cpu_percent(interval=None)

    parser = argparse.ArgumentParser(description="Run IntelliSMT with Naive Baseline")
    parser.add_argument("--path_to_data", type=str, default='../dataset',
                        help="Path to processed string constraints dataset file.")
    parser.add_argument('--seed', type=int, default=42, help="Set common system-level random seed.")
    parser.add_argument("--split", type=str, default='test', choices=['val', 'test'],
                        help="Evaluation split.")
    parser.add_argument('--n', type=int, default=5, help="Number of responses.")
    args = parser.parse_args()
    random.seed(args.seed)
    logger.info(f'Run arguments are: {args}')
    stratified_results, pct, pct_corr = stratify_by_interval(args.path_to_data, args.split, args.n)
    calculate_and_print_results(stratified_results, pct, pct_corr)

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
