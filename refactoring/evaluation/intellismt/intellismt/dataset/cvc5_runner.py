from codecarbon import EmissionsTracker
import psutil
import csv
import os
import argparse
import json
import time
from pathlib import Path
from tqdm import tqdm
import cvc5
import z3
from utils import build_smt2_formula_from_string_clauses, run_fast_scandir

BENCHMARK_EXTS = {
    "Leetcode": [".smt2"],
    "woorpje": [".smt"],
    "stringfuzz": [".smt20", ".smt25"],
    "nornbenchmarks": [".smt2"],
}


def read_smt2_file(path_to_smt2_file):
    with open(path_to_smt2_file, 'r') as f:
        smt2_formula = f.read()
    return smt2_formula


def extract_smt2_formula_placeholder(path_to_smt2_file):
    smt2_formula_placeholder = []
    assert_inserted = False
    with open(path_to_smt2_file, 'r') as f:
        for line in f.readlines():
            if not line.startswith('(assert'):
                smt2_formula_placeholder.append(line)
            else:
                if not assert_inserted:
                    smt2_formula_placeholder.append('<ASSERT>')
                    assert_inserted = True
    smt2_formula_placeholder = '\n'.join(smt2_formula_placeholder)
    return smt2_formula_placeholder


def simplify_formula(smt2_formula):
    goal = z3.Goal()
    z3_formula = z3.parse_smt2_string(smt2_formula)
    goal.add(z3_formula)
    algebraic_simplified_formula = z3.Tactic('simplify')(goal)
    assert algebraic_simplified_formula.__len__() == 1
    return algebraic_simplified_formula


def extract_constraints(algebraic_simplified_formula):
    all_constraints = [algebraic_simplified_formula[0][i].__repr__()
                       for i in range(algebraic_simplified_formula[0].__len__())]
    all_constraints = [" ".join([_item.strip() for _item in item.split('\n')])
                       for item in all_constraints]
    all_smt2_constraints = [algebraic_simplified_formula[0][i].sexpr()
                            for i in range(algebraic_simplified_formula[0].__len__())]
    all_smt2_constraints = [" ".join([_item.strip() for _item in item.split('\n')])
                            for item in all_smt2_constraints]
    return all_constraints, all_smt2_constraints


def initialize_cvc5_solver(smt2_formula_placeholder):
    solver = cvc5.Solver()
    if 'set-logic' not in smt2_formula_placeholder:
        solver.setLogic("QF_SLIA")
    solver.setOption("print-success", "true")
    solver.setOption("produce-models", "true")
    solver.setOption("produce-unsat-cores", "true")
    solver.setOption("unsat-cores-mode", "assumptions")
    solver.setOption("produce-difficulty", "true")
    return solver


def run_cvc5_solver_with_options(solver, simplified_smt2_formula):
    parser = cvc5.InputParser(solver)
    parser.setStringInput(
        cvc5.InputLanguage.SMT_LIB_2_6, simplified_smt2_formula, "smt2_formula"
    )
    symbol_manager = parser.getSymbolManager()
    while True:
        cmd = parser.nextCommand()
        if cmd.isNull():
            break
        cmd.invoke(solver, symbol_manager)


def process_unsat_core(solver, instance, unsat_check_start_time, mus_option):
    unsat_core = solver.getUnsatCore()
    unsat_core_end_time = time.time()

    if mus_option == "false":
        cvc5_assertions = [str(assertion) for assertion in solver.getAssertions()]
        statistics = solver.getStatistics().get()
        instance.update({
            'cvc5_assertions': cvc5_assertions,
            'unsat_check_time (in ms)': '{:.3f}'.format((unsat_core_end_time - unsat_check_start_time) * 1000),
            'unsat_core': [str(item) for item in unsat_core],
            'unsat_core_time (in ms)': "{:.3f}".format((unsat_core_end_time - unsat_check_start_time) * 1000),
            'unsat_core_statistics': statistics,
        })
    elif mus_option == "true":
        instance.update({
            'minimal_unsat_core': [str(item) for item in unsat_core],
            'mimimal_unsat_core_time (in ms)': "{:.3f}".format((unsat_core_end_time - unsat_check_start_time) * 1000),
        })


def analyze_with_cvc5(instance, simplified_smt2_formula, smt2_formula_placeholder):
    for mus_option in ["false", "true"]:
        solver = initialize_cvc5_solver(smt2_formula_placeholder)
        run_cvc5_solver_with_options(solver, simplified_smt2_formula)
        unsat_check_start_time = time.time()
        result = solver.checkSat()
        difficulty = solver.getDifficulty()
        instance['difficulty'] = dict(zip(
            [str(item) for item in difficulty.keys()],
            [str(item) for item in difficulty.values()],
        ))
        if not result.isUnsat():
            return None
        process_unsat_core(solver, instance, unsat_check_start_time, mus_option)
        solver.resetAssertions()
    return instance


def run_cvc5_solver(path_to_smt2_file):
    smt2_formula = read_smt2_file(path_to_smt2_file)
    smt2_formula_placeholder = extract_smt2_formula_placeholder(path_to_smt2_file)
    algebraic_simplified_formula = simplify_formula(smt2_formula)
    all_constraints, all_smt2_constraints = extract_constraints(algebraic_simplified_formula)

    if not [item for item in all_constraints if item != "False"]:
        return None
    simplified_smt2_formula = build_smt2_formula_from_string_clauses(
        all_constraints, all_smt2_constraints, smt2_formula_placeholder
    )
    instance = {
        'path_to_smt2_formula': str(path_to_smt2_file),
        'smt2_formula_placeholder': smt2_formula_placeholder,
        'constraints': all_constraints,
        'smt2_constraints': all_smt2_constraints
    }
    return analyze_with_cvc5(instance, simplified_smt2_formula, smt2_formula_placeholder)


if __name__ == "__main__":
    tracker = EmissionsTracker(project_name=r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\intellismt\dataset\cvc5_runner.py")
    tracker.start()
    mem_start = psutil.virtual_memory().used / (1024 ** 2)
    cpu_start = psutil.cpu_percent(interval=None)
    parser = argparse.ArgumentParser(
        description='Create dataset from SMT2 files in the benchmark folder.'
    )
    parser.add_argument('--benchmark_dir', type=str, default="models/Strings",
                        help='Benchmark folder')
    parser.add_argument('--type', type=str, default='Leetcode',
                        choices=['Leetcode', 'stringfuzz', 'woorpje', 'nornbenchmarks'],
                        help='Choice of benchmark.')
    parser.add_argument('--timeout', default=60000,
                        help='Maximum timeout (in milliseconds).')
    args = parser.parse_args()
    path_to_models = str(Path(args.benchmark_dir) / args.type)
    _, path_to_formulae = run_fast_scandir(path_to_models, BENCHMARK_EXTS[args.type])
    print(f'Number of examples in {args.type} benchmark: {len(path_to_formulae)}')
    path_to_formulae = sorted(path_to_formulae)
    data_instances = []
    for fid, path in tqdm(enumerate(path_to_formulae), total=len(path_to_formulae)):
        try:
            output = run_cvc5_solver(path)
            if output:
                data_instances.append(output)
        except:
            continue
    print(f"Number of data instances: {len(data_instances)}")
    print(f"Writing output to unsat.{args.type}.json")
    with open(f'unsat.{args.type}.json', 'w') as f:
        json.dump(data_instances, f, indent=2)
    mem_end = psutil.virtual_memory().used / (1024 ** 2)
    cpu_end = psutil.cpu_percent(interval=None)
    csv_file = "psutil_data.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(
                ["file", "mem_start_MB", "mem_end_MB", "mem_diff_MB", "cpu_start_percent", "cpu_end_percent"])
        writer.writerow([
            __file__,
            f"{mem_start:.2f}",
            f"{mem_end:.2f}",
            f"{mem_end - mem_start:.2f}",
            f"{cpu_start:.2f}",
            f"{cpu_end:.2f}"
        ])
    tracker.stop()
