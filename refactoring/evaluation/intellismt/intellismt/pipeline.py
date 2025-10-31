from codecarbon import EmissionsTracker
import psutil
import csv
import os
import argparse
import json
import logging
import random
import sys
from copy import deepcopy
from pathlib import Path

from dotenv import load_dotenv

import openai

from tqdm import tqdm

sys.path.append(Path.cwd().absolute() / 'intellismt')
from intellismt.modules.explorer import GPTExplorer, ClaudeExplorer, GeminiExplorer
from intellismt.modules.minimizers import SMT2Minimizer
from intellismt.modules.verifiers import UNSATVerifier

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_llm_explorer(args):
    """Initializes the LLM explorer based on the specified LLM type."""
    if args.llm == 'gpt':
        return GPTExplorer(
            model_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            few_shot=args.few_shot,
            parse_strategy=args.parse_strategy,
            combine_strategy=args.combine_strategy,
            top_p=args.top_p,
            temperature=args.temperature,
            num_responses=args.num_responses,
        )
    elif args.llm == 'claude':
        return ClaudeExplorer(
            model_name="claude-3-5-sonnet-20241022",
            few_shot=args.few_shot,
            parse_strategy=args.parse_strategy,
            combine_strategy=args.combine_strategy,
            top_p=args.top_p,
            temperature=args.temperature,
            num_responses=args.num_responses,
            max_tokens_to_sample=args.max_tokens_to_sample
        )
    elif args.llm == 'gemini':
        return GeminiExplorer(
            few_shot=args.few_shot,
            parse_strategy=args.parse_strategy,
            combine_strategy=args.combine_strategy,
            top_p=args.top_p,
            temperature=args.temperature,
            num_responses=args.num_responses,
        )
    else:
        raise ValueError(f"Unsupported LLM type: {args.llm}")


def validate_subset(subset, all_constraints, all_smt2_constraints, placeholder):
    """Validates a subset of constraints using the UNSATVerifier."""
    unsat_verifier = UNSATVerifier()
    return unsat_verifier.check(subset, all_constraints, all_smt2_constraints, placeholder)


def process_candidate(candidate_id, response, subset, usage, explorer_prompt, all_constraints, all_smt2_constraints, placeholder, logs):
    """Processes a single candidate subset and updates the logs."""
    candidate_logs = {
        "Explorer Prompt": deepcopy(explorer_prompt),
        "Explorer LLM Response": deepcopy(response),
        "Usage": deepcopy(usage),
    }

    if subset:
        is_unsat = validate_subset(subset, all_constraints, all_smt2_constraints, placeholder)
        validator_response = "UNSAT" if is_unsat else "SAT"

        logs[f"Candidate-{candidate_id}"] = {
            **candidate_logs,
            **{
                "Parsing Status": "Successful",
                "Parsed Subset": deepcopy(subset),
                "Validator Response": deepcopy(validator_response)
            },
        }
        return subset if is_unsat else None
    else:
        logs[f"Candidate-{candidate_id}"] = {
            **candidate_logs,
            **{"Parsing Status": "Unsuccessful"},
        }
        return None

def explore_and_validate(all_constraints, all_smt2_constraints, placeholder, logs, args):
    """Runs entire IntelliSMT pipeline, i.e.,
    Steps 1-2: Self-Exploration and SUS Validation
    Step 3: MUS Extraction

    Arguments:
        all_constraints (list): Complete list of constraints in string format.
        all_smt2_constraints (list): Complete list of constraints in SMT2-Lib format.
        placeholder (str): SMT2-Lib input file string placeholder, where all assertions
            are represented by "<ASSERT>" keyword.
        logs (dict): Progress logs.
        args (Namespace): Pipeline arguments.

    Returns:
        (tuple): Tuple of minimal unsatisfiable subset, as a list, and updated logs.
    """
    llm_explorer = initialize_llm_explorer(args)
    explorer_prompt = llm_explorer.get_prompt(all_constraints)
    seed = args.seed
    all_candidates = llm_explorer.explore_with_self_consistency(explorer_prompt, seed, all_constraints)
    best_subset, best_subset_id = all_constraints, None
    logs["OpenAI Chat Completion API Seed"] = seed

    for candidate_id, (response, subset, usage) in enumerate(all_candidates):
        candidate_subset = process_candidate(
            candidate_id, response, subset, usage, explorer_prompt, all_constraints, all_smt2_constraints, placeholder, logs
        )
        if candidate_subset and len(candidate_subset) < len(best_subset):
            best_subset = deepcopy(candidate_subset)
            best_subset_id = deepcopy(candidate_id)

    logs['Best candidate'] = best_subset_id
    if args.parse_strategy == 'embeddings' and hasattr(llm_explorer, "parser"):
        llm_explorer.parser.cleanup()

    return best_subset, logs


def minimize_with_smt2(
        subset, all_constraints, all_smt2_constraints, all_cvc5_assertions,
        placeholder
    ):
    """Minimizes SUS to extract the MUS with SMT solver.

    Arguments:
        subset (list): List of constraints in SUS, in string format.
        all_constraints (list): Complete list of constraints in string format.
        all_smt2_constraints (list): Complete list of constraints in SMT2-Lib format.
        all_cvc5_assertions (list): Complete list of constraints in cvc5 format.
        placeholder (str): SMT2-Lib input file string placeholder, where all assertions
            are represented by "<ASSERT>" keyword.

    Returns:
        (dict): Collection of MUS and MUS statistics.            
    """
    smt2_minimizer = SMT2Minimizer()
    mus = smt2_minimizer.minimize(
        subset, all_constraints, all_smt2_constraints, all_cvc5_assertions,
        placeholder
    )

    return {
        "SMT2 Minimizer": {
            "Number of constraints to SMT2-Minimizer": len(subset),
            "MUS": mus,
            "Number of constraints in MUS": len(mus['MUS']),
        }
    }


def minimize(
        subset, all_constraints, all_smt2_constraints, all_cvc5_assertions,
        placeholder, args,
    ):
    """Minimizes SUS to extract the MUS.

    Arguments:
        subset (list): List of constraints in SUS, in string format.
        all_constraints (list): Complete list of constraints in string format.
        all_smt2_constraints (list): Complete list of constraints in SMT2-Lib format.
        all_cvc5_assertions (list): Complete list of constraints in cvc5 format.
        placeholder (str): SMT2-Lib input file string placeholder, where all assertions
            are represented by "<ASSERT>" keyword.
        args (Namespace): Pipeline arguments.
    """
    # Use LLM for finding Minimum Unsatisfiable Subset.
    if args.use_minimizer_llm:
        pass
#        return minimize_with_llm(subset, args)

    # Use SMT2 for finding Minimum Unsatisfiable Subset.
    else:
        return minimize_with_smt2(
            subset, all_constraints, all_smt2_constraints, all_cvc5_assertions,
            placeholder
        )


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run IntelliSMT with OpenAI GPT-X")

    ## Pipeline arguments
    parser.add_argument("--path_to_data", type=str, default='dataset',
                        help="Path to processed string constraints dataset file.")
    parser.add_argument("--path_to_outputs", type=str, default='outputs_all/outputs_gpt4',
                        help="Path to processed string constraints dataset file.")
    parser.add_argument("--llm", type=str, default='gpt', choices=['gpt', 'claude', 'gemini'],
                        help=("LLM name."))
    parser.add_argument("--exrange", type=int, nargs=2, default=[0, 5],
                        help="Range of examples to process: upper-limit not considered.")
    parser.add_argument("--benchmark", type=str, default="Leetcode",
                        choices=["Leetcode", "woorpje", "nornbenchmarks", "kaluza"], help="Benchmark dataset")
    parser.add_argument("--few_shot", action='store_true',
                        help=("Whether to use exemplars for few-shot learning or not."))
    parser.add_argument("--explore_only", action='store_true',
                        help=("If True, run only Stage 1 and Stage 2 in IntelliSMT pipeline."))
    parser.add_argument("--minimize_only", action='store_true',
                        help=("If True, run only Stage 3 in IntelliSMT pipeline."))
    parser.add_argument('--num_responses', type=int, default=5,
                        help=('Number of responses to generate for Explorer LLM. Useful for '
                              'experimenting with self-consistency'))
    parser.add_argument('--seed', type=int, default=42, help="Set common system-level random seed.")
    parser.add_argument("--top_p", default=0.7, type=float,
                        help=("Only the most probable tokens with probabilities that add "
                              "up to top_p or higher are considered during decoding. "
                              "The valid range is 0.0 to 1.0. 1.0 is equivalent to disabled "
                              "and is the default. Only applies to sampling mode."))
    parser.add_argument("--temperature", default=0.6, type=float,
                        help=("A value used to warp next-token probabilities in sampling mode. "
                              "Values less than 1.0 sharpen the probability distribution, "
                              "resulting in a less random output. Values greater than 1.0 "
                              "flatten the probability distribution, resulting in a more "
                              "random output. A value of 1.0 has no effect and is the default."
                              "The allowed range is 0.0 to 2.0."))
    parser.add_argument("--use_minimizer_llm", action='store_true',
                        help=("Whether to use ``modules.minimizers.LLMMinimizer`` for finding "
                              " MUS. By default, the pipeline uses ``modules.minimizers.Z3Minimizer``"))
    parser.add_argument('--max_tokens_to_sample', type=int, default=4096,
                        help='Max tokens to sample in Claude LLM.')
    parser.add_argument("--split", type=str, default='test', choices=['val', 'test'],
                        help=("Evaluation split."))

    ## SubsetParser arguments
    parser.add_argument("--parse_strategy", type=str, default='edit', choices=['edit', 'embedding'],
                        help=("Strategy to parse and validate unsatisfiable subset "
                              "returned by Explorer LLM. If ``edit``, uses Levenshtein "
                              "distance to auto-correct. If ``embedding``, uses Cosine "
                              "Similarity based on CodeBERT embeddings to auto-correct. "
                              "Note that by default, it skips auto-correction."))
    parser.add_argument("--combine_strategy", type=str, default='mean',
                        choices=['mean', 'concat'],
                        help=("Strategy to aggregate sub-token embeddings. Only valid "
                              "when ``--parse_strategy`` is set to ``embedding``."))

    ## Unsatisfiability Validator arguments
    parser.add_argument('--timeout', type=int, default=30000,
                        help='Timeout for SMT-2 validator (in milliseconds).')

    args = parser.parse_args()

    # Set random seed.
    random.seed(args.seed)

    # Print arguments
    logger.info(f'Run arguments are: {args}')
    load_dotenv()

    # Load data
    path_to_benchmark = Path(args.path_to_data) / f"unsat.{args.benchmark}.{args.split}.json"
    logger.info(f'Loading data from {path_to_benchmark}')
    with open(path_to_benchmark, 'r') as f:
        data_instances = json.load(f)

    data_instances = data_instances[args.exrange[0]: args.exrange[1]]

    if args.use_minimizer_llm:
        path_to_outputs = Path(args.path_to_outputs) / "llm_minimizer" / 'sc' / args.split
    else:
        path_to_outputs = Path(args.path_to_outputs) / "smt2_minimizer" / 'sc' / args.split
    path_to_outputs.mkdir(exist_ok=True, parents=True)

    for instance in tqdm(data_instances, total=len(data_instances)):
        try:
            all_constraints = instance['constraints']
            all_smt2_constraints = instance['smt2_constraints']

            # File name to read from, or cache logs.
            prefix = Path(instance['path_to_smt2_formula']).parent.stem
            file_id = Path(instance['path_to_smt2_formula']).stem
            filename = f"{args.benchmark}.{prefix}.{file_id}.json"

            # IntelliSMT pipeline has three stages.
            # Stage 1: Explore logical state space with LLM-Explorer to predict UNSAT subset.
            # Stage 2: Validate if predicted UNSAT subset is indeed UNSAT.
            # Stage 3: Find MUS, either with SMT2-Minimizer or LLM-Minimizer.

            # If ``args.explore_only``, stop at the end of Stage 2 and cache logs.
            if args.explore_only:
                # Generate progress logs for Stage 1 and Stage 2.
                _, explore_only_logs = explore_and_validate(
                    all_constraints,
                    all_smt2_constraints,
                    instance['smt2_formula_placeholder'],
                    {"Path to SMT2 file": instance["path_to_smt2_formula"],
                     "Number of input constraints": len(all_constraints)},
                    args,
                )
                logs_to_save = explore_only_logs

            else:
                # If ``args.minimize_only`` is ``True``, directly skip to Stage 3. This
                # assumes that Stage 1 and Stage 2 have already been completed, and expects
                # corresponding cached logs in the ``args.path_to_outputs`` directory.
                if args.minimize_only:
                    with open(str(path_to_outputs / filename), 'r') as f:
                        explore_only_logs = json.load(f)

                    best_subset_id = explore_only_logs['Best candidate']
                    if best_subset_id:
                        subset = explore_only_logs[f'Candidate-{best_subset_id}']['Parsed Subset']
                    else:
                        subset = all_constraints

                # Since both ``args.explore_only`` and ``args.minimize_only`` are ``False``,
                # this is equivalent to running the complete IntelliSMT pipeline.
                else:
                    # Generate progress logs for Stage 1 and Stage 2.
                    subset, explore_only_logs = explore_and_validate(
                        all_constraints,
                        all_smt2_constraints,
                        instance['smt2_formula_placeholder'],
                        {"Path to SMT2 file": instance["path_to_smt2_formula"],
                        "Number of input constraints": len(all_constraints)},
                        args,
                    )

                # Generate progress logs for Stage 3.
                minimizer_logs = minimize(
                    subset,
                    all_constraints,
                    all_smt2_constraints,
                    instance['cvc5_assertions'],
                    instance['smt2_formula_placeholder'],
                    args
                )

                logs_to_save = {**explore_only_logs, **minimizer_logs}

                print(f"  Number of input constraints: {len(all_constraints)}")
                print(f"  Length of subset: {minimizer_logs['SMT2 Minimizer']['Number of constraints to SMT2-Minimizer']}")
                print(f"  Length of MUS: {len(minimizer_logs['SMT2 Minimizer']['MUS']['MUS'])}")

            # Save outputs.
            with open(str(path_to_outputs / filename), 'w') as f:
                json.dump(logs_to_save, f, indent=2)

        except:
            continue


if __name__ == '__main__':
    tracker = EmissionsTracker(project_name=r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\intellismt\pipeline.py")
    tracker.start()

    mem_start = psutil.virtual_memory().used / (1024 ** 2)
    cpu_start = psutil.cpu_percent(interval=None)

    main()

    # Collect final system metrics and stop tracker
    mem_end = psutil.virtual_memory().used / (1024 ** 2)
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
