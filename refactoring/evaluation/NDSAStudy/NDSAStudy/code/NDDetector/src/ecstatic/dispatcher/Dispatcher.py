from codecarbon import EmissionsTracker
import psutil
import csv
import os
import argparse
import importlib
import logging
import multiprocessing
import pathlib
from src.ecstatic.dispatcher import DockerManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

tracker = EmissionsTracker(project_name=r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\NDSAStudy\code\NDDetector\src\ecstatic\dispatcher\Dispatcher.py")
tracker.start()

mem_start = psutil.virtual_memory().used / (1024**2)
cpu_start = psutil.cpu_percent(interval=None)


def _get_choices(directory):
    return list(filter(lambda x: not x.startswith('__'), os.listdir(directory)))


def _add_argument(parser, *args, **kwargs):
    parser.add_argument(*args, **kwargs)

def _configure_parser():
    tools_dir = importlib.resources.path('src.resources', 'tools')
    benchmarks_dir = importlib.resources.path('src.resources', 'benchmarks')
    parser = argparse.ArgumentParser(description='Just a fuzzing benchmark for static analyzers')

    _add_argument(parser, '-t', '--tools', help='static analysis tools to run', nargs='+', required=True, choices=_get_choices(tools_dir))
    _add_argument(parser, '-b', '--benchmarks', help='benchmark programs to run', nargs='+', required=True, choices=_get_choices(benchmarks_dir))
    _add_argument(parser, '--tasks', help='tasks', nargs='+', required=True, default='cg', choices=['cg', 'taint', 'violation'])
    _add_argument(parser, '--no-cache', '-n', action='store_true', help='Build images without cache')
    _add_argument(parser, '--jobs', '-j', help='number of jobs', type=int, default=multiprocessing.cpu_count())
    _add_argument(parser, '--iterations', '-i', help='Number of iterations', type=int, default='1')
    _add_argument(parser, '--timeout', help='timeout', type=int)
    _add_argument(parser, '--verbose', '-v', help='verbosity', action='count', default=0)
    _add_argument(parser, '--results', help='results location', default='./results')
    _add_argument(parser, '--nondex', help='Run java tools with NonDex', action='store_true')

    return parser


def parse_args():
    parser = _configure_parser()
    return parser.parse_args()


def build_docker_images(args):
    DockerManager.build_image('base', args.no_cache)
    for tool in args.tools:
        DockerManager.build_image(tool, args.no_cache)


def run_analysis(args):
    for tool in args.tools:
        for benchmark in args.benchmarks:
            for task in args.tasks:
                DockerManager.start_runner(tool, benchmark, task, args)


def save_system_metrics(file_path, mem_start, mem_end, cpu_start, cpu_end):
    file_exists = os.path.isfile(file_path)
    with open(file_path, "a", newline="") as csvfile:
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

def main():
    args = parse_args()
    build_docker_images(args)
    run_analysis(args)


if __name__ == '__main__':
    main()

mem_end = psutil.virtual_memory().used / (1024**2)
cpu_end = psutil.cpu_percent(interval=None)

save_system_metrics("psutil_data.csv", mem_start, mem_end, cpu_start, cpu_end)

tracker.stop()
