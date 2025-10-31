#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import click
import importlib.metadata
import psutil
from codecarbon import EmissionsTracker

from magika import Magika, MagikaError, PredictionMode, colors
from magika.logger import get_logger
from magika.types import ContentTypeLabel, MagikaResult
from magika.types.overwrite_reason import OverwriteReason

VERSION = importlib.metadata.version("magika")
CONTACT_EMAIL = "magika-dev@google.com"
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
HELP_EPILOG = f"""
Magika version: "{VERSION}"\f
Default model: "{Magika._get_default_model_name()}"

Send any feedback to {CONTACT_EMAIL} or via GitHub issues.
"""


def configure_logging(verbose: bool, debug: bool, with_colors: bool) -> logging.Logger:
    _l = get_logger(use_colors=with_colors)
    if verbose:
        _l.setLevel(logging.INFO)
    if debug:
        _l.setLevel(logging.DEBUG)
    return _l


def handle_version(output_version: bool, _l: logging.Logger) -> None:
    if output_version:
        _l.raw_print_to_stdout("Magika python client")
        _l.raw_print_to_stdout(f"Magika version: {VERSION}")
        _l.raw_print_to_stdout(f"Default model: {Magika._get_default_model_name()}")
        sys.exit(0)


def validate_input_paths(
    files_paths: List[Path], _l: logging.Logger
) -> bool:
    read_from_stdin = False
    for p in files_paths:
        if str(p) == "-":
            read_from_stdin = True
        elif not p.exists():
            _l.error(f'File or directory "{str(p)}" does not exist.')
            sys.exit(1)
    if read_from_stdin:
        if len(files_paths) > 1:
            _l.error('If you pass "-", you cannot pass anything else.')
            sys.exit(1)
        return False
    return True


def expand_paths_if_recursive(files_paths: List[Path], recursive: bool) -> List[Path]:
    if not recursive:
        return files_paths

    expanded_paths = []
    for p in files_paths:
        if p.exists():
            if p.is_file():
                expanded_paths.append(p)
            elif p.is_dir():
                expanded_paths.extend(sorted(p.rglob("*")))
        elif str(p) == "-":
            pass
        else:
            pass  # Already handled in validate_input_paths

    return list(filter(lambda x: not x.is_dir(), expanded_paths))


def validate_arguments(
    files_paths: List[Path],
    batch_size: int,
    json_output: bool,
    jsonl_output: bool,
    mime_output: bool,
    label_output: bool,
    magic_compatibility_mode: bool,
    _l: logging.Logger,
) -> None:
    if len(files_paths) == 0:
        _l.error("You need to pass at least one path, or - to read from stdin.")
        sys.exit(1)

    if not validate_input_paths(files_paths, _l):
        return

    if batch_size <= 0 or batch_size > 512:
        _l.error("Batch size needs to be greater than 0 and less or equal than 512.")
        sys.exit(1)

    if json_output and jsonl_output:
        _l.error("You should use either --json or --jsonl, not both.")
        sys.exit(1)

    if int(mime_output) + int(label_output) + int(magic_compatibility_mode) > 1:
        _l.error("You should use only one of --mime, --label, --compatibility-mode.")
        sys.exit(1)


def initialize_magika(
    model_dir: Optional[Path],
    prediction_mode_str: str,
    no_dereference: bool,
    verbose: bool,
    debug: bool,
    with_colors: bool,
    _l: logging.Logger,
) -> Magika:
    if model_dir is None:
        model_dir_str = os.environ.get("MAGIKA_MODEL_DIR")
        if model_dir_str is not None and model_dir_str.strip() != "":
            model_dir = Path(model_dir_str)

    try:
        magika = Magika(
            model_dir=model_dir,
            prediction_mode=PredictionMode(prediction_mode_str),
            no_dereference=no_dereference,
            verbose=verbose,
            debug=debug,
            use_colors=with_colors,
        )
    except MagikaError as mr:
        _l.error(str(mr))
        sys.exit(1)
    return magika


def get_output_strings(
    result: MagikaResult,
    mime_output: bool,
    label_output: bool,
    magic_compatibility_mode: bool,
) -> Tuple[str, str, str]:
    start_color = ""
    end_color = ""
    if result.ok:
        if mime_output:
            output = result.prediction.output.mime_type
        elif label_output:
            output = str(result.prediction.output.label)
        elif magic_compatibility_mode:
            output = result.prediction.output.description
        else:
            output = f"{result.prediction.output.description} ({result.prediction.output.group})"
            if (
                result.prediction.dl.label != ContentTypeLabel.UNDEFINED
                and result.prediction.dl.label != result.prediction.output.label
                and result.prediction.overwrite_reason == OverwriteReason.LOW_CONFIDENCE
            ):
                output += (
                    " [Low-confidence model best-guess: "
                    f"{result.prediction.dl.description} ({result.prediction.dl.group}), "
                    f"score={result.prediction.score}]"
                )
        if not magic_compatibility_mode:
            color_by_group = {
                "document": colors.LIGHT_PURPLE,
                "executable": colors.LIGHT_GREEN,
                "archive": colors.LIGHT_RED,
                "audio": colors.YELLOW,
                "image": colors.YELLOW,
                "video": colors.YELLOW,
                "code": colors.LIGHT_BLUE,
            }
            start_color = color_by_group.get(
                result.prediction.output.group, colors.WHITE
            )
            end_color = colors.RESET
    else:
        output = result.status
    return start_color, end_color, output


def process_batch(
    magika: Magika,
    batch_files_paths: List[Path],
    json_output: bool,
    jsonl_output: bool,
    mime_output: bool,
    label_output: bool,
    magic_compatibility_mode: bool,
    output_score: bool,
    _l: logging.Logger,
    all_predictions: List[Tuple[Path, MagikaResult]],
) -> None:
    batch_predictions = (
        [get_magika_result_from_stdin(magika)]
        if should_read_from_stdin(batch_files_paths)
        else magika.identify_paths(batch_files_paths)
    )

    if json_output:
        all_predictions.extend(zip(batch_files_paths, batch_predictions))
    elif jsonl_output:
        for file_path, result in zip(batch_files_paths, batch_predictions):
            _l.raw_print_to_stdout(json.dumps(result.asdict()))
    else:
        for file_path, result in zip(batch_files_paths, batch_predictions):
            start_color, end_color, output = get_output_strings(
                result, mime_output, label_output, magic_compatibility_mode
            )

            if output_score and result.ok:
                score = int(result.prediction.score * 100)
                _l.raw_print_to_stdout(
                    f"{start_color}{file_path}: {output} {score}%{end_color}"
                )
            else:
                _l.raw_print_to_stdout(
                    f"{start_color}{file_path}: {output}{end_color}"
                )


def should_read_from_stdin(files_paths: List[Path]) -> bool:
    return len(files_paths) == 1 and str(files_paths[0]) == "-"


def get_magika_result_from_stdin(magika: Magika) -> MagikaResult:
    content = sys.stdin.buffer.read()
    result = magika.identify_bytes(content)
    return result


@click.command(
    context_settings=CONTEXT_SETTINGS,
    epilog=HELP_EPILOG,
)
@click.argument(
    "file",
    type=click.Path(exists=False, readable=False, path_type=Path),
    required=False,
    nargs=-1,
)
@click.option(
    "-r",
    "--recursive",
    is_flag=True,
    help='When passing this option, magika scans every file within directories, instead of outputting "directory"',
)
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format.")
@click.option("--jsonl", "jsonl_output", is_flag=True, help="Output in JSONL format.")
@click.option(
    "-i",
    "--mime-type",
    "mime_output",
    is_flag=True,
    help="Output the MIME type instead of a verbose content type description.",
)
@click.option(
    "-l",
    "--label",
    "label_output",
    is_flag=True,
    help="Output a simple label instead of a verbose content type description. Use --list-output-content-types for the list of supported output.",
)
@click.option(
    "-c",
    "--compatibility-mode",
    "magic_compatibility_mode",
    is_flag=True,
    help="Compatibility mode: output is as close as possible to `file` and colors are disabled.",
)
@click.option(
    "-s",
    "--output-score",
    is_flag=True,
    help="Output the prediction's score in addition to the content type.",
)
@click.option(
    "-m",
    "--prediction-mode",
    "prediction_mode_str",
    type=click.Choice(PredictionMode.get_valid_prediction_modes(), case_sensitive=True),
    default=PredictionMode.HIGH_CONFIDENCE,
)
@click.option(
    "--batch-size", default=32, help="How many files to process in one batch."
)
@click.option(
    "--no-dereference",
    is_flag=True,
    help="This option causes symlinks not to be followed. By default, symlinks are dereferenced.",
)
@click.option(
    "--colors/--no-colors",
    "with_colors",
    is_flag=True,
    default=True,
    help="Enable/disable use of colors.",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable more verbose output.")
@click.option("-vv", "--debug", is_flag=True, help="Enable debug logging.")
@click.option(
    "--version", "output_version", is_flag=True, help="Print the version and exit."
)
@click.option(
    "--model-dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
    ),
    help="Use a custom model.",
)
def main(
    file: List[Path],
    recursive: bool,
    json_output: bool,
    jsonl_output: bool,
    mime_output: bool,
    label_output: bool,
    magic_compatibility_mode: bool,
    output_score: bool,
    prediction_mode_str: str,
    batch_size: int,
    no_dereference: bool,
    with_colors: bool,
    verbose: bool,
    debug: bool,
    output_version: bool,
    model_dir: Optional[Path],
) -> None:
    """
    Magika - Determine type of FILEs with deep-learning.
    """
    files_paths = file

    if magic_compatibility_mode:
        with_colors = False

    _l = configure_logging(verbose, debug, with_colors)
    handle_version(output_version, _l)

    files_paths = expand_paths_if_recursive(files_paths, recursive)

    validate_arguments(
        files_paths,
        batch_size,
        json_output,
        jsonl_output,
        mime_output,
        label_output,
        magic_compatibility_mode,
        _l,
    )

    _l.info(f"Considering {len(files_paths)} files")
    _l.debug(f"Files: {files_paths}")

    magika = initialize_magika(
        model_dir,
        prediction_mode_str,
        no_dereference,
        verbose,
        debug,
        with_colors,
        _l,
    )

    all_predictions: List[Tuple[Path, MagikaResult]] = []

    batches_num = len(files_paths) // batch_size
    if len(files_paths) % batch_size != 0:
        batches_num += 1

    for batch_idx in range(batches_num):
        batch_files_paths = files_paths[
            batch_idx * batch_size : (batch_idx + 1) * batch_size
        ]
        process_batch(
            magika,
            batch_files_paths,
            json_output,
            jsonl_output,
            mime_output,
            label_output,
            magic_compatibility_mode,
            output_score,
            _l,
            all_predictions,
        )

    if json_output:
        _l.raw_print_to_stdout(
            json.dumps(
                [result.asdict() for (_, result) in all_predictions],
                indent=4,
            )
        )


if __name__ == "__main__":
    tracker = EmissionsTracker(project_name=r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\magika\python\src\magika\cli\magika_client.py")
    tracker.start()

    mem_start = psutil.virtual_memory().used / (1024**2)
    cpu_start = psutil.cpu_percent(interval=None)

    main()

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
