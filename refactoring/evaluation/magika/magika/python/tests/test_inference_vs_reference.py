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

from codecarbon import EmissionsTracker
import psutil
import csv
import os

# Start CodeCarbon tracker with project name "C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\magika\python\tests\test_inference_vs_reference.py" and collect initial system metrics
tracker = EmissionsTracker(project_name=r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\magika\python\tests\test_inference_vs_reference.py")
tracker.start()

mem_start = psutil.virtual_memory().used / (1024**2)
cpu_start = psutil.cpu_percent(interval=None)

from __future__ import annotations

import base64
import enum
import json
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set, Tuple

import click
import dacite
import pytest
from tqdm import tqdm

from magika import ContentTypeLabel, Magika, PredictionMode
from magika.types import MagikaResult, OverwriteReason
from magika.types.status import Status

try:
    from tests import utils as test_utils
except ImportError:
    # Hack to support both `uv run pytest tests/` and `uv run ./tests/test_...
    # <command>`
    import sys

    sys.path.append(str(Path(__file__).parent.parent))
    from tests import utils as test_utils


@click.group()
def cli():
    pass


@cli.command()
@click.option("--debug/--no-debug", is_flag=True, default=True)
def run_tests(debug: bool) -> None:
    test_inference_vs_reference(debug=debug)


@cli.command()
@click.option("--test-mode", is_flag=True)
def generate_tests(test_mode: bool) -> None:
    _generate_reference_for_inference(test_mode=test_mode)


def test_inference_vs_reference(debug: bool = False) -> None:
    repo_root_dir = test_utils.get_repo_root_dir()

    magika_by_prediction_mode: Dict[PredictionMode, Magika] = {}
    for prediction_mode in [
        PredictionMode.HIGH_CONFIDENCE,
        PredictionMode.MEDIUM_CONFIDENCE,
        PredictionMode.BEST_GUESS,
    ]:
        magika_by_prediction_mode[prediction_mode] = Magika(
            prediction_mode=prediction_mode
        )

    model_name = magika_by_prediction_mode[
        PredictionMode.HIGH_CONFIDENCE
    ].get_model_name()

    examples_by_path = _get_examples_by_path(model_name)
    if debug:
        print(f"Loaded {len(examples_by_path)} examples by path")
    for example in tqdm(examples_by_path, disable=not debug):
        m = magika_by_prediction_mode[example.prediction_mode]
        abs_path = repo_root_dir / example.path
        _run_path_tests(m, abs_path, example)

        example_content = abs_path.read_bytes()
        _run_content_tests(m, example_content, example)

    examples_by_content = _get_examples_by_content(model_name)
    if debug:
        print(f"Loaded {len(examples_by_content)} examples by content")
    for example in tqdm(examples_by_content, disable=not debug):
        m = magika_by_prediction_mode[example.prediction_mode]
        example_content = base64.b64decode(example.content_base64)
        _run_content_tests(m, example_content, example)


def _run_path_tests(m: Magika, abs_path: Path, example: ExampleByPath) -> None:
    result = m.identify_path(abs_path)
    _check_result_vs_reference_example(
        result, abs_path, example.status, example.prediction
    )

    result = m.identify_bytes(abs_path.read_bytes())
    _check_result_vs_reference_example(
        result, Path("-"), example.status, example.prediction
    )

    with open(abs_path, "rb") as f:
        result = m.identify_stream(f)
        _check_result_vs_reference_example(
            result, Path("-"), example.status, example.prediction
        )


def _run_content_tests(m: Magika, content: bytes, example: ExampleByContent) -> None:
    result = m.identify_bytes(content)
    _check_result_vs_reference_example(
        result, Path("-"), example.status, example.prediction
    )

    with tempfile.TemporaryDirectory() as td:
        tf_path = Path(td) / "file.bin"
        tf_path.write_bytes(content)

        result = m.identify_path(tf_path)
        _check_result_vs_reference_example(
            result, tf_path, example.status, example.prediction
        )

        with open(tf_path, "rb") as f:
            result = m.identify_stream(f)
            _check_result_vs_reference_example(
                result, Path("-"), example.status, example.prediction
            )


def test_reference_generation() -> None:
    # This is useful to exercise the various paths to make sure the reference
    # generation stays up to date.
    _generate_reference_for_inference(test_mode=True)


def _get_examples_by_path(model_name: str) -> List[ExampleByPath]:
    reference_for_inference_examples_by_path = (
        test_utils.get_reference_for_inference_examples_by_path_path(model_name)
    )
    return _load_examples(
        reference_for_inference_examples_by_path, ExampleByPath
    )


def _get_examples_by_content(model_name: str) -> List[ExampleByContent]:
    reference_for_inference_examples_by_content = (
        test_utils.get_reference_for_inference_examples_by_content_path(model_name)
    )
    return _load_examples(
        reference_for_inference_examples_by_content, ExampleByContent
    )


def _load_examples(path: Path, example_type: type) -> List[type]:
    return [
        dacite.from_dict(
            example_type,
            entry,
            config=dacite.Config(
                cast=[ContentTypeLabel, OverwriteReason, PredictionMode, Status]
            ),
        )
        for entry in json.loads(
            test_utils.gzip_decompress(path.read_bytes())
        )
    ]


def _generate_reference_for_inference(test_mode: bool) -> None:
    model_name = Magika._get_default_model_name()
    examples_by_path = _generate_examples_by_path_reference(model_name)
    _dump_examples_by_path(model_name, examples_by_path, test_mode=test_mode)
    examples_by_content = _generate_examples_by_content_reference(
        model_name, test_mode=test_mode
    )
    _dump_examples_by_content(model_name, examples_by_content, test_mode=test_mode)


def _generate_examples_by_path_reference(
    model_name: str,
) -> List[ExampleByPath]:
    print(f'Generating examples by path for model "{model_name}"...')

    repo_root_dir = test_utils.get_repo_root_dir()
    tests_paths = test_utils.get_basic_test_files_paths()
    examples_by_path = []

    for prediction_mode in [
        PredictionMode.HIGH_CONFIDENCE,
        PredictionMode.MEDIUM_CONFIDENCE,
        PredictionMode.BEST_GUESS,
    ]:
        m = Magika(prediction_mode=prediction_mode)
        assert m.get_model_name() == model_name

        for test_path in tqdm(tests_paths):
            result = m.identify_path(test_path)
            example = _create_example_by_path(
                result, test_path, repo_root_dir, prediction_mode
            )
            examples_by_path.append(example)

    return examples_by_path


def _create_example_by_path(
    result: MagikaResult,
    test_path: Path,
    repo_root_dir: Path,
    prediction_mode: PredictionMode,
) -> ExampleByPath:
    if result.ok:
        example = ExampleByPath(
            prediction_mode=prediction_mode,
            path=str(test_path.resolve().relative_to(repo_root_dir)),
            status=result.status,
            prediction=Prediction(
                dl=result.prediction.dl.label,
                output=result.prediction.output.label,
                score=result.prediction.score,
                overwrite_reason=result.prediction.overwrite_reason,
            ),
        )
    else:
        example = ExampleByPath(
            prediction_mode=prediction_mode,
            path=str(test_path),
            status=result.status,
            prediction=None,
        )
    return example


def _generate_examples_by_content_reference(
    model_name: str, test_mode: bool
) -> List[ExampleByContent]:
    random.seed(42)

    print(f'Generating examples by content for model "{model_name}"...')
    magika = Magika()
    assert magika.get_model_name() == model_name

    content_list = _generate_content_list(magika)
    examples_by_content = _generate_examples_for_all_modes(
        magika, content_list, test_mode
    )

    return examples_by_content


def _generate_content_list(magika: Magika) -> List[bytes]:
    content_list = [b""]
    config = magika._model_config
    for size in [
        config.min_file_size_for_dl - 1,
        config.min_file_size_for_dl,
        config.min_file_size_for_dl + 1,
        config.beg_size - 1,
        config.beg_size,
        config.beg_size + 1,
        config.end_size - 1,
        config.end_size,
        config.end_size + 1,
        config.beg_size + config.end_size - 1,
        config.beg_size + config.end_size,
        config.beg_size + config.end_size + 1,
        config.block_size - 1,
        config.block_size,
        config.block_size + 1,
    ]:
        content_list.append(test_utils.generate_pattern(size, only_printable=True))
        content_list.append(test_utils.generate_pattern(size, only_printable=False))

    collector = CornerCaseCollector(magika)
    generator = collector.get_corner_case_candidates_generator()
    for candidate_idx, (source_info, content) in enumerate(generator):
        is_useful, _, _ = collector.inspect_content(content)
        if is_useful:
            content_list.append(content)
            if collector.is_complete():
                break

    return content_list


def _generate_examples_for_all_modes(
    magika: Magika, content_list: List[bytes], test_mode: bool
) -> List[ExampleByContent]:
    examples_by_content = []
    for prediction_mode in [
        PredictionMode.HIGH_CONFIDENCE,
        PredictionMode.MEDIUM_CONFIDENCE,
        PredictionMode.BEST_GUESS,
    ]:
        magika_instance = Magika(prediction_mode=prediction_mode)
        for content in content_list:
            result = magika_instance.identify_bytes(content)
            example = _create_example_by_content(
                result, content, prediction_mode
            )
            examples_by_content.append(example)
        if test_mode and len(examples_by_content) > 100:
            break
    return examples_by_content


def _create_example_by_content(
    result: MagikaResult, content: bytes, prediction_mode: PredictionMode
) -> ExampleByContent:
    if result.ok:
        example = ExampleByContent(
            prediction_mode=prediction_mode,
            content_base64=base64.b64encode(content).decode("ascii"),
            status=result.status,
            prediction=Prediction(
                dl=result.prediction.dl.label,
                output=result.prediction.output.label,
                score=result.prediction.score,
                overwrite_reason=result.prediction.overwrite_reason,
            ),
        )
    else:
        example = ExampleByContent(
            prediction_mode=prediction_mode,
            content_base64=base64.b64encode(content).decode("ascii"),
            status=result.status,
            prediction=None,
        )
    return example


def _dump_examples_by_path(
    model_name: str,
    examples_by_path: List[ExampleByPath],
    test_mode: bool,
) -> None:
    examples_by_path_path = (
        test_utils.get_reference_for_inference_examples_by_path_path(model_name)
    )

    if test_mode:
        print(
            f'WARNING: running in "test_mode", not writing examples by path to {examples_by_path_path}'
        )
    else:
        examples_by_path_path.parent.mkdir(parents=True, exist_ok=True)
        _write_examples(examples_by_path, examples_by_path_path)
        print(
            f"Wrote {len(examples_by_path)} examples by path to {examples_by_path_path}"
        )


def _dump_examples_by_content(
    model_name: str,
    examples_by_content: List[ExampleByContent],
    test_mode: bool,
) -> None:
    examples_by_content_path = (
        test_utils.get_reference_for_inference_examples_by_content_path(model_name)
    )

    if test_mode:
        print(
            f'WARNING: running in "test_mode", not writing examples by content to {examples_by_content_path}'
        )
    else:
        examples_by_content_path.parent.mkdir(parents=True, exist_ok=True)
        _write_examples(examples_by_content, examples_by_content_path)
        print(
            f"Wrote {len(examples_by_content)} examples by content to {examples_by_content_path}"
        )


def _write_examples(examples: List[object], file_path: Path) -> None:
    file_path.write_bytes(
        test_utils.gzip_compress(
            json.dumps(
                [asdict(example) for example in examples],
                separators=(",", ":"),
            ).encode("ascii")
        )
    )


@dataclass(frozen=True)
class CornerCaseInfo:
    label_category: LabelCategory
    with_threshold: bool
    with_overwrite: bool
    score_range: ScoreRange

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.label_category},"
            f"{'TH' if self.with_threshold else 'NO_TH'},"
            f"{'OW' if self.with_overwrite else 'NO_OW'},"
            f"{self.score_range})"
        )


class LabelCategory(enum.Enum):
    GENERIC_TEXT = enum.auto()
    GENERIC_BINARY = enum.auto()
    NON_GENERIC_TEXT = enum.auto()
    NON_GENERIC_BINARY = enum.auto()


class ScoreRange(enum.Enum):
    LT_050 = enum.auto()
    GE_050 = enum.auto()
    GE_050_LT_T = enum.auto()
    GE_T = enum.auto()


class CornerCaseCollector:
    def __init__(self, magika: Magika):
        self._magika = magika
        self._missing_corner_cases: Set[CornerCaseInfo] = set()
        # fmt: off
        self._missing_corner_cases.update({
            # NON_GENERIC_TEXT
            CornerCaseInfo(LabelCategory.NON_GENERIC_TEXT, False, False, ScoreRange.LT_050),
            CornerCaseInfo(LabelCategory.NON_GENERIC_TEXT, False, False, ScoreRange.GE_050),
            CornerCaseInfo(LabelCategory.NON_GENERIC_TEXT, True, False, ScoreRange.LT_050),
            CornerCaseInfo(LabelCategory.NON_GENERIC_TEXT, True, False, ScoreRange.GE_050_LT_T),
            CornerCaseInfo(LabelCategory.NON_GENERIC_TEXT, True, False, ScoreRange.GE_T),
            CornerCaseInfo(LabelCategory.NON_GENERIC_TEXT, False, True, ScoreRange.LT_050),
            CornerCaseInfo(LabelCategory.NON_GENERIC_TEXT, False, True, ScoreRange.GE_050),
            # NON_GENERIC_BINARY
            CornerCaseInfo(LabelCategory.NON_GENERIC_BINARY, False, False, ScoreRange.LT_050),
            CornerCaseInfo(LabelCategory.NON_GENERIC_BINARY, False, False, ScoreRange.GE_050),
            CornerCaseInfo(LabelCategory.NON_GENERIC_BINARY, True, False, ScoreRange.LT_050),
            CornerCaseInfo(LabelCategory.NON_GENERIC_BINARY, True, False, ScoreRange.GE_050_LT_T),
            CornerCaseInfo(LabelCategory.NON_GENERIC_BINARY, True, False, ScoreRange.GE_T),
            CornerCaseInfo(LabelCategory.NON_GENERIC_BINARY, False, True, ScoreRange.LT_050),
            CornerCaseInfo(LabelCategory.NON_GENERIC_BINARY, False, True, ScoreRange.GE_050),
        })
        self._missing_corner_cases.update({
            CornerCaseInfo(LabelCategory.GENERIC_TEXT, False, False, ScoreRange.LT_050),
            CornerCaseInfo(LabelCategory.GENERIC_TEXT, False, False, ScoreRange.GE_050),
            # No GENERIC_BINARY (aka UNKNOWN) because the model would never output that
        })
        # fmt: on

    def inspect_content(
        self, content: bytes
    ) -> Tuple[bool, MagikaResult, CornerCaseInfo]:
        res = self._magika.identify_bytes(content)
        cce = self._get_cornern_case_example(res.dl.label, res.score)
        if cce in self._missing_corner_cases:
            self._missing_corner_cases.remove(cce)
            return True, res, cce
        return False, res, cce

    def is_complete(self) -> bool:
        return self.get_missing_examples_num() == 0

    def get_missing_examples(self) -> Set[CornerCaseInfo]:
        return self._missing_corner_cases

    def get_missing_examples_num(self) -> int:
        return len(self._missing_corner_cases)

    def _get_cornern_case_example(
        self, dl_label: ContentTypeLabel, score: float
    ) -> CornerCaseInfo:
        return CornerCaseInfo(
            label_category=self._get_label_category(dl_label),
            with_threshold=self._has_threshold(dl_label),
            with_overwrite=self._has_overwrite(dl_label),
            score_range=self._get_score_range(dl_label, score),
        )

    def _get_label_category(self, dl_label: ContentTypeLabel) -> LabelCategory:
        m = {
            # is_generic, is_text
            (True, True): LabelCategory.GENERIC_TEXT,
            (True, False): LabelCategory.GENERIC_BINARY,
            (False, True): LabelCategory.NON_GENERIC_TEXT,
            (False, False): LabelCategory.NON_GENERIC_BINARY,
        }
        return m[
            self._is_generic(dl_label),
            self._is_text(dl_label),
        ]

    def _is_generic(self, dl_label: ContentTypeLabel) -> bool:
        return dl_label in [ContentTypeLabel.TXT, ContentTypeLabel.UNKNOWN]

    def _is_text(self, dl_label: ContentTypeLabel) -> bool:
        return self._magika._cts_infos[dl_label].is_text

    def _has_threshold(self, dl_label: ContentTypeLabel) -> bool:
        return dl_label in self._magika._model_config.thresholds.keys()

    def _get_threshold(self, dl_label: ContentTypeLabel) -> float:
        return self._magika._model_config.thresholds[dl_label]

    def _has_overwrite(self, dl_label: ContentTypeLabel) -> bool:
        return dl_label in self._magika._model_config.overwrite_map.keys()

    def _get_score_range(self, dl_label: ContentTypeLabel, score: float) -> ScoreRange:
        if score < 0.50:
            return ScoreRange.LT_050
        else:
            if self._has_threshold(dl_label):
                if score < self._get_threshold(dl_label):
                    return ScoreRange.GE_050_LT_T
                else:
                    return ScoreRange.GE_T
            else:
                return ScoreRange.GE_050

    def get_corner_case_candidates_generator(
        self,
    ) -> Generator[Tuple[str, bytes], None, None]:
        beg_size = self._magika._model_config.beg_size
        end_size = self._magika._model_config.end_size

        print("Using random bytes")
        for n in range(1_000):
            if random.random() < 0.5:
                yield (
                    "randomtxt",
                    test_utils.get_random_ascii_bytes(
                        random.randrange(8, beg_size + end_size)
                    ),
                )
            else:
                yield (
                    "randombytes",
                    test_utils.get_random_bytes(
                        random.randrange(8, beg_size + end_size)
                    ),
                )

        base_examples = []
        base_examples.append(
            ("randomtxt", test_utils.get_random_ascii_bytes(beg_size + end_size))
        )
        base_examples.append(
            ("randombytes", test_utils.get_random_bytes(beg_size + end_size))
        )
        for example_path in test_utils.get_basic_test_files_paths():
            example_content = example_path.read_bytes()
            if len(example_content) < beg_size + end_size:
                base_content = example_content
            else:
                base_content = b""
                if beg_size > 0:
                    example_content += base_content[:beg_size]
                if end_size > 0:
                    example_content += base_content[-end_size:]
            base_example = (str(example_path), base_content)
            yield base_example
            base_examples.append(base_example)

        for base_source, base_content in base_examples:
            print(f"Using {base_source} as base")
            for only_printable in [True, False]:
                for n in range(
                    0,
                    min(
                        beg_size,
                        end_size,
                        len(base_content),
                    ),
                ):
                    patched_content = bytearray(base_content[:])
                    patched_content[0:n] = test_utils.generate_pattern(
                        n, only_printable=only_printable
                    )
                    yield (f"base_{base_source}_beg_{n}", bytes(patched_content))

                    patched_content[len(base_content) - n : len(base_content)] = (
                        test_utils.generate_pattern(n, only_printable=only_printable)
                    )
                    yield (f"base_{base_source}_end_{n}", bytes(patched_content))


def _check_result_vs_reference_example(
    result: MagikaResult,
    expected_path: Path,
    expected_status: Status,
    expected_prediction: Prediction,
) -> None:
    assert result.path == expected_path
    assert result.status == expected_status
    if result.ok:
        assert result.prediction.dl.label == expected_prediction.dl
        assert result.prediction.output.label == expected_prediction.output
        assert result.prediction.score == pytest.approx(
            expected_prediction.score, abs=1e-5
        )
        assert (
            result.prediction.overwrite_reason == expected_prediction.overwrite_reason
        )


@dataclass
class ExampleByPath:
    """Data model for <model_name>-inference_examples_by_path.json.gz."""

    prediction_mode: PredictionMode
    path: str
    status: Status
    prediction: Optional[Prediction]


@dataclass
class ExampleByContent:
    """Data model for <model_name>-inference_examples_by_content.json.gz."""

    prediction_mode: PredictionMode
    content_base64: str
    status: Status
    prediction: Optional[Prediction]


@dataclass
class Prediction:
    dl: ContentTypeLabel
    output: ContentTypeLabel
    score: float
    overwrite_reason: OverwriteReason


if __name__ == "__main__":
    cli()

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
