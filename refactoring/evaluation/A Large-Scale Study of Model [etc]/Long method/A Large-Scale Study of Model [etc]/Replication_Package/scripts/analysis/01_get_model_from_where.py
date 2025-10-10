import psutil
import csv
import os
import ast
import astunparse
import json
import pandas as pd
import multiprocessing
import tqdm
from codecarbon import EmissionsTracker

INPUT = 'data/filtered_dataset.csv'
OUTPUT = 'data/filtered_dataset.csv'
REPO_STORAGE = '<INSERT PATH HERE>'
REUSE_STORAGE = '<INSERT PATH HERE>'
DICT_STORAGE = '<INSERT PATH HERE>'
N_PROCESSES = 4

INIT_API_CALLS = ['load', 'hf_hub_download',
                  'load_composite_model', 'download_model_with_test_data', 'from_pretrained', 'load_state_dict',
                  'load_url', 'load_state_dict_from_url']

IGNORE = {'np', 'numpy', 'pd', 'pandas', 'json', 'pickle', 'pkl'}

HUB_MAPPING = {
    'torch': ['torch'],
    'tensorflow': ['tensorflow', 'tf', 'keras'],
    'huggingface': ['huggingface', 'transformers'],
    'modelhub': ['modelhub'],
    'model_zoo': ['model_zoo'],
    'onnx': ['onnx'],
    'unknown model': []
}


class LoadVisitor(ast.NodeVisitor):
    def __init__(self, file):
        file_name = file.replace('\\', '/').split('/')[7:]
        self._repo_folder = os.path.join(REUSE_STORAGE, os.path.join(*file_name[0:2]))
        self._folder = os.path.join(REUSE_STORAGE, os.path.join(*file_name[:-1]))
        self._file_path = os.path.join(self._folder, file_name[-1].split('.')[0]) + '.txt'
        os.makedirs(self._repo_folder, exist_ok=True)

        self._file = file
        self.result = 0
        self.unsure = 0
        self.results_dict = {
            'torch': 0,
            'tensorflow': 0,
            'huggingface': 0,
            'modelhub': 0,
            'onnx': 0,
            'unknown model': 0
        }
        self._is_first = True

    def visit_Call(self, node):
        if self._is_model_load_api_call(node):
            hub_found = self._handle_hub_mapping(node)
            if not hub_found:
                hub_found = self.prompt(node)

            if hub_found:
                self.result += 1
                self._dump_sub_tree(node)

        ast.NodeVisitor.generic_visit(self, node)

    def _is_model_load_api_call(self, node):
        return (isinstance(node.func, ast.Attribute) and node.func.attr in INIT_API_CALLS and
                ((isinstance(node.func.value, ast.Name) and not node.func.value.id in IGNORE) or
                 (isinstance(node.func.value, ast.Attribute) and not node.func.value.attr in IGNORE)))

    def _handle_hub_mapping(self, node):
        for hub in HUB_MAPPING.keys():
            if self._is_in_hub_mapping(node, hub):
                self.results_dict[hub] += 1
                return True
        return False

    def _is_in_hub_mapping(self, node, hub):
        if isinstance(node.func.value, ast.Name) and node.func.value.id in HUB_MAPPING[hub]:
            return True
        elif (isinstance(node.func.value, ast.Attribute) and isinstance(node.func.value.value, ast.Name)
              and node.func.value.value.id in HUB_MAPPING[hub]):
            return True
        return False

    def visit_Import(self, node):
        for name in node.names:
            if isinstance(name, ast.alias) and name.asname is not None:
                self._update_hub_mapping(name.name, name.asname)
        ast.NodeVisitor.generic_visit(self, node)

    def _update_hub_mapping(self, original_name, alias):
        for hub in HUB_MAPPING.keys():
            if hub == original_name:
                HUB_MAPPING[hub].append(alias)
                HUB_MAPPING[hub] = list(set(HUB_MAPPING[hub]))

    def visit_ImportFrom(self, node):
        if node.module == 'transformers':
            for name in node.names:
                if isinstance(name, ast.alias):
                    self._add_to_huggingface_mapping(name.name)
        ast.NodeVisitor.generic_visit(self, node)

    def _add_to_huggingface_mapping(self, name):
        HUB_MAPPING['huggingface'].append(name)
        HUB_MAPPING['huggingface'] = list(set(HUB_MAPPING['huggingface']))

    def _dump_sub_tree(self, node):
        if self._is_first:
            self._is_first = False
            os.makedirs(self._folder, exist_ok=True)

        with open(self._file_path, 'a') as out:
            out.write(astunparse.unparse(node) + '\n')

    def prompt(self, node):
        self._print_prompt_info(node)
        response = input("Do you want to add this call to the results? (Y/n)")

        if response in ['yes', 'Yes', 'y', 'Y']:
            return self._handle_add_to_results(node)
        elif response in ['no', 'No', 'n', 'N']:
            return self._handle_ignore_call(node)
        else:
            return self.prompt(node)

    def _print_prompt_info(self, node):
        print(f"In {self._file}")
        print("Following call was found and not categorized automatically:")
        print(ast.unparse(node))
        print("Known categories:")
        print(HUB_MAPPING)

    def _handle_add_to_results(self, node):
        hub_answer = input("To which categories do you want to add the call? Type the name of a known or unknown category.")
        if hub_answer in HUB_MAPPING.keys():
            HUB_MAPPING[hub_answer].append(node.name)
            self.results_dict[hub_answer] += 1
            return True
        else:
            return self._handle_new_category(hub_answer, node)

    def _handle_new_category(self, hub_answer, node):
        new_response = input("Could not match your input to one of the known categories. Do you want to add a new category? (Y/n)")
        if new_response in ['yes', 'Yes', 'y', 'Y']:
            HUB_MAPPING[hub_answer] = [node.name]
            self.results_dict[hub_answer] += 1
            return True
        else:
            return False

    def _handle_ignore_call(self, node):
        ignore_response = input("Do you want to ignore this call? (y/N)")
        if ignore_response in ['yes', 'Yes', 'y', 'Y']:
            if isinstance(node.func.value, ast.Name):
                IGNORE.add(node.func.value.id)
            elif isinstance(node.func.value, ast.Attribute):
                IGNORE.add(node.func.value.attr)
        return False


def analyze_file(file_path):
    skipped = False
    result = 0
    unsure = 0
    hub_count = {hub: 0 for hub in HUB_MAPPING}

    try:
        with open(file_path, 'r', encoding='utf-8') as source:
            try:
                tree = ast.parse(source.read())
                visitor = LoadVisitor(file_path)
                visitor.visit(tree)

                for hub, count in visitor.results_dict.items():
                    hub_count[hub] += count
                result += visitor.result
                unsure += visitor.unsure

            except Exception:
                skipped = True
                pass

    except Exception:
        skipped = True
        pass

    return skipped, result, unsure, hub_count


def process_repo(item):
    repo = item[1]
    repo_path = os.path.join(REPO_STORAGE, repo['full_name'])

    reuse_path = os.path.join(REUSE_STORAGE, *repo_path.split('/')[-2:])
    if os.path.isdir(reuse_path):
        return repo

    repo['pre_trained'] = False
    files = []
    skipped_in_repo = 0
    result = 0
    unsure_per_repo = 0
    total_hub_count = {hub: 0 for hub in HUB_MAPPING}

    for dirpath, dirnames, filenames in os.walk(repo_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if filepath.endswith('.py'):
                was_skipped, result_per_file, unsure_per_file, hub_count = analyze_file(filepath)
                if result_per_file > 0:
                    repo['pre_trained'] = True
                    result += result_per_file
                    unsure_per_repo += unsure_per_file
                    for hub, count in hub_count.items():
                        total_hub_count[hub] += count
                    files.append(filename)
                if was_skipped:
                    skipped_in_repo += 1

    repo['number_loads'] = result
    repo['load_from_where'] = total_hub_count

    _save_mappings()

    return repo


def _save_mappings():
    os.makedirs(DICT_STORAGE, exist_ok=True)
    with open(os.path.join(DICT_STORAGE, 'mapping.txt'), 'w', encoding='utf-8') as f:
        json.dump(HUB_MAPPING, f)
    with open(os.path.join(DICT_STORAGE, "ignore-list.txt"), 'w', encoding='utf-8') as f:
        json.dump(list(IGNORE), f)


def run_single(data):
    results_list = []
    for i, repo in data.iterrows():
        print(f'{i}/{len(data)}')
        result = process_repo((i, repo))
        results_list.append(result)

    results_df = pd.DataFrame(results_list)
    print(results_df)


def run_non_parallel(data):
    results_list = []
    for i, repo in data.iterrows():
        print(f'{i}/{len(data)}')
        result = process_repo((i, repo))
        results_list.append(result)

        results_df = pd.DataFrame(results_list)
        results_df.set_index('full_name', inplace=True)
        data.set_index('full_name', inplace=True)
        data.update(results_df)
        data = data.reset_index()
        data.to_csv(OUTPUT, index=False)


def run_parallel(data):
    results_list = []
    with multiprocessing.Pool(processes=N_PROCESSES) as pool:
        for i in tqdm.tqdm(pool.imap_unordered(process_repo, data.iterrows()), total=len(data)):
            results_list.append(i)

    results_df = pd.DataFrame(results_list)
    print(results_df)
    results_df.to_csv(OUTPUT, index=False)


if __name__ == "__main__":
    tracker = EmissionsTracker()
    tracker.start()

    data = pd.read_csv(INPUT)

    # run_single(data.iloc[1:10])
    # run_non_parallel(data)
    run_parallel(data)

    tracker.stop()
