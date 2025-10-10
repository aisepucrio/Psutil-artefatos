import psutil
import csv
import os
import pandas as pd
import ast
import astunparse
import json
import multiprocessing
import tqdm
from codecarbon import EmissionsTracker

INPUT = 'scripts/demo/data/07_filtered_final.csv'
OUTPUT = 'scripts/demo/data/07_filtered_final.csv'
REPO_STORAGE = 'scripts/demo/data/cloned_repos'
REUSE_STORAGE = 'scripts/demo/data//reuse_from_where/ast_dump/'
DICT_STORAGE = 'scripts/demo/data/reuse_from_where/dictionaries/'
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

    def visit_Call(self, node):
        if self._is_model_load_call(node):
            if self._is_valid_load_target(node):
                hub = self._get_hub_from_call(node)
                if hub:
                    self.results_dict[hub] += 1
                    self.result += 1
                    self._dump_sub_tree(node)
                else:
                    self.unsure += 1
                    if self._prompt_user(node):
                        self.result += 1
                        self._dump_sub_tree(node)

        ast.NodeVisitor.generic_visit(self, node)

    def visit_Import(self, node):
        self._handle_import_alias(node)
        ast.NodeVisitor.generic_visit(self, node)

    def visit_ImportFrom(self, node):
        self._handle_import_from(node)
        ast.NodeVisitor.generic_visit(self, node)

    def _is_model_load_call(self, node):
        return isinstance(node.func, ast.Attribute) and node.func.attr in INIT_API_CALLS

    def _is_valid_load_target(self, node):
        return (isinstance(node.func.value, ast.Name) and not node.func.value.id in IGNORE) or (
                isinstance(node.func.value, ast.Attribute) and not node.func.value.attr in IGNORE)

    def _get_hub_from_call(self, node):
        for hub, libs in HUB_MAPPING.items():
            if self._is_hub_match(node, libs):
                return hub
        return None

    def _is_hub_match(self, node, libs):
        if isinstance(node.func.value, ast.Name) and node.func.value.id in libs:
            return True
        if (isinstance(node.func.value, ast.Attribute) and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id in libs):
            return True
        return False

    def _prompt_user(self, node):
        print(f"In {self._file}")
        print("Following call was found and not categorized automatically:")
        print(ast.unparse(node))

        response = input("Do you want to add this call to the results? (Y/n)")
        if response.lower() in ['yes', 'y']:
            return self._handle_user_category_input(node)
        elif response.lower() in ['no', 'n']:
            self._handle_ignore_call(node)
        return False

    def _handle_user_category_input(self, node):
        print("Known categories:")
        print(HUB_MAPPING)
        hub_answer = input("To which categories do you want to add the call? Type the name of a known or unknown category.")
        if hub_answer in HUB_MAPPING:
            self._update_hub_mapping(hub_answer, node)
            return True
        else:
            if self._ask_to_create_new_category(hub_answer, node):
                return True
        return False

    def _ask_to_create_new_category(self, hub_answer, node):
        new_response = input(
            "Could not match your input to one of the known categories. Do you want to add a new category? (Y/n)")
        if new_response.lower() in ['yes', 'y']:
            HUB_MAPPING[hub_answer] = [node.func.value.id] if isinstance(node.func.value, ast.Name) else [
                node.func.value.attr]
            self.results_dict[hub_answer] += 1
            return True
        return False

    def _update_hub_mapping(self, hub_answer, node):
        if isinstance(node.func.value, ast.Name):
            HUB_MAPPING[hub_answer].append(node.func.value.id)
        elif isinstance(node.func.value, ast.Attribute):
            HUB_MAPPING[hub_answer].append(node.func.value.attr)
        HUB_MAPPING[hub_answer] = list(set(HUB_MAPPING[hub_answer]))
        self.results_dict[hub_answer] += 1

    def _handle_ignore_call(self, node):
        ignore_response = input("Do you want to ignore this call? (y/N)")
        if ignore_response.lower() in ['yes', 'y']:
            if isinstance(node.func.value, ast.Name):
                IGNORE.add(node.func.value.id)
            elif isinstance(node.func.value, ast.Attribute):
                IGNORE.add(node.func.value.attr)

    def _handle_import_alias(self, node):
        for name in node.names:
            if isinstance(name, ast.alias) and name.asname is not None:
                for hub in HUB_MAPPING.keys():
                    if hub == name.name:
                        HUB_MAPPING[hub].append(name.asname)
                        HUB_MAPPING[hub] = list(set(HUB_MAPPING[hub]))

    def _handle_import_from(self, node):
        if node.module == 'transformers':
            for name in node.names:
                if isinstance(name, ast.alias):
                    HUB_MAPPING['huggingface'].append(name.name)
                    HUB_MAPPING['huggingface'] = list(set(HUB_MAPPING['huggingface']))

    def _dump_sub_tree(self, node):
        if not hasattr(self, '_is_first'):
            self._is_first = True

        if self._is_first:
            self._is_first = False
            os.makedirs(self._folder, exist_ok=True)

        with open(self._file_path, 'a') as out:
            out.write(astunparse.unparse(node) + '\n')


def has_reuse_api(file):
    skipped = False
    result = 0
    unsure = 0
    hub_count = {
        'torch': 0,
        'tensorflow': 0,
        'huggingface': 0,
        'modelhub': 0,
        'onnx': 0,
        'unknown model': 0
    }

    try:
        with open(file, 'r', encoding='utf-8') as source:
            try:
                tree = ast.parse(source.read())
                visitor = LoadVisitor(file)
                visitor.visit(tree)
                for hub in visitor.results_dict:
                    hub_count[hub] += visitor.results_dict.get(hub)
                result += visitor.result
                unsure += visitor.unsure
            except Exception as e:
                skipped = True
                pass
    except Exception as e:
        skipped = True
        pass
    return skipped, result, unsure, hub_count


def process_repo(repo):
    repo_item = repo[1]
    repo_path = os.path.join(REPO_STORAGE, repo_item['full_name'])

    if os.path.isdir(os.path.join(REUSE_STORAGE, *repo_item['full_name'].split('/')[-2:])):
        return repo_item

    repo_item['pre_trained'] = False
    result = 0
    unsure_per_repo = 0
    total_hub_count = {hub: 0 for hub in HUB_MAPPING}
    skipped_in_repo = 0

    for dirpath, dirnames, filenames in os.walk(repo_path):
        for f in filenames:
            if f.endswith('.py'):
                filepath = os.path.join(dirpath, f)
                was_skipped, result_per_file, unsure_per_file, hub_count = has_reuse_api(filepath)
                if result_per_file > 0:
                    repo_item['pre_trained'] = True
                    result += result_per_file
                    unsure_per_repo += unsure_per_file
                    for hub, count in hub_count.items():
                        total_hub_count[hub] += count
                if was_skipped:
                    skipped_in_repo += 1

    repo_item['number_loads'] = result
    repo_item['load_from_where'] = total_hub_count

    return repo_item


def save_metadata():
    os.makedirs(DICT_STORAGE, exist_ok=True)
    with open(os.path.join(DICT_STORAGE, 'mapping.txt'), 'w', encoding='utf-8') as dict_file:
        json.dump(HUB_MAPPING, dict_file)
    with open(os.path.join(DICT_STORAGE, "ignore-list.txt"), 'w', encoding='utf-8') as ignore_list:
        json.dump(list(IGNORE), ignore_list)


def single_run():
    data = pd.read_csv(INPUT).iloc[1:10]
    results_list = [process_repo((i, repo)) for i, repo in data.iterrows()]
    results_df = pd.DataFrame(results_list)
    print(results_df)
    # results_df.to_csv(OUTPUT,index=False)


def non_parallel():
    data = pd.read_csv(INPUT)
    results_list = [process_repo((i, repo)) for i, repo in data.iterrows()]
    results_df = pd.DataFrame(results_list)
    results_df.set_index('full_name', inplace=True)
    data.set_index('full_name', inplace=True)
    data.update(results_df)
    data = data.reset_index()
    data.to_csv(OUTPUT, index=False)


def parallel_run():
    data = pd.read_csv(INPUT)
    results_list = []
    with multiprocessing.Pool(processes=N_PROCESSES) as pool:
        for result in tqdm.tqdm(pool.imap_unordered(process_repo, data.iterrows()), total=len(data)):
            results_list.append(result)
    results_df = pd.DataFrame(results_list)
    print(results_df)
    results_df.to_csv(OUTPUT, index=False)


if __name__ == "__main__":
    tracker = EmissionsTracker()
    tracker.start()
    single_run()
    # non_parallel()
    # parallel_run()
    save_metadata()
    tracker.stop()
