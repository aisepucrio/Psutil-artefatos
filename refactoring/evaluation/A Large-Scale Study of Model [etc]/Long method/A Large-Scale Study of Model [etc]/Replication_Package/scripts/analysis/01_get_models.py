import psutil
import csv
import os
import ast
from treelib import Node, Tree
import multiprocessing
import tqdm
import pandas as pd
from codecarbon import EmissionsTracker

INPUT = 'data/filtered_dataset.csv'
OUTPUT = 'data/filtered_dataset.csv'
REPO_STORAGE = '<INSERT PATH HERE>'
MODEL_NAMES = '<INSERT PATH HERE>'
MODEL_FILES = '<INSERT PATH HERE>'
PROCESSES = 4


def get_base(node):
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return node.attr
    elif isinstance(node, ast.Subscript):
        return get_base(node.value)
    elif isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, str):
        return node
    elif isinstance(node, ast.Call):
        return get_base(node.func)
    elif isinstance(node, ast.IfExp):
        return get_base(node.body)
    elif isinstance(node, ast.BoolOp):
        return get_base(node.values[0])
    elif isinstance(node, ast.BinOp):
        print(ast.dump(node))
    else:
        return get_base(node.value)


def parse_file_for_classes(filepath):
    class_list = []
    class_files = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as source:
            tree = ast.parse(source.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_list.append(node)
                class_files[node.name] = filepath
    except (SyntaxError, UnicodeDecodeError, FileNotFoundError, ValueError):
        pass
    return class_list, class_files


def build_class_tree(class_list, default_modules):
    class_tree = Tree()
    class_tree.create_node('object', 'object')
    items_done = []

    for i in class_list:
        if len(i.bases) > 0:
            full_base = get_base(i.bases[0])
            if not any(full_base == j.name for j in class_list):
                if not class_tree.contains(full_base):
                    class_tree.create_node(full_base, full_base, parent='object')
                if not i.name == 'object':
                    try:
                        class_tree.create_node(i.name, i.name, parent=full_base)
                    except Exception:
                        pass
                    items_done.append(i)
    class_list = [i for i in class_list if i not in items_done]

    while len(class_list) > 0:
        items_done = []
        for i in class_list:
            if len(i.bases) == 0:
                base = 'object'
            else:
                base = get_base(i.bases[0])
            if class_tree.contains(i.name):
                items_done.append(i)
            elif class_tree.contains(base):
                class_tree.create_node(i.name, i.name, parent=base)
                items_done.append(i)
        if len(items_done) == 0:
            break
        class_list = [i for i in class_list if i not in items_done]
    return class_tree


def extract_modules_from_tree(class_tree, default_modules):
    module_nodes = [
        i.identifier for i in class_tree.filter_nodes(
            lambda node: any(i == node.tag for i in default_modules)
        )
    ]
    modules = []
    for module in module_nodes:
        modules += [
            i.tag for i in class_tree.subtree(module).all_nodes()
            if i.identifier not in module_nodes
        ]
    return modules


def save_model_files(repo_path, class_files, modules):
    model_files = []
    for k, v in class_files.items():
        if k in modules:
            model_files.append(v + '\n')

    path_head, repo_name = os.path.split(repo_path)
    path_rest, repo_owner = os.path.split(path_head)
    full_name = os.path.join(repo_owner, repo_name)
    class_files_path = os.path.join(MODEL_FILES, full_name)
    folder_path = os.path.join(MODEL_FILES, repo_owner)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    with open(class_files_path, 'w') as f:
        f.writelines(set(model_files))


def get_module_classes(repo_path, default_modules, save=True):
    class_list = []
    class_files = {}
    for dirpath, dirnames, filenames in os.walk(repo_path):
        for f in filenames:
            if f.endswith('.py'):
                filepath = os.path.join(dirpath, f)
                classes, files = parse_file_for_classes(filepath)
                class_list.extend(classes)
                class_files.update(files)

    class_names = [i.name for i in class_list]
    class_tree = build_class_tree(class_list, default_modules)
    modules = extract_modules_from_tree(class_tree, default_modules)

    if save:
        save_model_files(repo_path, class_files, modules)

    return modules, class_names


def check_call(node, module_classes):
    if isinstance(node, ast.Attribute):
        return node.attr in module_classes
    elif isinstance(node, ast.Call):
        return check_call(node.func, module_classes)
    elif isinstance(node, ast.Subscript):
        return check_call(node.value, module_classes)
    elif isinstance(node, ast.Name):
        return node.id in module_classes
    else:
        return False


def get_id(node):
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return get_id(node.value)
    elif isinstance(node, ast.Call):
        return get_id(node.func)
    elif isinstance(node, ast.Subscript):
        return get_id(node.value)
    elif isinstance(node, ast.IfExp):
        return get_id(node.body)
    else:
        print(ast.dump(node))
        return False


def get_module_objects(repo_path, module_classes):
    object_list = []
    for dirpath, dirnames, filenames in os.walk(repo_path):
        for f in filenames:
            if f.endswith('.py'):
                filepath = os.path.join(dirpath, f)
                try:
                    with open(filepath, 'r', encoding='utf-8') as source:
                        tree = ast.parse(source.read())
                except (SyntaxError, UnicodeDecodeError, FileNotFoundError, ValueError):
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and check_call(node.func, module_classes):
                        object_list.append(get_id(node))
    return object_list


def process_repo(repo_item, default_modules):
    repo = repo_item[1]
    module_classes, classes = get_module_classes(repo, default_modules)
    filepath = os.path.join(MODEL_NAMES, repo['full_name'])
    folder_path = os.path.split(filepath)[0]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(filepath, 'w') as f:
        f.writelines([i + '\n' for i in module_classes])
    repo['classes'] = len(classes)
    module_objects = get_module_objects(repo, module_classes + default_modules)
    repo['module_classes'] = len(module_classes)
    repo['module_objects'] = len(module_objects)
    return repo


def single_run():
    df_in = pd.read_csv(INPUT)[:10]
    results = []
    for i, row in df_in.iterrows():
        print(i, row['html_url'])
        with open('data/module_classes.txt') as f:
            default_modules = f.read().splitlines()
        row = process_repo((i, row), default_modules)
        results.append(row)
    df_out = pd.DataFrame(results)
    print(df_out)


def parallel_run():
    df_in = pd.read_csv(INPUT)
    results = []
    with open('data/module_classes.txt') as f:
        default_modules = f.read().splitlines()
    pool = multiprocessing.Pool(processes=PROCESSES)
    for i in tqdm.tqdm(
            pool.imap_unordered(
                lambda item: process_repo(item, default_modules),
                df_in.iterrows(),
            ),
            total=len(df_in),
    ):
        results.append(i)
    df_out = pd.DataFrame(results)
    print(df_out)


def print_modules(project):
    data = pd.read_csv('data/tmp.csv')
    for i, row in data.iterrows():
        if row['full_name'].endswith(project):
            print(i, row['html_url'])
            with open('data/module_classes.txt') as f:
                default_modules = f.read().splitlines()
            modules, classes = get_module_classes(row, default_modules)
            print(modules)


if __name__ == "__main__":
    tracker = EmissionsTracker(
        project_name=r"C:\Users\guicu\OneDrive\Documentos\prog\aise\artifact\artifacts\A Large-Scale Study of Model Integration in ML-Enabled Software Systems\Replication_Package\scripts\analysis\01_get_models.py"
    )
    tracker.start()
    # mem_start = psutil.virtual_memory().used / (1024**2)
    # cpu_start = psutil.cpu_percent(interval=None)
    # tracker = EmissionsTracker()
    # tracker.start()

    # single_run()
    parallel_run()
    # print_modules()
    tracker.stop()
