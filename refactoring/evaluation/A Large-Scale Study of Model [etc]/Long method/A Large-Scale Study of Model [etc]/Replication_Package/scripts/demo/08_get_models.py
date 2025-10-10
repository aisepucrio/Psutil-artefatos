import psutil
import csv
import os
import ast
import pandas as pd
import multiprocessing
import tqdm
from treelib import Node, Tree
from codecarbon import EmissionsTracker

INPUT = 'scripts/demo/data/07_filtered_final.csv'
OUTPUT = 'scripts/demo/data/07_filtered_final.csv'
REPO_STORAGE = 'scripts/demo/data/cloned_repos'
MODEL_NAMES = 'scripts/demo/data/models'
MODEL_FILES = 'scripts/demo/data/model_files'
PROCESSES = 4


def safe_ast_parse(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as source:
            return ast.parse(source.read())
    except (SyntaxError, UnicodeDecodeError, FileNotFoundError, ValueError):
        return None


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


def extract_classes(repo_path):
    class_list = []
    class_files = {}
    for dirpath, dirnames, filenames in os.walk(repo_path):
        for f in filenames:
            if f.endswith('.py'):
                filepath = os.path.join(dirpath, f)
                tree = safe_ast_parse(filepath)
                if tree is None:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        if node.name in [i.name for i in class_list]:
                            continue
                        elif len(node.bases) > 0 and get_base(node.bases[0]) == node.name:
                            continue
                        else:
                            class_list.append(node)
                            class_files[node.name] = filepath
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


def identify_modules(class_tree, default_modules):
    module_nodes = [i.identifier for i in class_tree.filter_nodes(
        lambda node: any(i == node.tag for i in default_modules))]
    modules = []
    for module in module_nodes:
        modules += [i.tag for i in class_tree.subtree(module).all_nodes() if
                    i.identifier not in module_nodes]
    return modules


def save_model_files(repo, model_files):
    repo_path = os.path.join(REPO_STORAGE, repo['full_name'])
    path_head, repo_name = os.path.split(repo_path)
    path_rest, repo_owner = os.path.split(path_head)
    full_name = os.path.join(repo_owner, repo_name)
    class_files_path = os.path.join(MODEL_FILES, full_name)
    folder_path = os.path.join(MODEL_FILES, repo_owner)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    with open(class_files_path, 'w') as f:
        f.writelines(set(model_files))


def get_module_classes(repo, default_modules, save=True):
    repo_path = os.path.join(REPO_STORAGE, repo['full_name'])
    class_list, class_files = extract_classes(repo_path)
    class_names = [i.name for i in class_list]
    class_tree = build_class_tree(class_list, default_modules)
    modules = identify_modules(class_tree, default_modules)
    model_files = [v + '\n' for k, v in class_files.items() if k in modules]
    if save:
        save_model_files(repo, model_files)
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


def get_module_objects(repo, module_classes):
    object_list = []
    repo_path = os.path.join(REPO_STORAGE, repo['full_name'])
    for dirpath, dirnames, filenames in os.walk(repo_path):
        for f in filenames:
            if f.endswith('.py'):
                filepath = os.path.join(dirpath, f)
                tree = safe_ast_parse(filepath)
                if tree is None:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and check_call(node.func,
                                                                 module_classes):
                        object_list.append(get_id(node))
    return object_list


def process_repo(repo):
    with open('module_classes.txt') as f:
        default_modules = f.read().splitlines()
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
        row = process_repo((i, row))
        results.append(row)
    df_out = pd.DataFrame(results)
    print(df_out)


def parallel_run():
    df_in = pd.read_csv(INPUT)
    results = []
    pool = multiprocessing.Pool(processes=PROCESSES)
    for i in tqdm.tqdm(pool.imap_unordered(process_repo, df_in.iterrows()),
                       total=len(df_in)):
        results.append(i)
    df_out = pd.DataFrame(results)
    print(df_out)
    df_out.to_csv(OUTPUT, index=False)


def print_modules(project):
    data = pd.read_csv('data/tmp.csv')
    for i, row in data.iterrows():
        if row['full_name'].endswith(project):
            print(i, row['html_url'])
            modules, classes = get_module_classes(row)
            print(modules)


if __name__ == "__main__":
    tracker = EmissionsTracker()
    tracker.start()
    # single_run()
    parallel_run()
    # print_modules()
    tracker.stop()
