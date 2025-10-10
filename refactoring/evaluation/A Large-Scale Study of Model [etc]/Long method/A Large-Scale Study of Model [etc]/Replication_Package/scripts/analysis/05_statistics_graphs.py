import psutil
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
from codecarbon import EmissionsTracker

DATA_PATH = 'data/filtered_dataset.csv'
PROJECT_TYPES_PATH = 'data/'
RESULTS_PATH = 'results'
FIGSIZE = (10, 4)
WSPACE = 0.5
plt.rcParams.update({'font.size': 16})

if not os.path.exists("figures"):
    os.makedirs("figures")

tracker = EmissionsTracker()
tracker.start()
data = pd.read_csv(DATA_PATH)


def calculate_and_print_statistics(data, columns, labels):
    print('Min')
    for col in columns:
        print(np.min(data[col]))

    print('Max')
    for col in columns:
        print(np.max(data[col]))

    print("Average")
    for col in columns:
        print(np.average(data[col]))

    print("Median")
    for col in columns:
        print(np.median(data[col]))


def create_boxplot(ax, data, labels, yscale='linear'):
    ax.boxplot(data, showfliers=True)
    ax.set_yscale(yscale)
    ax.set_xticks([])


def configure_axis_labels(ax, labels):
    for i, label in enumerate(labels):
        ax[i].set_xlabel(label)


def general_boxplots():
    data = pd.read_csv(DATA_PATH)
    columns = ['stargazers_count', 'n_commits', 'py', 'py_lines', 'classes', 'functions_py']
    labels = ["# Stars", "# Commits", "# Source\nfiles", "# Lines\nof code", "# Classes", "# Functions"]
    data_arrays = [data.loc[:, col].to_numpy() for col in columns]

    calculate_and_print_statistics(data, columns, labels)

    fig, ax = plt.subplots(1, 6, figsize=(10, 3))
    plt.subplots_adjust(wspace=0.8)

    for i in range(6):
        create_boxplot(ax[i], [data_arrays[i]], [])

    configure_axis_labels(ax, labels)

    fig.savefig('figures/statistics.pdf', bbox_inches='tight', pad_inches=0.25)


def analyze_ml_libraries():
    library_stringlist = data.loc[:, 'ml_libraries'].to_list()
    libraries = []
    for l in library_stringlist:
        sublist = l[2:-2].split('\', \'')
        if sublist[0] == 't':
            libraries.append([])
        else:
            libraries.append(sublist)

    occurences = np.zeros(6, dtype=int)
    library_count = {
        "torch": 0,
        "sklearn": 0,
        "tensorflow": 0,
        "keras": 0,
    }
    for i in libraries:
        occurences[len(i)] += 1
        for l in library_count.keys():
            library_count[l] += l in i

    fig, ax = plt.subplots(figsize=(10, 2))

    lib_names = plt.barh(np.arange(1, 5), occurences[1:5], height=0.6)
    ax.bar_label(lib_names)
    plt.yticks(np.arange(1, 5))
    plt.ylabel("Number of\nML libraries")
    plt.xlabel("Number of projects")
    ax.margins(x=0.08)
    fig.tight_layout()
    plt.savefig('figures/results_library_number.pdf')
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 2))

    lib_counts = plt.barh(['PyTorch', 'Scikit-Learn', 'TensorFlow', 'Keras'], library_count.values(), height=0.6)
    ax.bar_label(lib_counts)
    plt.ylabel("Library")
    plt.xlabel("Number of projects")
    ax.margins(x=0.08)
    fig.tight_layout()
    plt.savefig('figures/results_library.pdf')
    plt.show()


def prepare_project_data():
    classification = pd.read_csv('data/classified_random_sample_160.csv')[:160]
    relations = {
        'Business': [],
        'Conceptual': [],
        'Tool': []
    }
    for i, r in classification.iterrows():
        for rel in relations:
            if rel in r['final relation']:
                relations[rel].append(r['html_url'])
    return relations


def analyze_ml_vs_source():
    full_data = pd.read_csv(DATA_PATH)
    relations = prepare_project_data()
    business = full_data[full_data['html_url'].isin(relations['Business'])]
    conceptual = full_data[full_data['html_url'].isin(relations['Conceptual'])]
    tool = full_data[full_data['html_url'].isin(relations['Tool'])]

    datasets = [full_data, business, conceptual, tool]
    x_labels = ['Full dataset', 'Business Focused\n(sample)', 'Conceptual\n(sample)', 'ML tools\n(sample)']
    fig, ax = plt.subplots(1, 4, figsize=(10, 3))
    plt.subplots_adjust(wspace=WSPACE)

    for i, data in enumerate(datasets):
        print(f"\n=============================\n{x_labels[i]}")
        ml_files = data.loc[:, 'ml_files_py'].to_numpy()
        source_files = data.loc[:, 'py'].to_numpy()
        functions = data.loc[:, 'functions_py'].to_numpy()
        ml_functions = data.loc[:, 'ml_functions_py'].to_numpy()
        model_files = data.loc[:, 'model_files'].to_numpy()

        file_relation = ml_files / source_files
        function_relation = ml_functions / functions
        with_model_files = np.nonzero(model_files)[0]
        avg_model_files = np.median(model_files[with_model_files])

        print(f"Median share of ml files\t{np.median(file_relation)}")
        print(f"Median share of ml functions\t{np.median(function_relation)}")
        print(f"Median number of model files:\t{np.median(model_files)}")
        print(f"Share of projects with model files:\t{len(with_model_files) / len(data)}")
        print(f"Median number of model files for projects with at least 1 model file:\t{avg_model_files}")

        print(f"Average share of ml files\t{np.average(file_relation)}")
        print(f"Average share of ml functions\t{np.average(function_relation)}")
        print(f"Average number of model files:\t{np.average(model_files)}")

        ax[i].boxplot([file_relation, function_relation], showfliers=True)
        ax[i].set_xticks([0, 1], ['# Files', '# Functions'], rotation=45)
        ax[i].set_xlabel(x_labels[i])
        ax[i].set_ylim([0, 1])

    ax[0].set_ylabel('Share of ML-related\nfiles/functions')
    plt.savefig("figures/results_ml_files.pdf", bbox_inches='tight')


def analyze_module_classes():
    classes = data.loc[:, 'classes'].to_numpy()
    module_classes = data.loc[:, 'module_classes'].to_numpy()
    module_instances = data.loc[:, 'module_objects'].to_numpy()
    model_files = data.loc[:, 'model_files'].to_numpy()

    with_modules = np.nonzero(module_classes)
    module_classes = module_classes[with_modules]
    module_instances = module_instances[with_modules]
    with_models_files = np.nonzero(model_files)
    model_files = model_files[with_models_files]

    module_class_share = np.nan_to_num(module_classes / classes[with_modules])
    instances_per_class = np.nan_to_num(module_instances / module_classes)

    print(f"Average number of classes:\t{np.average(classes)}")
    print(f"Average number of ml module classes:\t{np.average(module_classes)}")
    print(f"Average number of module instances:\t{np.average(module_instances)}")
    print(f"Ml modules / Classes:\t{np.average(module_class_share)}")
    print(f"Average number of instances per class:\t{np.average(instances_per_class)}")
    print(f'Projects with model files: {len(model_files)}')

    print(f"Median number of classes:\t{np.median(classes)}")
    print(f"Median number of ml module classes:\t{np.median(module_classes)}")
    print(f"Median number of module instances:\t{np.median(module_instances)}")
    print(f"Ml modules / Classes:\t{np.median(module_class_share)}")
    print(f"Median number of instances per class:\t{np.median(instances_per_class)}")
    print(f'Projects with model files: {len(model_files)}')

    fig, ax = plt.subplots(1, figsize=(10, 1.5))
    plt.subplots_adjust(wspace=0.5)
    ax.boxplot([classes, module_classes, model_files], vert=False)
    ax.set_xscale('log')
    ax.set_yticklabels(['# Classes', '# ML-module classes', '# Model binary files'])
    plt.savefig("figures/results_modules.pdf", bbox_inches='tight')


def analyze_test_coverage():
    function_coverage = data.loc[:, 'function_coverage'].to_numpy()
    ml_coverage = data.loc[:, 'ml_coverage'].to_numpy()
    with_test_files = np.nonzero(function_coverage)[0]
    function_coverage = function_coverage[with_test_files]
    ml_coverage = ml_coverage[with_test_files]

    avg_function_cov = np.average(function_coverage)
    avg_ml_cov = np.average(ml_coverage)
    med_function_cov = np.median(function_coverage)
    med_ml_cov = np.median(ml_coverage)

    print(f"Number of projects with test files:\t{len(with_test_files)}")
    print(f"Average test coverage for all functions:\t{avg_function_cov}")
    print(f"Average test coverage for ml functions:\t{avg_ml_cov}")
    print(f"Median test coverage for all functions:\t{med_function_cov}")
    print(f"Median test coverage for ml functions:\t{med_ml_cov}")

    fig, ax = plt.subplots(1, figsize=(10, 1))
    ax.boxplot([function_coverage, ml_coverage], showfliers=True, vert=False, widths=0.4)
    ax.set_yticklabels(['All functions', 'ML functions'])
    ax.set_xlabel('Test coverage')
    plt.savefig("figures/results_tests.pdf", bbox_inches='tight')


def analyze_model_reuse():
    only_hub_load = np.array(data.loc[:, 'hub_only_api'])
    files_with_hub = np.array(data.loc[:, 'files_with_hub'])

    print(f'Number of repositories with pretrained models [HUB LOAD]: {np.size(np.where(only_hub_load > 0))} of {np.size(only_hub_load)}')
    print(f'Average number HUb loads per system: {np.average(only_hub_load)}')

    fig, ax = plt.subplots(2, figsize=(10, 2.5), gridspec_kw={'height_ratios': [1, 3]})
    plt.subplots_adjust(wspace=0.5)
    fig.subplots_adjust(bottom=0.2)

    ax[0].boxplot([np.array(only_hub_load)[np.array(only_hub_load) > 0]], showfliers=True, vert=False)
    ax[0].set_xlabel('# API calls', labelpad=-5)
    ax[0].set_xscale('log')
    ax[0].set_yticklabels(['systems'])

    from_where = data.loc[:, 'where_load_hub']

    in_test_per_repo = np.array([ast.literal_eval(d)['test_count'] for d in from_where])
    in_example_per_repo = np.array([ast.literal_eval(d)['example_count'] for d in from_where])
    other_per_repo = np.array([ast.literal_eval(d)['other'] for d in from_where])

    ax[1].boxplot([
        other_per_repo[np.where(other_per_repo > 0)],
        in_test_per_repo[np.where(in_test_per_repo > 0)],
        in_example_per_repo[np.where(in_example_per_repo > 0)],
        files_with_hub[np.where(files_with_hub > 0)]
    ], showfliers=True, vert=False)

    ax[1].set_xscale('log')
    ax[1].set_yticklabels(['other', 'test dir', 'demo dir', 'total'])
    ax[1].set_xlabel('# files', labelpad=-10)

    fig.tight_layout(pad=0.20)
    plt.show()

    relative_test = in_test_per_repo[np.where(files_with_hub > 0)] / files_with_hub[np.where(files_with_hub > 0)] * 100
    relative_demo = in_example_per_repo[np.where(files_with_hub > 0)] / files_with_hub[np.where(files_with_hub > 0)] * 100
    relative_other = other_per_repo[np.where(files_with_hub > 0)] / files_with_hub[np.where(files_with_hub > 0)] * 100

    print(f"{np.size(in_example_per_repo[np.where(in_example_per_repo > 0)])} of systems using ptm load them in a demo directory.")
    print(f"In average {np.mean(relative_test)} % of ptm are loaded in test directories for those systems that load ptm.")
    print(f"{np.size(in_test_per_repo[np.where(in_test_per_repo > 0 )])} of systems using ptm load them in a test directory")
    print(f"In average {np.mean(relative_demo)} % of ptm are loaded in demo directories for those systems that load ptm.")
    print(f"{np.size(other_per_repo[np.where(other_per_repo > 0)])} of systems using ptm load them in other directories")
    print(f"In average {np.mean(relative_other)} % of ptm are loaded in other directories for those systems that load ptm.")

    print(f"Sanity Check: {relative_test + relative_other + relative_demo}")

    matrix = create_location_matrix(in_test_per_repo, in_example_per_repo, other_per_repo, files_with_hub)

    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap='viridis')

    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, f'{val:.0f}', ha='center', va='center', color='white')

    ax.set_xticklabels(['', 'demo and other', 'no demo and other', 'demo and no other', 'no demo and no other'])
    ax.set_yticklabels(['', 'Test: yes', 'Test: no'])
    plt.xticks(rotation=70)
    plt.tight_layout()
    fig.colorbar(cax)
    plt.show()


def create_location_matrix(in_test_per_repo, in_example_per_repo, other_per_repo, files_with_hub):
    matrix = np.zeros((2, 4))
    for test, demo, other in zip(in_test_per_repo[np.where(files_with_hub > 0)], in_example_per_repo[np.where(files_with_hub > 0)],
                                 other_per_repo[np.where(files_with_hub > 0)]):
        if test > 0:
            if demo > 0:
                if other > 0:
                    matrix[0, 0] += 1
                else:
                    matrix[0, 2] += 1
            else:
                if other > 0:
                    matrix[0, 1] += 1
                else:
                    matrix[0, 3] += 1
        else:
            if demo > 0:
                if other > 0:
                    matrix[1, 0] += 1
                else:
                    matrix[1, 2] += 1
            else:
                if other > 0:
                    matrix[1, 1] += 1
                else:
                    matrix[1, 3] += 1
    return matrix


def analyze_model_loading():
    load_api = np.array(data.loc[:, 'reuse_api_usage'])
    unsure_api = data.loc[:, 'unsure_api_usage']
    files_with_load = np.array(data.loc[:, 'files_with_load'])
    print(f'Number of repositories loading models: {np.size(np.where(load_api > 0))} of {np.size(load_api)}')
    print(f'Average number of unclassified APIs: {np.average(unsure_api)}')
    print(f'Average number of API calls: {np.average(np.array(load_api))}')

    fig, ax = plt.subplots(1, 5, figsize=FIGSIZE, sharey=True)
    plt.subplots_adjust(wspace=0.5)
    fig.subplots_adjust(bottom=0.2)

    create_boxplot(ax[0], [files_with_load[np.where(files_with_load > 0)]], [], yscale='log')
    ax[0].set_xticks([])
    ax[0].set_ylabel('Number of files in a system')
    ax[0].set_xlabel('Files\nwith load')

    create_boxplot(ax[1], [np.array(load_api)[load_api > 0]], [], yscale='log')
    ax[1].set_xlabel('in a\nsystem')
    ax[1].set_xticks([])

    from_where = data.loc[:, 'where_load']
    in_test_per_repo = np.array([ast.literal_eval(d)['test_count'] for d in from_where])
    in_example_per_repo = np.array([ast.literal_eval(d)['example_count'] for d in from_where])
    other_per_repo = np.array([ast.literal_eval(d)['other'] for d in from_where])

    create_boxplot(ax[2], [in_test_per_repo[np.where(in_test_per_repo > 0)]], [])
    ax[2].set_xticks([])
    ax[2].set_xlabel('in test\ndirectories')
    create_boxplot(ax[3], [in_example_per_repo[np.where(in_example_per_repo > 0)]], [])
    ax[3].set_xticks([])
    ax[3].set_xlabel('in demo\ndirectories')
    create_boxplot(ax[4], [other_per_repo[np.where(other_per_repo > 0)]], [])
    ax[4].set_xlabel('in other\ndirectories')
    ax[4].set_xticks([])

    ax[1].set_ylabel('Number of loads')

    fig.tight_layout(pad=0.20)
    plt.savefig("figures/results_all_loads.pdf")
    plt.show()


def analyze_project_types():
    data = pd.read_csv('data/classified_random_sample_160.csv')[:160]
    categories = {
        'Plugin': 0,
        'Framework': 0,
        'Library': 0,
        'Application': 0
    }
    relations = {
        'ML-Tool': 0,
        'Conceptual': 0,
        'Business Focused': 0
    }
    for i, r in data.iterrows():
        for c in categories:
            categories[c] += c in r['final type']
        for rel in relations:
            relations[rel] += rel in r['final relation']

    print(categories)
    print(relations)
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    cat = ax.barh(list(categories.keys()), categories.values(), height=0.6)
    rel = ax.barh(list(relations.keys()), relations.values(), height=0.6)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.bar_label(cat)
    ax.bar_label(rel)
    ax.set_xlabel('Number of Systems')
    fig.tight_layout()
    plt.savefig('figures/project_types.pdf')
    plt.show()


if __name__ == "__main__":
    general_boxplots()
    analyze_ml_libraries()
    analyze_ml_vs_source()
    analyze_module_classes()
    analyze_test_coverage()
    analyze_project_types()
    analyze_model_reuse()
    analyze_model_loading()

tracker.stop()
