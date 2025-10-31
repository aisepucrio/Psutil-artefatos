from codecarbon import EmissionsTracker
import psutil
import csv
import os
import hashlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathvalidate import sanitize_filename
from scipy.ndimage import gaussian_filter1d
import copy
import logging
from diskcache import Index
from pathlib import Path

abc = 0
full_abc = ['a', 'b', 'c', 'd', 'e', 'f',
            'g', 'h', 'i', 'j', 'k', 'l', 'm']

def _remove_individual_legends_and_add_common_legend(all_axes, fig, bbox_to_anchor=(0.5, 0.3)):
    """
    Remove individual legends from all axes and add a common legend to the figure.
    """
    for ax in all_axes.flatten():
        try:
            ax.get_legend().remove()
        except:
            pass

    handles, labels = all_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=bbox_to_anchor,
               handletextpad=0.01, columnspacing=0.1, borderpad=0.1, labelspacing=0.1)

def _get_abbreviated_name(name):
    """
    Map a model or dataset name to its abbreviated paper name.
    """
    name_map = {
        'openai-community/openai-gpt': 'GPT',
        'openai-communityopenai-gpt': 'GPT',
        'google-bert/bert-base-cased': 'BERT',
        'resnet18': 'ResNet',
        'densenet121': 'DenseNet',
        'dbpedia_14': 'DBpedia',
        'yahoo_answers_topics': 'Yahoo-Answers',
        'mnist': 'MNIST',
        'cifar10': 'CIFAR10',
        'pathmnist': 'Colon-Pathology',
        'organamnist': 'Abdominal-CT',
        'PathologicalPartitioner-3': 'Pathological',
        'non_iid_dirichlet': 'Dirichlet'
    }
    return name_map.get(name, name)

def _smooth_data(column_values, sigma=2):
    """
    Apply a Gaussian filter to smooth a sequence of values.
    """
    return gaussian_filter1d(column_values, sigma=sigma)

def _generate_hashed_name(name, algorithm='md5'):
    """
    Compute the hash of a given string using a specified algorithm.
    """
    hash_algorithms = {
        'md5': hashlib.md5,
        'sha1': hashlib.sha1,
        'sha256': hashlib.sha256
    }
    if algorithm not in hash_algorithms:
        raise ValueError(
            "Unsupported algorithm. Use 'md5', 'sha1', or 'sha256'.")
    hash_object = hash_algorithms[algorithm](name.encode())
    return hash_object.hexdigest()

def _convert_cache_to_csv(cache):
    """
    Convert cached provenance results to CSV files and log the file paths.
    """
    csv_paths = []
    for key in cache.keys():
        round2prov_result = cache[key]["round2prov_result"]
        prov_cfg = cache[key]["prov_cfg"]
        avg_prov_time_per_round = cache[key]["avg_prov_time_per_round"]
        each_round_prov_result = []

        for r2prov in round2prov_result:
            r2prov['training_cache_path'] = cache[key]["training_cache_path"]
            r2prov['avg_prov_time_per_round'] = avg_prov_time_per_round
            r2prov['prov_cfg'] = prov_cfg
            r2prov['Model'] = prov_cfg.model.name
            r2prov['Dataset'] = prov_cfg.dataset.name
            r2prov['Num Clients'] = prov_cfg.num_clients
            r2prov['Dirichlet Alpha'] = prov_cfg.dirichlet_alpha

            if 'Error' in r2prov:
                continue

            for m, v in r2prov['eval_metrics'].items():
                r2prov[m] = v
            each_round_prov_result.append(copy.deepcopy(r2prov))

        df = pd.DataFrame(each_round_prov_result)
        key = sanitize_filename(key)
        csv_path = f"results_csvs/prov_{key}.csv"

        if len(csv_path) > 250:
            logging.warn(f"CSV path too long, using hashed name: {csv_path}")
            hashed_name = _generate_hashed_name(csv_path)
            csv_path = f"results_csvs/{hashed_name}.csv"
            logging.warn(f"hashed name: {csv_path}")

        csv_paths.append(csv_path)
        df.to_csv(csv_path)

    with open('csv_paths.log', 'w') as f:
        for path in sorted(csv_paths):
            f.write(f"{path}\n")

def _plot_line(axis, x, y, label, linestyle, linewidth=2):
    """
    Plot a line on a given axis.
    """
    axis.plot(x, y, label=label, linestyle=linestyle, linewidth=linewidth)

def _save_figure(fig, filename_base, file_formats=['png', 'svg', 'pdf'], dpi=600):
    """
    Save a figure in multiple file formats.
    """
    for ext in file_formats:
        directory_path = Path(f"graphs/{ext}s")
        directory_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"graphs/{ext}s/{filename_base}.{ext}", bbox_inches='tight', format=ext, dpi=dpi)
    plt.close('all')

def _create_subplots(width_inches=3.3374, height_inches=3.3374/1.618, nrows=1, ncols=1, **kwargs):
    """
    Create a figure and axes for plotting.
    """
    if nrows != -1 and ncols != -1:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width_inches, height_inches), **kwargs)
    else:
        fig, axes = plt.subplots(figsize=(width_inches, height_inches), **kwargs)
    return fig, axes

def _plot_label_distribution_motivation_helper():
    """Helper function to plot label distribution."""
    data = {
        'Labels': ['Adipose', 'Background', 'Debris', 'Lymphocytes', 'Mucus', 'Smooth Muscle',
                   'Normal Colon Mucosa', 'Cancer-associated Stroma', 'Colorectal Adenocarcinoma'],
        'H0': [670, 667, 711, 0, 0, 0, 0, 0, 0],
        'H1': [0, 568, 623, 857, 0, 0, 0, 0, 0],
        'H2': [0, 0, 602, 807, 639, 0, 0, 0, 0],
        'H3': [0, 0, 0, 737, 508, 803, 0, 0, 0],
        'H4': [0, 0, 0, 0, 587, 846, 615, 0, 0],
        'H5': [0, 0, 0, 0, 0, 891, 533, 624, 0],
        'H6': [0, 0, 0, 0, 0, 0, 528, 635, 885],
        'H7': [516, 0, 0, 0, 0, 0, 0, 611, 921],
        'H8': [540, 521, 0, 0, 0, 0, 0, 0, 987],
        'H9': [672, 666, 710, 0, 0, 0, 0, 0, 0]
    }
    df = pd.DataFrame(data)
    df_melted = df.melt(id_vars='Labels', var_name='Partition ID', value_name='Count')
    df_melted_pivot = df_melted.pivot_table(index='Partition ID', columns='Labels', values='Count', aggfunc='sum', fill_value=0)
    return df_melted_pivot

def _plot_label_distribution_motivation():
    """
    Plot the label distribution for motivation purposes.
    """
    df_melted_pivot = _plot_label_distribution_motivation_helper()
    fig, axes = _create_subplots(width_inches=3.3374*4.6, height_inches=(3.3374*2)/1.618, nrows=1, ncols=1)
    df_melted_pivot.plot(kind='bar', stacked=True, ax=axes)
    plt.title('Per Hospital Labels Distribution')
    plt.xlabel('Hospital ID')
    plt.ylabel('Number of Data Points')
    plt.legend(title='Labels', bbox_to_anchor=(0.5, 1.15), loc='center', ncol=3, frameon=False)
    plt.tight_layout()
    _save_figure(fig, "label_distribution_proper_labels_without_flip_pathmnist")

def _get_classification_df(mname, dname, alpha, trounds):
    """Helper function to read the classification results CSV."""
    fname = f'results_csvs/prov_image_classification_exp-{mname}-{dname}-faulty_clients[[]]-noise_rateNone-TClients100-fedavg-(R{trounds}-clientsPerR10)-non_iid_dirichlet{alpha}-batch32-epochs2-lr0.001.csv'
    return pd.read_csv(fname)

def _plot_single_classification(ax, mname, dname, alpha, trounds):
    """Helper function to plot a single classification result."""
    df = _get_classification_df(mname, dname, alpha, trounds)
    model = _get_abbreviated_name(df['Model'][0])
    dataset = _get_abbreviated_name(df['Dataset'][0])

    df['Accuracy'] = df['Accuracy'] * 100
    df['test_data_acc'] = df['test_data_acc'] * 100

    _plot_line(ax, range(len(df)), _smooth_data(
        df['Accuracy']), label='TraceFL-Smooth', linestyle='-')
    _plot_line(ax, range(len(df)), _smooth_data(
        df['test_data_acc']), label='FL Training-Smooth', linestyle='--')
    
    temp_dict = {'Average Accuracy': df['Accuracy'].mean(
    ), 'Total Rounds': trounds, 'Total Accuracy': sum(df['Accuracy'])}

    global abc
    title = (f"{full_abc[abc]}) {dataset}\n{model}")
    abc += 1
    ax.text(0.5, 0.5, f"TraceFL \n Avg. Acc {temp_dict['Average Accuracy']:.1f}",
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize=9)
    ax.set_title(title)
    ax.legend()
    return temp_dict

def _plot_model_dataset_configs(axes, mnames, dnames, num_rounds, alpha, all_config_summary):
    """Helper function to plot model and dataset configurations."""
    all_config_summary.append(_plot_single_classification(axes[0], mnames[0], dnames[0], alpha, num_rounds))
    all_config_summary.append(_plot_single_classification(axes[1], mnames[0], dnames[1], alpha, num_rounds))
    all_config_summary.append(_plot_single_classification(axes[2], mnames[1], dnames[0], alpha, num_rounds))
    all_config_summary.append(_plot_single_classification(axes[3], mnames[1], dnames[1], alpha, num_rounds))

def _plot_text_image_audio_classification_results():
    """
    Plot classification results for text, image, and audio modalities.
    """
    alpha = 0.3
    global abc
    abc = 0
    width = 3.3374*1.6
    height = 3.3374 * 1.5
    fig, all_axes = _create_subplots(
        width_inches=width, height_inches=height, nrows=3, ncols=4, sharey=True)

    all_config_summary = []

    text_models = ['openai-communityopenai-gpt',
                   'google-bertbert-base-cased']
    text_datasets = ['dbpedia_14', 'yahoo_answers_topics']
    image_models = ['resnet18', 'densenet121']
    standd_datasets = ['mnist', 'cifar10']
    medical_datasets = ['pathmnist', 'organamnist']

    _plot_model_dataset_configs(all_axes[0], image_models, medical_datasets, 25, alpha, all_config_summary)
    _plot_model_dataset_configs(all_axes[1], text_models, text_datasets, 25, alpha, all_config_summary)
    _plot_model_dataset_configs(all_axes[2], image_models, standd_datasets, 50, alpha, all_config_summary)

    _remove_individual_legends_and_add_common_legend(all_axes, fig, bbox_to_anchor=(0.5, 0.04))
    fig.supxlabel('Communication Rounds', fontsize=12)
    fig.supylabel('Accuracy (%)', fontsize=12)
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    plt.tight_layout()
    fname = f"text_image_audio_classification_results_{alpha}_alpha"
    _save_figure(fig, fname)

    total_rounds = sum([x['Total Rounds'] for x in all_config_summary])
    total_accuracy = sum([x['Total Accuracy'] for x in all_config_summary])
    average_accuracy = total_accuracy / total_rounds

    logging.info(f"-------------- {fname} --------------")
    logging.info(f"Total Rounds: {total_rounds}")
    logging.info(f"Total Accuracy: {total_accuracy}")
    logging.info(f"Average Accuracy: {average_accuracy}")
    logging.info(f'Total Models Trained  {total_rounds * 10}')

    return {'Total Rounds': total_rounds, 'Total Accuracy': total_accuracy, 'Average Accuracy': average_accuracy, 'Total Models Trained': total_rounds * 10}

def _get_scaling_df(num_clients):
    fname = f'results_csvs/prov_Scaling-openai-communityopenai-gpt-dbpedia_14-faulty_clients[[]]-noise_rateNone-TClients{num_clients}-fedavg-(R15-clientsPerR10)-non_iid_dirichlet0.3-batch32-epochs2-lr0.001.csv'
    return pd.read_csv(fname)

def _get_clients_per_round_df(clients_per_round):
    fname = f'results_csvs/prov_Scaling-openai-communityopenai-gpt-dbpedia_14-faulty_clients[[]]-noise_rateNone-TClients400-fedavg-(R15-clientsPerR{clients_per_round})-non_iid_dirichlet0.3-batch32-epochs2-lr0.001.csv'
    return pd.read_csv(fname)

def _get_num_rounds_df():
    fname = f'results_csvs/prov_Scaling-openai-communityopenai-gpt-dbpedia_14-faulty_clients[[]]-noise_rateNone-TClients400-fedavg-(R100-clientsPerR10)-non_iid_dirichlet0.3-batch32-epochs2-lr0.001.csv'
    return pd.read_csv(fname)

def _plot_scalability_helper(ax, df, r, label):
    df['Accuracy'] = df['Accuracy'][:r] * 100
    df['test_data_acc'] = df['test_data_acc'][:r] * 100
    _plot_line(ax, range(len(df)), _smooth_data(
        df['Accuracy']), label=f'{label}-Smooth', linestyle='-')
    _plot_line(ax, range(len(df)), _smooth_data(
        df['test_data_acc']), label='FL Training-Smooth', linestyle='--')
    ax.set_ylim([0, 105])
    tracefl_avg_acc = df['Accuracy'].mean()
    ax.text(0.5, 0.5, f"TraceFL \n Avg. Acc {tracefl_avg_acc:.1f} %",
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize=9)
    ax.set_xlabel("Communication Rounds")
    ax.set_ylabel("Accuracy (%)")

def _generate_scalability_table_and_plot():
    """Generates the scalability results table and plots."""
    scaling_clients = [200, 400, 600, 800, 1000]
    per_round_clients = [20, 30, 40, 50]

    scaling_total_clients_dicts = []
    total_rounds_clients200_1000 = 0
    total_accuracy_clients200_1000 = 0

    for num_clients in scaling_clients:
        df = _get_scaling_df(num_clients)
        total_rounds_clients200_1000 += len(df)
        temp_dict = {
            'Total Clients': num_clients,
            'Global Model Accuracy %': round(df['test_data_acc'].max() * 100, 2),
            'TraceFL Avg. Accuracy %': round(df['Accuracy'].mean() * 100, 2)
        }
        scaling_total_clients_dicts.append(temp_dict)
        total_accuracy_clients200_1000 += (df['Accuracy'].sum() * 100)
    
    total_model_trained_clients200_1000 = total_rounds_clients200_1000 * 10

    logging.info("-------------- Total Clients 200-1000 --------------")
    logging.info(f"Total Rounds: {total_rounds_clients200_1000}")
    logging.info(f"Total Accuracy: {total_accuracy_clients200_1000}")
    logging.info(f"Average Accuracy: {total_accuracy_clients200_1000 / total_rounds_clients200_1000}")
    logging.info(f'Total Models Trained:  {total_model_trained_clients200_1000}')

    df_toal_clients = pd.DataFrame(scaling_total_clients_dicts)
    logging.info(df_toal_clients)

    latex_code_toal_clients = df_toal_clients.to_latex(index=False, float_format="%.2f")
    with open("graphs/tables/scalability_results_table_total_clients.tex", "w") as f:
        f.write(latex_code_toal_clients)

    clients_per_round_dicts = []
    total_rounds_per_round_client20_50 = 0
    total_accuracy_per_round_client20_50 = 0

    for clients_per_round in per_round_clients:
        df = _get_clients_per_round_df(clients_per_round)
        temp_dict = {
            'Clients Per Round': clients_per_round,
            'Global Model Accuracy %': round(df['test_data_acc'].max() * 100, 2),
            'TraceFL Avg. Accuracy %': round(df['Accuracy'].mean() * 100, 2)
        }
        clients_per_round_dicts.append(temp_dict)
        total_rounds_per_round_client20_50 += len(df)
        total_accuracy_per_round_client20_50 += (df['Accuracy'].sum() * 100)
    
    total_model_trained_per_round_client20_50 = sum([len(df) * client_per_round for df, client_per_round in zip([_get_clients_per_round_df(c) for c in per_round_clients], per_round_clients)])

    logging.info("-------------- Scalability Clients Per Round 20-50 --------------")
    logging.info(f"Total Rounds: {total_rounds_per_round_client20_50}")
    logging.info(f"Total Accuracy: {total_accuracy_per_round_client20_50}")
    logging.info(f"Average Accuracy: {total_accuracy_per_round_client20_50 / total_rounds_per_round_client20_50}")
    logging.info(f'Total Models Trained:  {total_model_trained_per_round_client20_50}')

    df_clients_per_round = pd.DataFrame(clients_per_round_dicts)
    logging.info(df_clients_per_round)

    latex_code_clients_per_round = df_clients_per_round.to_latex(index=False, float_format="%.2f")
    with open("graphs/tables/scalability_results_table_clients_per_round.tex", "w") as f:
        f.write(latex_code_clients_per_round)

    width = 3.3374
    height = 3.3374 / 1.6
    fig, all_axes = _create_subplots(width_inches=width, height_inches=height, nrows=1, ncols=1, sharey=True)

    num_rounds_exp = 80
    df = _get_num_rounds_df()
    _plot_scalability_helper(all_axes, df, num_rounds_exp, 'TraceFL')

    total_accuracy_num_rounds_80 = df['Accuracy'][:num_rounds_exp].sum()
    logging.info("-------------- Scalability Number of Rounds 80 --------------")
    logging.info(f"Total Rounds: {num_rounds_exp}")
    logging.info(f"Total Accuracy: {total_accuracy_num_rounds_80}")
    logging.info(f"Average Accuracy: {total_accuracy_num_rounds_80 / num_rounds_exp}")
    total_model_trained_rounds_80 = 80 * 10
    logging.info(f'Total Models Trained:  {total_model_trained_rounds_80}')

    all_axes.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.05), handletextpad=0.01, columnspacing=0.1, borderpad=0.1, labelspacing=0.1)
    plt.tight_layout()
    _save_figure(fig, f"scalability_results_400_clients_rounds_{num_rounds_exp}")

    total_rounds = total_rounds_clients200_1000 + total_rounds_per_round_client20_50 + num_rounds_exp
    total_accuracy = total_accuracy_clients200_1000 + total_accuracy_per_round_client20_50 + total_accuracy_num_rounds_80
    average_accuracy = total_accuracy / total_rounds
    total_model_trained = total_model_trained_clients200_1000 + total_model_trained_per_round_client20_50 + total_model_trained_rounds_80

    return {
        'Total Rounds': total_rounds,
        'Total Accuracy': total_accuracy,
        'Average Accuracy': average_accuracy,
        'Total Models Trained': total_model_trained
    }

def _get_dp_results(dclip, alpha, dp_noises):
    """Helper function to get DP results."""
    def _get_df_gpt(noise, clip, alpha):
        fname = f'results_csvs/prov_DP-(noise{noise}+clip{clip})-DP-text-openai-communityopenai-gpt-dbpedia_14-faulty_clients[[]]-noise_rateNone-TClients100-fedavg-(R15-clientsPerR10)-non_iid_dirichlet{alpha}-batch32-epochs2-lr0.001.csv'
        return pd.read_csv(fname)
    
    all_dfs = [_get_df_gpt(noise=ds, clip=dclip, alpha=alpha) for ds in dp_noises]
    average_prov_on_each_alpha = [df['Accuracy'].mean() * 100 for df in all_dfs]
    max_gm_acc_on_each_alpha = [df['test_data_acc'].max() * 100 for df in all_dfs]
    return {
        'prov': average_prov_on_each_alpha,
        'gm': max_gm_acc_on_each_alpha,
        'Total Rounds': sum([len(df) for df in all_dfs]),
        'Total Accuracy': sum([df['Accuracy'].sum() * 100 for df in all_dfs])
    }

def _plot_dp_single(ax, clip, alpha, dp_noises):
    """Helper function to plot a single DP result."""
    temp_dict = _get_dp_results(clip, alpha, dp_noises)
    ax.plot(dp_noises, temp_dict['prov'], label='Avg. TraceFL Accuracy')
    ax.plot(dp_noises, temp_dict['gm'], label='FL Training Accuracy')
    ax.set_xlabel("Differential Privacy Noise")
    ax.legend()
    return temp_dict

def _generate_dp_results_and_table():
    """Generates the DP results plot and table."""
    dp_noises = [0.0001, 0.0003, 0.0007, 0.0009, 0.001, 0.003]
    alpha_for_dp_exp = 0.2
    fig, all_axes = _create_subplots(width_inches=3.3374*1.3, height_inches=3.3374/1.718, nrows=1, ncols=1, sharey=True)
    temp_dict1 = _plot_dp_single(all_axes, 50, alpha_for_dp_exp, dp_noises)
    fig.supylabel('Accuracy (%)')
    _remove_individual_legends_and_add_common_legend(all_axes, fig, bbox_to_anchor=(0.5, 0.3))
    plt.tight_layout()
    _save_figure(fig, f"differential_privacy_results_alpha_{alpha_for_dp_exp}")
    
    # Table plotting
    dp_noises = [0.003]
    temp_dict2 = _get_dp_results(dclip=15, alpha=0.3, dp_noises=dp_noises)
    dict_for_df2 = {'DP Noise': 0.003, 'DP Clip': 15, 'FL Training Accuracy': temp_dict2['gm'], 'TraceFL Avg. Accuracy': temp_dict2['prov']}
    dp_noises = [0.006]
    temp_dict3 = _get_dp_results(dclip=10, alpha=0.3, dp_noises=dp_noises)
    dict_for_df3 = {'DP Noise': 0.006, 'DP Clip': 10, 'FL Training Accuracy': temp_dict3['gm'], 'TraceFL Avg. Accuracy': temp_dict3['prov']}
    dp_noises = [0.012]
    temp_dict4 = _get_dp_results(dclip=15, alpha=0.3, dp_noises=dp_noises)
    dict_for_df4 = {'DP Noise': 0.012, 'DP Clip': 15, 'FL Training Accuracy': temp_dict4['gm'], 'TraceFL Avg. Accuracy': temp_dict4['prov']}
    df = pd.DataFrame([dict_for_df2, dict_for_df3, dict_for_df4])
    logging.info(f'------- DP Results Table -------')
    logging.info(df)
    latex_code = df.to_latex(index=False)
    with open("graphs/tables/differential_privacy_results_table.tex", "w") as f:
        f.write(latex_code)
    total_rounds = temp_dict1['Total Rounds'] + temp_dict2['Total Rounds'] + temp_dict3['Total Rounds'] + temp_dict4['Total Rounds']
    total_accuracy = temp_dict1['Total Accuracy'] + temp_dict2['Total Accuracy'] + temp_dict3['Total Accuracy'] + temp_dict4['Total Accuracy']
    average_accuracy = total_accuracy / total_rounds
    total_model_trained = total_rounds * 10
    logging.info(f"-------------- differential_privacy_results_alpha_{alpha_for_dp_exp} --------------")
    logging.info(f"Total Rounds: {total_rounds}")
    logging.info(f"Total Accuracy: {total_accuracy}")
    logging.info(f"Average Accuracy: {average_accuracy}")
    logging.info(f'Total Models Trained:  {total_model_trained}')
    return {'Total Rounds': total_rounds, 'Total Accuracy': total_accuracy, 'Average Accuracy': average_accuracy, 'Total Models Trained': total_model_trained}

def _get_dirichlet_df(mname, dname, alpha):
    fname = f'results_csvs/prov_Dirichlet-Alpha-{mname}-{dname}-faulty_clients[[]]-noise_rateNone-TClients100-fedavg-(R15-clientsPerR10)-non_iid_dirichlet{alpha}-batch32-epochs2-lr0.001.csv'
    return pd.read_csv(fname)

def _get_avg_and_max_acc_dirichlet(mname, dname, all_alphas):
    all_dfs = [_get_dirichlet_df(mname, dname, a) for a in all_alphas]
    average_prov_on_each_alpha = [df['Accuracy'].mean() * 100 for df in all_dfs]
    max_gm_acc_on_each_alpha = [df['test_data_acc'].max() * 100 for df in all_dfs]
    return {
        'prov': average_prov_on_each_alpha,
        'gm': max_gm_acc_on_each_alpha,
        'Total Rounds': sum([len(df) for df in all_dfs]),
        'Total Accuracy': sum([df['Accuracy'].sum() * 100 for df in all_dfs])
    }

def _plot_dirichlet_single(ax, mname, dname, all_alphas):
    """Helper function to plot a single Dirichlet alpha result."""
    global abc
    dataset = _get_abbreviated_name(dname)
    temp_dict = _get_avg_and_max_acc_dirichlet(mname, dname, all_alphas)
    ax.plot(all_alphas, temp_dict['prov'], label='Avg. TraceFL Accuracy')
    ax.plot(all_alphas, temp_dict['gm'], label='Max Global Model Accuracy')
    title = f"{full_abc[abc]}) {dataset}"
    logging.info(f"Dirchelet Title: {title}")
    logging.info(f'alpha {all_alphas[0:5]}, gm {temp_dict["gm"][0:5]}')
    ax.set_title(title)
    ax.legend()
    abc += 1
    return temp_dict

def _generate_dirichlet_alpha_vs_accuracy_plot():
    """Generates the Dirichlet alpha vs accuracy plot."""
    global abc
    abc = 0
    width = 3.3374 * 1.6
    height = 3.3374 * 1.5
    all_alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    fig, axes = _create_subplots(width_inches=width, height_inches=height, nrows=3, ncols=2, sharey=True)
    temp_dict1 = _plot_dirichlet_single(axes[0][0], 'densenet121', 'pathmnist', all_alphas)
