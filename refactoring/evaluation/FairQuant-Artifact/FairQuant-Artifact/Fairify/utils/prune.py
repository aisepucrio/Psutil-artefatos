# -*- coding: utf-8 -*-

from codecarbon import EmissionsTracker
import psutil
import csv
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn import metrics
import collections
import time
import copy
from pathlib import Path
from random import randrange
from utils.standard_data import *
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from z3 import *

warnings.filterwarnings('ignore')

tracker = EmissionsTracker(project_name=r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\FairQuant-Artifact\Fairify\utils\prune.py")
tracker.start()

mem_start = psutil.virtual_memory().used / (1024**2)
cpu_start = psutil.cpu_percent(interval=None)


def create_z3_real_array(size, prefix):
    return np.array([Real(f'{prefix}{i}') for i in range(size)])


def create_z3_int_array(size, prefix):
    return np.array([Int(f'{prefix}{i}') for i in range(size)])


def relu(x):
    return np.vectorize(lambda y: If(y >= 0, y, 0))(x)


def layer_ws_net(x, w, b):
    return w.T @ x + b


def z3_layer1_ws_net_general(x, w, b, num_inputs):
    fl_x = create_z3_real_array(num_inputs, 'fl_x')
    for i in range(len(x)):
        fl_x[i] = ToReal(x[i])
    return layer_ws_net(fl_x, w[0], b[0])


def z3_layer1_ws_net_adult(x, w, b):
    return z3_layer1_ws_net_general(x, w, b, 13)


def z3_layer1_ws_net_german(x, w, b):
    return z3_layer1_ws_net_general(x, w, b, 20)


def z3_layer1_ws_net_bank(x, w, b):
    return z3_layer1_ws_net_general(x, w, b, 16)


def z3_layer_ws_net(x, w, b, layer_index):
    layer_functions = {
        1: z3_layer2_ws_net,
        2: z3_layer3_ws_net,
        3: z3_layer4_ws_net,
        4: z3_layer5_ws_net,
        5: z3_layer6_ws_net,
        6: z3_layer7_ws_net,
        7: z3_layer8_ws_net,
        8: z3_layer9_ws_net,
        9: z3_layer10_ws_net,
    }
    if layer_index in layer_functions:
        return layer_functions[layer_index](x, w, b)
    return None


def z3_layer2_ws_net(x, w, b):
    return layer_ws_net(x, w[1], b[1])


def z3_layer3_ws_net(x, w, b):
    return layer_ws_net(x, w[2], b[2])


def z3_layer4_ws_net(x, w, b):
    return layer_ws_net(x, w[3], b[3])


def z3_layer5_ws_net(x, w, b):
    return layer_ws_net(x, w[4], b[4])


def z3_layer6_ws_net(x, w, b):
    return layer_ws_net(x, w[5], b[5])


def z3_layer7_ws_net(x, w, b):
    return layer_ws_net(x, w[6], b[6])


def z3_layer8_ws_net(x, w, b):
    return layer_ws_net(x, w[7], b[7])


def z3_layer9_ws_net(x, w, b):
    return layer_ws_net(x, w[8], b[8])


def z3_layer10_ws_net(x, w, b):
    return layer_ws_net(x, w[9], b[9])


def harsh_prune(df, weight, bias, simulation_size, layer_net, range_dict):
    x_data = df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=False)
    sim_df = simluate_data(x_data, simulation_size, range_dict)
    candidates, _ = candidate_dead_nodes(sim_df.to_numpy(), weight, bias, layer_net)
    print(candidates)
    pr_w, pr_b = prune_neurons(weight, bias, candidates)
    compression = compression_ratio(candidates)
    print(round(compression, 2), '% pruning')
    return pr_w, pr_b


def calculate_neuron_bounds(df, w, b, range_dict):
    ub = []
    lb = []
    for col in df.columns:
        lb.append(range_dict[col][0])
        ub.append(range_dict[col][1])

    layer_bounds = []
    for l in range(len(w)):
        in_size = len(w[l])
        layer_size = len(w[l][0])
        min_arr_ws = []
        max_arr_ws = []
        min_arr_pl = []
        max_arr_pl = []

        for j in range(layer_size):
            min_t = 0
            max_t = 0
            for i in range(in_size):
                weight = w[l][i][j]
                if weight < 0:
                    min_t += weight * ub[i]
                    max_t += weight * lb[i]
                else:
                    min_t += weight * lb[i]
                    max_t += weight * ub[i]

            min_ws = min_t + b[l][j]
            max_ws = max_t + b[l][j]
            min_arr_ws.append(min_ws)
            max_arr_ws.append(max_ws)

            min_pl = 0 if min_ws < 0 else min_ws
            max_pl = 0 if max_ws < 0 else max_ws
            min_arr_pl.append(min_pl)
            max_arr_pl.append(max_pl)

        lb = min_arr_pl
        ub = max_arr_pl
        layer_bounds.append((min_arr_ws, max_arr_ws, min_arr_pl, max_arr_pl))
    return layer_bounds


def candidate_dead_nodes(data, weight, bias, layer_net):
    layer_counts = []
    layers = [layer_net(data[0], weight, bias) for _ in data]
    num_layers = len(layers[0])
    for _ in range(num_layers):
        layer_counts.append([])

    for i in range(len(data)):
        for l in range(len(layers[i])):
            if not layer_counts[l]:
                layer_counts[l] = [0] * len(layers[i][l])
            for j in range(len(layers[i][l])):
                if layers[i][l][j] != 0:
                    layer_counts[l][j] += 1

    dead_nodes = copy.deepcopy(layer_counts)
    for l in dead_nodes:
        for i in range(len(l)):
            l[i] = 1 if l[i] == 0 else 0

    positive_prob = copy.deepcopy(layer_counts)
    for l in positive_prob:
        for i in range(len(l)):
            l[i] = l[i] / len(data)
    return dead_nodes, positive_prob


def compression_ratio(deads):
    orig_neuron_count = 0
    compressed_neuron_count = 0
    for layer_index in range(len(deads)):
        for dead in deads[layer_index]:
            orig_neuron_count += 1
            if not dead:
                compressed_neuron_count += 1
    return 1 - (compressed_neuron_count / orig_neuron_count)


def simluate_data(df, size, range_dict):
    cols = df.columns.values
    sim_data = []
    for _ in range(size):
        data_instance = []
        for col in cols:
            data_instance.append(np.random.randint(range_dict[col][0], range_dict[col][1] + 1))
        sim_data.append(data_instance)
    sim_data_arr = np.asarray(sim_data)
    sim_df = pd.DataFrame(data=sim_data_arr, columns=cols)
    return sim_df


def dead_node_from_bound(cand, weight, bias, range_dict, ws_ub):
    print('INTERVAL BASED PRUNING')
    candidates = copy.deepcopy(cand)
    dead_node_mask = copy.deepcopy(candidates)
    count_finds = 0
    total_counts = 0

    for layer_index in range(len(ws_ub)):
        if layer_index == len(ws_ub) - 1:
            break
        for neuron_index in range(len(ws_ub[layer_index])):
            total_counts += 1
            if candidates[layer_index][neuron_index] == 0:
                continue
            if ws_ub[layer_index][neuron_index] <= 0:
                count_finds += 1
                dead_node_mask[layer_index][neuron_index] = 1
                candidates[layer_index][neuron_index] = 0
            else:
                dead_node_mask[layer_index][neuron_index] = 0
    return dead_node_mask, candidates, count_finds / total_counts


def create_input_domain_constraints(df, x, ranges):
    props = []
    for col_index, col in enumerate(df.columns.values):
        lb = ranges[col][0]
        ub = ranges[col][1]
        props.append(And(x[col_index] >= lb, x[col_index] <= ub))
    return props


def create_intermediate_domain_constraints(x, pl_lb, pl_ub, layer_index):
    props = []
    for neuron_index in range(len(pl_lb[layer_index])):
        props.append(x[neuron_index] >= pl_lb[layer_index][neuron_index])
        props.append(x[neuron_index] <= pl_ub[layer_index][neuron_index])
    return props


def singular_verification(cand, df, weight, bias, ranges, pl_lb, pl_ub,
                          layer_function, num_input_features):
    print('SINGULAR VERIFICATION')
    candidates = copy.deepcopy(cand)
    dead_node_mask = copy.deepcopy(candidates)
    count_finds = 0
    total_counts = 0

    for layer_index in range(len(candidates)):
        if layer_index == len(candidates) - 1:
            break
        for neuron_index in range(len(candidates[layer_index])):
            total_counts += 1
            if candidates[layer_index][neuron_index] == 0:
                continue

            if layer_index == 0:
                x = create_z3_int_array(num_input_features, 'x')
                in_props = create_input_domain_constraints(df, x, ranges)
            else:
                x = create_z3_real_array(len(weight[layer_index]), 'x')
                in_props = create_intermediate_domain_constraints(x, pl_lb, pl_ub, layer_index)

            y = z3_layer_ws_net(x, weight, bias, layer_index)

            s = Solver()
            for prop in in_props:
                s.add(prop)
            s.add(y[neuron_index] > 0)
            res = s.check()

            if res == unsat:
                count_finds += 1
                dead_node_mask[layer_index][neuron_index] = 1
                candidates[layer_index][neuron_index] = 0
            else:
                dead_node_mask[layer_index][neuron_index] = 0
    return dead_node_mask, candidates, count_finds / total_counts


def singular_verification_adult(cand, df, weight, bias, ranges, pl_lb, pl_ub):
    return singular_verification(cand, df, weight, bias, ranges, pl_lb, pl_ub,
                                 z3_layer1_ws_net_adult, 13)


def singular_verification_german(cand, df, weight, bias, ranges, pl_lb, pl_ub):
    return singular_verification(cand, df, weight, bias, ranges, pl_lb, pl_ub,
                                 z3_layer1_ws_net_german, 20)


def singular_verification_bank(cand, df, weight, bias, ranges, pl_lb, pl_ub):
    return singular_verification(cand, df, weight, bias, ranges, pl_lb, pl_ub,
                                 z3_layer1_ws_net_bank, 16)


def singular_verification_fairval(cand, df, weight, bias, ranges, pl_lb, pl_ub):
    return singular_verification(cand, df, weight, bias, ranges, pl_lb, pl_ub,
                                 z3_layer1_ws_net, len(weight[0]))


def sound_prune(df, weight, bias, simulation_size, layer_net, range_dict,
                 verification_function, label_name):
    x_df = df.drop(labels=[label_name], axis=1, inplace=False)
    sim_df = simluate_data(x_df, simulation_size, range_dict)
    candidates, pos_prob = candidate_dead_nodes(sim_df.to_numpy(), weight, bias, layer_net)
    layer_bounds = calculate_neuron_bounds(sim_df, weight, bias, range_dict)
    ws_lb, ws_ub, pl_lb, pl_ub = zip(*layer_bounds)
    b_dead_node_mask, b_candidates, _ = dead_node_from_bound(candidates, weight, bias, range_dict, ws_ub)

    for l in b_dead_node_mask:
        if not 0 in l:
            l[0] = 0

    s_dead_node_mask, s_candidates, _ = verification_function(b_candidates, x_df, weight, bias, range_dict, pl_lb, pl_ub)

    for l in s_dead_node_mask:
        if not 0 in l:
            l[0] = 0

    dead_nodes = merge_dead_nodes(b_dead_node_mask, s_dead_node_mask)

    for l in dead_nodes:
        if not 0 in l:
            l[0] = 0
    return layer_bounds, candidates, s_candidates, b_dead_node_mask, s_dead_node_mask, dead_nodes, pos_prob, sim_df


def sound_prune_adult(df, weight, bias, simulation_size, layer_net, range_dict):
    return sound_prune(df, weight, bias, simulation_size, layer_net, range_dict,
                       singular_verification_adult, 'income-per-year')


def sound_prune_bank(df, weight, bias, simulation_size, layer_net, range_dict):
    return sound_prune(df, weight, bias, simulation_size, layer_net, range_dict,
                       singular_verification_bank, 'y')


def sound_prune_german(df, weight, bias, simulation_size, layer_net, range_dict):
    return sound_prune(df, weight, bias, simulation_size, layer_net, range_dict,
                       singular_verification_german, 'credit')


def sound_prune_compas(df, weight, bias, simulation_size, layer_net, range_dict):
    return sound_prune(df, weight, bias, simulation_size, layer_net, range_dict,
                       singular_verification_fairval, 'score_factor')


def heuristic_prune(bounds, candidates, s_candidates, deads, pos_prob, perc_threshold, w, b):
    (ws_lb, ws_ub, pl_lb, pl_ub) = bounds
    new_deads = copy.deepcopy(candidates)
    for l in new_deads:
        for i in range(len(l)):
            l[i] = 0

    for layer_id in range(len(candidates)):
        if layer_id == len(candidates) - 1:
            break
        cand = []
        noncand = []
        for neuron_id in range(len(candidates[layer_id])):
            if candidates[layer_id][neuron_id]:
                cand.append(ws_ub[layer_id][neuron_id])
            else:
                noncand.append(ws_ub[layer_id][neuron_id])

        if len(noncand) == 0:
            for neuron_id in range(len(s_candidates[layer_id])):
                new_deads[layer_id][neuron_id] = 1
        elif len(cand) == 0:
            pass
        else:
            cand_min = min(cand)
            noncand_min = min(noncand)
            cand_max = max(cand)
            noncand_max = max(noncand)
            cand_mean = np.mean(np.array(cand))
            noncand_mean = np.mean(np.array(noncand))
            cand_median = np.median(np.array(cand))
            noncand_median = np.median(np.array(noncand))
            cand_5perc = np.percentile(np.array(cand), perc_threshold)
            noncand_5perc = np.percentile(np.array(noncand), perc_threshold)
            cand_95perc = np.percentile(np.array(cand), 100 - perc_threshold)
            noncand_95perc = np.percentile(np.array(noncand), 100 - perc_threshold)

            if noncand_mean > 2 * cand_mean and noncand_median > 2 * cand_median:
                for neuron_id in range(len(s_candidates[layer_id])):
                    if s_candidates[layer_id][neuron_id]:
                        if ws_ub[layer_id][neuron_id] < noncand_5perc:
                            if ws_ub[layer_id][neuron_id] < 0.1 * noncand_95perc:
                                if ws_ub[layer_id][neuron_id] < abs(ws_lb[layer_id][neuron_id]):
                                    new_deads[layer_id][neuron_id] = 1
    for l in new_deads:
        if not 0 in l:
            l[0] = 0
    merged_deads = merge_dead_nodes(deads, new_deads)
    for l in merged_deads:
        if not 0 in l:
            l[0] = 0
    return new_deads, merged_deads


def merge_dead_nodes(b, s):
    merged = copy.deepcopy(b)
    for layer_index in range(len(s)):
        for neuron_index in range(len(s[layer_index])):
            if s[layer_index][neuron_index] == 1:
                merged[layer_index][neuron_index] = 1
    return merged


def prune_neurons(weight, bias, candidates):
    pr_w = copy.deepcopy(weight)
    pr_b = copy.deepcopy(bias)
    for i in range(len(weight)):
        c = 0
        for j in range(len(candidates[i])):
            if candidates[i][j]:
                pr_w[i] = np.delete(pr_w[i], j - c, 1)
                pr_b[i] = np.delete(pr_b[i], j - c, 0)
                if i != len(weight) - 1:
                    pr_w[i + 1] = np.delete(pr_w[i + 1], j - c, 0)
                c += 1
    print('Pruning done!')
    return pr_w, pr_b


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
