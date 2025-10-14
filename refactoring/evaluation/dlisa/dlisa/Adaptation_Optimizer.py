from codecarbon import EmissionsTracker
import psutil
import csv
import os
import random
from itertools import combinations
import numpy as np
import pandas as pd
import time
from Genetic_Algorithm import GeneticAlgorithm

class AdaptationOptimizer:
    def __init__(self, max_generation, pop_size, mutation_rate, crossover_rate, compared_algorithms, system, optimization_goal):
        self.max_generation = max_generation
        self.pop_size = pop_size
        self.compared_algorithms = compared_algorithms
        self.system = system
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.optimization_goal = optimization_goal

    def dynamic_optimization(self, data_folder, data_files, run_no):
        initial_seeds = None
        initial_seeds_ids = None

        for selected_algorithm in self.compared_algorithms:
            self._initialize_algorithm_data()
            output_folders = self._create_output_folders(selected_algorithm, run_no)

            for i, csv_file in enumerate(data_files):
                environment_name = os.path.splitext(csv_file)[0]
                data = pd.read_csv(os.path.join(data_folder, csv_file))
                config_space = data.iloc[:, :-1].values
                perf_space = data.iloc[:, -1].values

                if i < 2:
                    self.similarity_score[environment_name] = 0

                time_start = time.time()
                init_pop_config, init_pop_config_ids = self._get_initial_population(i, initial_seeds, initial_seeds_ids, config_space, selected_algorithm, environment_name)
                ga = GeneticAlgorithm(self.pop_size, self.mutation_rate, self.crossover_rate, self.optimization_goal)
                optimized_pop_configs, optimized_pop_perfs, optimized_pop_indices, evaluated_configs_to_perfs = ga.run(
                    init_pop_config, init_pop_config_ids, config_space, perf_space, self.max_generation,
                    self._get_environmental_selection_type(selected_algorithm), selected_algorithm, run_no, self.system,
                    environment_name)
                time_end = time.time()
                time_sum = np.array([time_end - time_start])

                self._save_optimization_results(output_folders, environment_name, optimized_pop_configs, optimized_pop_perfs,
                                             optimized_pop_indices, time_sum, evaluated_configs_to_perfs)
                self._update_historical_data(evaluated_configs_to_perfs, optimized_pop_configs, optimized_pop_perfs,
                                             optimized_pop_indices, environment_name)

            self._save_similarity_score(output_folders)

    def _initialize_algorithm_data(self):
        self.his_pop_configs = []
        self.his_pop_perfs = []
        self.his_pop_ids = []
        self.his_envs_name = []
        self.his_evaluated_configs_to_perfs = []
        self.similarity_score = {}

    def _create_output_folders(self, selected_algorithm, run_no):
        output_folder_pop_perf = f'results/{self.system}/tuning_results/{selected_algorithm}/optimized_pop_perf_run_{run_no}'
        output_folder_pop_config = f'results/{self.system}/tuning_results/{selected_algorithm}/optimized_pop_config_run_{run_no}'
        os.makedirs(output_folder_pop_perf, exist_ok=True)
        os.makedirs(output_folder_pop_config, exist_ok=True)
        return output_folder_pop_perf, output_folder_pop_config

    def _get_initial_population(self, i, initial_seeds, initial_seeds_ids, config_space, selected_algorithm,
                                environment_name):
        if i == 0:
            if initial_seeds is None:
                initial_seeds, initial_seeds_ids = self.initialize_population(config_space, self.pop_size)
            return initial_seeds, initial_seeds_ids
        else:
            return self.generate_next_population(config_space, selected_algorithm, environment_name)

    def _get_environmental_selection_type(self, selected_algorithm):
        return 'LiDOS_selection' if selected_algorithm == 'LiDOS' else 'Traditional_selection'

    def _save_optimization_results(self, output_folders, environment_name, optimized_pop_configs, optimized_pop_perfs,
                                    optimized_pop_indices, time_sum, evaluated_configs_to_perfs):
        output_folder_pop_perf, output_folder_pop_config = output_folders
        np.savetxt(os.path.join(output_folder_pop_config, f'{environment_name}_config.csv'), optimized_pop_configs,
                   fmt='%f', delimiter=',')
        np.savetxt(os.path.join(output_folder_pop_perf, f'{environment_name}_perf.csv'), optimized_pop_perfs,
                   delimiter=',')
        np.savetxt(os.path.join(output_folder_pop_config, f'{environment_name}_indices.csv'), optimized_pop_indices,
                   delimiter=',', fmt='%d')
        np.savetxt(os.path.join(output_folder_pop_perf, f'{environment_name}_time.csv'), time_sum, delimiter=',')

    def _update_historical_data(self, evaluated_configs_to_perfs, optimized_pop_configs, optimized_pop_perfs,
                                  optimized_pop_indices, environment_name):
        self.his_evaluated_configs_to_perfs.append(evaluated_configs_to_perfs)
        self.his_pop_configs.append(optimized_pop_configs)
        self.his_pop_perfs.append(optimized_pop_perfs)
        self.his_pop_ids.append(optimized_pop_indices)
        self.his_envs_name.append(environment_name)

    def _save_similarity_score(self, output_folders):
        output_folder_pop_perf, _ = output_folders
        df = pd.DataFrame([self.similarity_score])
        df.to_csv(os.path.join(output_folder_pop_perf, 'similarity_score.csv'), index_label=False)


    def initialize_population(self, config_space, required_size, existing_configs=None, existing_ids=None):
        existing_configs_hashes = self._get_existing_config_hashes(existing_configs)
        existing_ids = set(existing_ids) if existing_ids is not None else set()

        pop_configs_from_data, pop_ids_from_data = self._get_configs_from_data(config_space, required_size, existing_configs_hashes, existing_ids)
        pop_configs_from_random = self._get_configs_from_random(config_space, required_size, existing_configs_hashes)
        population_configs = np.vstack((pop_configs_from_data, pop_configs_from_random))
        population_ids = np.concatenate((pop_ids_from_data, self._get_ids_from_random(config_space, population_configs, existing_ids)))
        return population_configs, population_ids

    def _get_existing_config_hashes(self, existing_configs):
        return set(map(lambda x: hash(x.tobytes()), existing_configs)) if existing_configs is not None else set()

    def _get_configs_from_data(self, config_space, required_size, existing_configs_hashes, existing_ids):
        pop_size_from_data = required_size // 2
        pop_configs_from_data = []
        pop_ids_from_data = []

        while len(pop_configs_from_data) < pop_size_from_data:
            idx = np.random.choice(len(config_space))
            config = config_space[idx]
            config_hash = hash(config.tobytes())
            if idx not in existing_ids:
                pop_configs_from_data.append(config)
                pop_ids_from_data.append(idx)
                existing_configs_hashes.add(config_hash)
                existing_ids.add(idx)
        return np.array(pop_configs_from_data), pop_ids_from_data

    def _get_configs_from_random(self, config_space, required_size, existing_configs_hashes):
        pop_configs_from_random = []
        while len(pop_configs_from_random) < required_size // 2:
            config = np.array([np.random.choice(np.unique(config_space[:, i])) for i in range(config_space.shape[1])])
            config_hash = hash(config.tobytes())
            if config_hash not in existing_configs_hashes:
                pop_configs_from_random.append(config)
                existing_configs_hashes.add(config_hash)
        return np.array(pop_configs_from_random)

    def _get_ids_from_random(self, config_space, random_configs, existing_ids):
        pop_ids_from_random = []
        for config in random_configs:
            matches = np.where((config_space == config).all(axis=1))[0]
            id = matches[0] if matches.size > 0 else -1
            pop_ids_from_random.append(id)
        return pop_ids_from_random

    def generate_next_population(self, config_space, selected_algorithm, environment_name, beta=0.3):
        if selected_algorithm == 'FEMOSAA':
            return self.initialize_population(config_space, self.pop_size)
        elif selected_algorithm == 'Seed-EA':
            return self._seed_ea_population()
        elif selected_algorithm == 'D-SOGA':
            return self._d_soga_population(config_space)
        elif selected_algorithm == 'LiDOS':
            return self._lidos_population()
        elif selected_algorithm == 'DLiSA':
            return self._dlisa_population(config_space, environment_name, beta)
        elif selected_algorithm in ('DLiSA-0.7', 'DLiSA-0.8', 'DLiSA-0.9', 'DLiSA-0.1', 'DLiSA-0.2', 'DLiSA-0.3', 'DLiSA-0.4', 'DLiSA-0.5', 'DLiSA-0.6', 'DLiSA-0.0'):
            return self.generate_next_population(config_space, 'DLiSA', environment_name, float(selected_algorithm.split('-')[1]))
        elif selected_algorithm == 'DLiSA-I':
            return self._dlisa_i_population(config_space, beta)
        elif selected_algorithm == 'DLiSA-II':
            return self._dlisa_ii_population(config_space, beta)
        return self.initialize_population(config_space, self.pop_size)

    def _seed_ea_population(self):
        return self.his_pop_configs[-1], self.his_pop_ids[-1]

    def _d_soga_population(self, config_space):
        full_pop_size_his_configs = self.his_pop_configs[-1]
        full_pop_size_his_config_ids = self.his_pop_ids[-1]
        full_pop_size_his_config_ids = np.array(full_pop_size_his_config_ids).flatten()
        selected_indices = np.random.choice(self.pop_size, size=self.pop_size // 10 * 8, replace=False)
        memory_pop_config = full_pop_size_his_configs[selected_indices]
        memory_pop_config_ids = full_pop_size_his_config_ids[selected_indices]
        random_pop_config, random_pop_config_ids = self.initialize_population(config_space, self.pop_size // 10 * 2,
                                                                              memory_pop_config,
                                                                              memory_pop_config_ids)
        init_pop_config = np.vstack((memory_pop_config, random_pop_config))
        init_pop_config_ids = np.concatenate((memory_pop_config_ids, random_pop_config_ids))
        return init_pop_config, init_pop_config_ids

    def _lidos_population(self):
        return self.his_pop_configs[-1], self.his_pop_ids[-1]

    def _dlisa_population(self, config_space, environment_name, beta):
        if not self.his_pop_ids:
            raise ValueError("The list of his_pop_ids is empty")
        if len(self.his_pop_ids) == 1:
            return self.generate_next_population_based_medium_similarity(config_space)
        average_similarity = self.calculate_average_similarity(self.his_evaluated_configs_to_perfs, beta)
        self.similarity_score[environment_name] = average_similarity
        if average_similarity >= beta:
            return self.generate_next_population_based_high_similarity(config_space)
        else:
            return self.initialize_population(config_space, self.pop_size)

    def _dlisa_i_population(self, config_space, beta):
        if not self.his_pop_ids:
            raise ValueError("The list of his_pop_ids is empty")
        if len(self.his_pop_ids) == 1:
            return self.generate_next_population_based_medium_similarity(config_space)
        average_similarity = self.calculate_average_similarity(self.his_evaluated_configs_to_perfs, beta)
        if average_similarity >= beta:
            return self._dlisa_i_high_similarity_population(config_space)
        else:
            return self.initialize_population(config_space, self.pop_size)

    def _dlisa_i_high_similarity_population(self, config_space):
        all_historical_configs, all_historical_ids = self._collect_historical_configs_and_ids()
        hashable_optima_configs, counts = self._get_unique_optima_configs_and_counts(all_historical_configs)
        config_to_id_mapping = {tuple(config): id for config, id in zip(all_historical_configs, all_historical_ids)}
        selected_config_tuples, selected_ids = self._select_configs_for_transfer(config_space, unique_optima_configs, counts, config_to_id_mapping)
        init_pop_config = np.array([list(config) for config in selected_config_tuples])
        init_pop_config_ids = selected_ids
        return init_pop_config, init_pop_config_ids

    def _dlisa_ii_population(self, config_space, beta):
        if not self.his_pop_ids:
            raise ValueError("The list of his_pop_ids is empty")
        if len(self.his_pop_ids) == 1:
            return self.generate_next_population_based_medium_similarity(config_space)
        average_similarity = random.random()
        if average_similarity >= beta:
            return self.generate_next_population_based_high_similarity(config_space)
        else:
            return self.initialize_population(config_space, self.pop_size)

    def _collect_historical_configs_and_ids(self):
        all_historical_configs = []
        all_historical_ids = []
        for configs, perfs, configs_ids in zip(self.his_pop_configs, self.his_pop_perfs, self.his_pop_ids):
            all_historical_configs.extend(configs)
            all_historical_ids.extend(configs_ids)
        return all_historical_configs, all_historical_ids

    def _get_unique_optima_configs_and_counts(self, all_historical_configs):
        hashable_optima_configs = [tuple(config) for config in all_historical_configs]
        return np.unique(hashable_optima_configs, axis=0, return_counts=True)

    def _select_configs_for_transfer(self, config_space, unique_optima_configs, counts, config_to_id_mapping):
        selected_config_tuples = []
        selected_ids = []
        if len(unique_optima_configs) >= self.pop_size:
            selected_indices = np.random.choice(len(unique_optima_configs), size=self.pop_size, replace=False)
            for idx in selected_indices:
                config_tuple = unique_optima_configs[idx]
                selected_config_tuples.append(config_tuple)
                selected_ids.append(config_to_id_mapping[tuple(config_tuple)])
        else:
            for config in unique_optima_configs:
                selected_config_tuples.append(config)
                selected_ids.append(config_to_id_mapping[tuple(config)])
            while len(selected_config_tuples) < self.pop_size:
                idx = np.random.choice(len(config_space))
                config = config_space[idx]
                if idx not in selected_ids:
                    selected_config_tuples.append(config)
                    selected_ids.append(idx)
        return selected_config_tuples, selected_ids

    def find_top_k_configs(self, configs, perfs, configs_ids, top_k=10):
        top_indices = np.argsort(perfs)[:top_k] if self.optimization_goal == 'minimum' else np.argsort(perfs)[::-1][:top_k]
        top_k_configs = [configs[i] for i in top_indices]
        top_k_configs_ids = [configs_ids[i] for i in top_indices]
        return top_k_configs, top_k_configs_ids

    def generate_next_population_based_high_similarity(self, config_space):
        all_local_optima_configs = []
        all_local_optima_ids = []
        for configs, perfs, configs_ids in zip(self.his_pop_configs, self.his_pop_perfs, self.his_pop_ids):
            top_k_configs, top_k_configs_ids = self.find_top_k_configs(configs, perfs, configs_ids, top_k=10)
            all_local_optima_configs.extend(top_k_configs)
            all_local_optima_ids.extend(top_k_configs_ids)
        hashable_optima_configs = [tuple(config) for config in all_local_optima_configs]
        unique_optima_configs, counts = np.unique(hashable_optima_configs, axis=0, return_counts=True)
        config_to_id_mapping = {tuple(config): id for config, id in zip(all_local_optima_configs, all_local_optima_ids)}
        latest_env_num = self._calculate_latest_env_num(unique_optima_configs)
        compound_weights = self._calculate_compound_weights(unique_optima_configs, counts, latest_env_num)
        probabilities = compound_weights / compound_weights.sum()
        selected_config_tuples, selected_ids = self._select_configs_for_transfer(config_space, unique_optima_configs, counts, config_to_id_mapping)
        init_pop_config = np.array([list(config) for config in selected_config_tuples])
        init_pop_config_ids = selected_ids
        return init_pop_config, init_pop_config_ids

    def _calculate_latest_env_num(self, unique_optima_configs):
        latest_env_num = {}
        for config in unique_optima_configs:
            latest_env = -1
            for i, env_configs in enumerate(self.his_pop_configs):
                if tuple(config) in [tuple(cfg) for cfg in env_configs]:
                    latest_env = i
            latest_env_num[tuple(config)] = latest_env
        return latest_env_num

    def _calculate_compound_weights(self, unique_optima_configs, counts, latest_env_num):
        compound_weights = []
        for config, count in zip(unique_optima_configs, counts):
            repeat_weight = count / len(self.his_pop_ids)
            latest_weight = (1 + latest_env_num[tuple(config)]) / len(self.his_pop_ids)
            compound_weight = repeat_weight + latest_weight
            compound_weights.append(compound_weight)
        return compound_weights

    def generate_next_population_based_medium_similarity(self, config_space):
        similar_pop_config = self.his_pop_configs[-1][:self.pop_size//2]
        similar_pop_config_ids = self.his_pop_ids[-1][:self.pop_size//2]
        random_pop_config, random_pop_config_ids = self.initialize_population(config_space, self.pop_size // 2,
                                                                              similar_pop_config,
                                                                              similar_pop_config_ids)
        init_pop_config = np.vstack((similar_pop_config, random_pop_config))
        init_pop_config_ids = np.concatenate((similar_pop_config_ids, random_pop_config_ids))
        return init_pop_config, init_pop_config_ids

    def calculate_similarity(self, env1, env2, common_solutions, beta):
        if len(common_solutions) > self.pop_size * 0.25:
            total_pairs = 0
            consistent_pairs = 0
            for sol1, sol2 in combinations(common_solutions, 2):
                total_pairs += 1
                perf_env1_sol1 = env1[sol1]
                perf_env1_sol2 = env1[sol2]
                perf_env2_sol1 = env2[sol1]
                perf_env2_sol2 = env2[sol2]
                if (perf_env1_sol1 > perf_env1_sol2) == (perf_env2_sol1 > perf_env2_sol2):
                    consistent_pairs += 1
            similarity_score = consistent_pairs / total_pairs if total_pairs > 0 else 0
        else:
            if beta == 0.0:
                beta = 0.3
            similarity_score = random.uniform(0, beta)
        return similarity_score

    def calculate_average_similarity(self, his_evaluated_configs_to_perfs, beta):
        n = len(his_evaluated_configs_to_perfs)
        total_similarity = 0
        count = 0
        for i in range(n-1):
            env1_evaluated_configs_to_perfs = his_evaluated_configs_to_perfs[i]
            env2_evaluated_configs_to_perfs = his_evaluated_configs_to_perfs[i + 1]
            common_solutions = set(env1_evaluated_configs_to_perfs.keys()) & set(env2_evaluated_configs_to_perfs.keys())
            similarity = self.calculate_similarity(env1_evaluated_configs_to_perfs, env2_evaluated_configs_to_perfs,
                                                    common_solutions, beta)
            total_similarity += similarity
            count += 1
        average_similarity = round(total_similarity / count, 2) if count > 0 else 0
        return average_similarity

# Start CodeCarbon tracker with project name "C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\dlisa\Adaptation_Optimizer.py" and collect initial system metrics
tracker = EmissionsTracker(project_name=r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\dlisa\Adaptation_Optimizer.py")
tracker.start()

mem_start = psutil.virtual_memory().used / (1024**2)
cpu_start = psutil.cpu_percent(interval=None)

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
