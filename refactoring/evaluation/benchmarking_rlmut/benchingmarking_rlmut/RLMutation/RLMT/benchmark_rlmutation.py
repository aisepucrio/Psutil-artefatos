import psutil
import csv
import os
import subprocess
import sys
import shutil
from codecarbon import EmissionsTracker

def setup_codecarbon():
    tracker = EmissionsTracker()
    tracker.start()
    return tracker

def run_test(new_environment_name, algorithm, stdout_err_dir):
    stdout_file_name = os.path.join(stdout_err_dir, f"{new_environment_name}_{algorithm}_stdout_test.txt")
    stderr_file_name = os.path.join(stdout_err_dir, f"{new_environment_name}_{algorithm}_stderr_test.txt")

    test_agent_script = "run_test_agent.sh"
    command = ["bash", test_agent_script, new_environment_name, algorithm]

    with open(stdout_file_name, "w") as stdout, open(stderr_file_name, "w") as stderr:
        subprocess.run(command, stdout=stdout, stderr=stderr, check=True)


def compute_mutation_results(new_environment_name, algorithm, test_generator_type, mutant_type_from_file, results_dir, stdout_err_dir):
    stdout_file_name = os.path.join(results_dir, f"{new_environment_name}_{algorithm}_{test_generator_type}_{mutant_type_from_file}_stdout_mutation_result.txt")
    stderr_file_name = os.path.join(stdout_err_dir, f"{new_environment_name}_{algorithm}_{test_generator_type}_{mutant_type_from_file}_stderr_mutation_result.txt")
    test_agent_script = "mutation_results.sh"
    command = ["bash", test_agent_script, new_environment_name, algorithm, test_generator_type, mutant_type_from_file]
    with open(stdout_file_name, "w") as stdout, open(stderr_file_name, "w") as stderr:
        subprocess.run(command, stdout=stdout, stderr=stderr)

def get_environment_details(env, agent, algorithms):
    if env == "CartPole-v1" and agent == "Healthy":
        new_environment_name = "myCartPole-v1"
        target_directory = os.path.join(os.path.dirname(os.getcwd()), "experiments/Healthy_Agents")
        rename_script = "rename_folders_healthy.sh"
    elif env == "CartPole-v1" and agent == "Mutated":
        new_environment_name = "myCartPole-v1"
        target_directory = os.path.join(os.path.dirname(os.getcwd()), "experiments/Mutated_Agents/SingleOrderMutation/incorrect_loss_function")
        rename_script = "rename_folders_mutated.sh"
    elif env == "LunarLander-v2" and agent == "Healthy":
        new_environment_name = "myLunarLander-v1"
        target_directory = os.path.join(os.path.dirname(os.getcwd()), "experiments/Healthy_Agents")
        rename_script = "rename_folders_healthy.sh"
    elif env == "LunarLander-v2" and agent == "Mutated":
        new_environment_name = "myLunarLander-v1"
        target_directory = os.path.join(os.path.dirname(os.getcwd()), "experiments/Mutated_Agents/SingleOrderMutation/incorrect_loss_function")
        rename_script = "rename_folders_mutated.sh"
    else:
        return None, None, None  # or raise an exception

    items = os.listdir(target_directory)
    folders = [item for item in items if os.path.isdir(os.path.join(target_directory, item))]
    folders_with_substring = [folder for folder in folders if new_environment_name in folder]
    old_environment_name = new_environment_name if len(folders_with_substring) > 0 else env

    return old_environment_name, new_environment_name, rename_script

def rename_environment(old_environment_name, new_environment_name, algorithm, rename_program_script, stdout_err_dir):
    stdout_file_name = os.path.join(stdout_err_dir, f"{new_environment_name}_{algorithm}_stdout_rename.txt")
    stderr_file_name = os.path.join(stdout_err_dir, f"{new_environment_name}_{algorithm}_stderr_rename.txt")
    command = ["bash", rename_program_script, old_environment_name, new_environment_name, algorithm]
    with open(stdout_file_name, "w") as stdout, open(stderr_file_name, "w") as stderr:
        subprocess.run(command, stdout=stdout, stderr=stderr, check=True)

def rename(stdout_err_dir):
    environments = ["CartPole-v1", "LunarLander-v2"]
    agent_types = ["Healthy", "Mutated"]
    algorithms = ["PPO", "DQN"]

    for algorithm in algorithms:
        for env in environments:
            for agent_type in agent_types:
                result = get_environment_details(env, agent_type, algorithms)
                if result is None:
                    continue
                old_environment_name, new_environment_name, rename_program_script = result
                if rename_program_script:
                    rename_environment(old_environment_name, new_environment_name, algorithm, rename_program_script, stdout_err_dir)



def process_file(file_path, root, algorithm, run_test_func, compute_mutation_results_func, results_dir, stdout_err_dir):
    file_name = os.path.basename(file_path)
    parts = file_name.split("-")
    mutant_type_from_file = parts[-2]
    environment = root.rsplit("/", 1)[-1]
    if environment == "CartPole-v1":
        new_environment_name = "myCartPole-v1"
    elif environment == "LunarLander-v2":
        new_environment_name = "myLunarLander-v1"
    else:
        return

    destination_folder = 'testing'
    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)
    os.makedirs(destination_folder)
    shutil.copy(file_path, destination_folder)
    run_test_func(new_environment_name, algorithm, stdout_err_dir)
    shutil.rmtree(destination_folder)
    compute_mutation_results_func(new_environment_name, algorithm, root.rsplit("/", 1)[-1], mutant_type_from_file, results_dir, stdout_err_dir)

def run_tool(directory, results_dir, stdout_err_dir):
    rename(stdout_err_dir)
    for root, _, files in os.walk(directory):
        root_substring = root.rsplit("/", 1)[-1]
        if root_substring in ("CartPole-v1", "LunarLander-v2", "ppo", "dqn", "strong", "weak"):
            if root_substring in ("ppo", "dqn"):
                algorithm = root_substring
                for file in files:
                    if file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        process_file(file_path, root, algorithm, run_test, compute_mutation_results, results_dir, stdout_err_dir)
            elif root_substring in ("strong", "weak"):
                continue #Skip, handled in process_file
            
results_dir = 'results_mutation_benchmark'
if os.path.exists(results_dir):
    shutil.rmtree(results_dir)
os.makedirs(results_dir)
misc_stdout_err = 'stdout_err'
if os.path.exists(misc_stdout_err):
    shutil.rmtree(misc_stdout_err)
os.makedirs(misc_stdout_err)

def main():
    tracker = setup_codecarbon()
    top_directory = 'configurations'
    run_tool(top_directory, results_dir, misc_stdout_err)
    tracker.stop()

if __name__ == "__main__":
    main()
