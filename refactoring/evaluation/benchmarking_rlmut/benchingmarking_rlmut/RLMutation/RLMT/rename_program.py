import os
import sys
import argparse
import psutil
from codecarbon import EmissionsTracker

def initialize_tracker(project_name):
    tracker = EmissionsTracker(project_name=project_name)
    tracker.start()
    return tracker

def get_parent_directory():
    full_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.dirname(full_path)

def handle_environment_name_casing(old_environment_name):
    if old_environment_name in {"CartPole-V1", "LunarLander-V2"}:
        return old_environment_name[:-2] + old_environment_name[-2:].lower()
    return old_environment_name

def should_handle_logs_directory_rename(old_substring, folder_name):
    return old_substring == new_substring and old_substring not in folder_name and "logs" in folder_name

def get_corrected_substring_for_logs(old_substring):
    if "CartPole" in old_substring:
        return "CartPole-v1"
    elif "LunarLander" in old_substring:
        return "LunarLander-v2"
    return old_substring

def rename_folder(directory_path, folder_name, old_substring, new_substring):
    old_folder_path = os.path.join(directory_path, folder_name)
    if os.path.isdir(old_folder_path):
        new_folder_name = folder_name.replace(old_substring, new_substring)

        older_folder_path = os.path.join(directory_path, folder_name)
        new_folder_path = os.path.join(directory_path, new_folder_name)

        try:
            os.rename(older_folder_path, new_folder_path)
        except FileExistsError:
            print(f"Error: Folder '{new_folder_name}' already exists.")

def rename_folders(directory_path, old_substring, new_substring):
    if not os.path.exists(directory_path):
        print("Path ", directory_path, " doesn't exist")
        return

    for folder_name in os.listdir(directory_path):
        if should_handle_logs_directory_rename(old_substring, folder_name):
            old_substring = get_corrected_substring_for_logs(old_substring)
        rename_folder(directory_path, folder_name, old_substring, new_substring)

def construct_paths(parent_directory, args):
    if args.agent_type == 'healthy':
        old_environment_name = handle_environment_name_casing(args.old_environment_name)
        first_dir_path = os.path.join(parent_directory, 'experiments', 'Healthy_Agents', old_environment_name, args.algorithm, 'logs')
        second_dir_path = os.path.join(parent_directory, 'experiments', 'Healthy_Agents')
        return first_dir_path, second_dir_path, old_environment_name

    elif args.agent_type == 'mutated':
        mutation_path = os.path.join(parent_directory, 'experiments', 'Mutated_Agents', 'SingleOrderMutation', args.operator)
        if args.operator_value != "None":
            first_dir_path = os.path.join(mutation_path, args.old_environment_name, args.algorithm, args.operator_value, 'logs')
            second_dir_path = mutation_path
        else:
            first_dir_path = os.path.join(mutation_path, args.old_environment_name, args.algorithm, 'logs')
            second_dir_path = mutation_path
        return first_dir_path, second_dir_path, args.old_environment_name

    else:
        print("There's an error")
        sys.exit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('agent_type', help='Enter whether the agent is healthy or mutated')
    parser.add_argument('old_environment_name', help='Enter the name of the environment name you want to replace')
    parser.add_argument('new_environment_name', help='Enter the name of the environment name you want to replace your old name with')
    parser.add_argument('-op', '--operator', required=False, help='Enter the name of the mutation operator')
    parser.add_argument('-algo', '--algorithm', required=False, help='Enter the name of the algorithm')
    parser.add_argument('-op_val', '--operator_value', required=False, help='Enter the value of the operator parameter')
    args = parser.parse_args()

    print("Agent Type = ", args.agent_type)
    print("Old Environment Name = ", args.old_environment_name)
    print("New Environment Name = ", args.new_environment_name)
    print("Mutation Operator = ", args.operator)
    print("Algorithm = ", args.algorithm)
    print("Mutation Operator Value = ", args.operator_value)

    project_name = os.path.abspath(os.path.dirname(__file__))
    tracker = initialize_tracker(project_name)

    parent_directory = get_parent_directory()
    first_dir_path, second_dir_path, old_substring = construct_paths(parent_directory, args)

    new_substring = args.new_environment_name

    if args.operator == "policy_activation_change" and not os.path.exists(first_dir_path):
        first_dir_path = os.path.join(parent_directory, 'experiments', 'Mutated_Agents', 'SingleOrderMutation', args.operator, args.new_environment_name, args.algorithm, args.operator_value, 'logs')
        second_dir_path = os.path.join(parent_directory, 'experiments', 'Mutated_Agents', 'SingleOrderMutation', args.operator)
        rename_folders(first_dir_path, old_substring, new_substring)
    else:
        rename_folders(first_dir_path, old_substring, new_substring)
        rename_folders(second_dir_path, old_substring, new_substring)

    tracker.stop()

if __name__ == "__main__":
    main()
