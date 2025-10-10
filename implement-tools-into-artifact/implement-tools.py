import os
import csv

def implement_codecarbon_and_psutil(base_path):
    end_block_template = [
        '\n',
        '# Collect final system metrics and stop tracker\n',
        'mem_end = psutil.virtual_memory().used / (1024**2)\n',
        'cpu_end = psutil.cpu_percent(interval=None)\n',
        '\n',
        '# Save psutil data to psutil_data.csv\n',
        'csv_file = "psutil_data.csv"\n',
        'file_exists = os.path.isfile(csv_file)\n',
        'with open(csv_file, "a", newline="") as csvfile:\n',
        '    writer = csv.writer(csvfile)\n',
        '    if not file_exists:\n',
        '        writer.writerow(["file", "mem_start_MB", "mem_end_MB", "mem_diff_MB", "cpu_start_percent", "cpu_end_percent"])\n',
        '    writer.writerow([\n',
        '        __file__,\n',
        '        f"{mem_start:.2f}",\n',
        '        f"{mem_end:.2f}",\n',
        '        f"{mem_end - mem_start:.2f}",\n',
        '        f"{cpu_start:.2f}",\n',
        '        f"{cpu_end:.2f}"\n',
        '    ])\n',
        '\n',
        'tracker.stop()\n'
    ]

    for root, _, files in os.walk(base_path):
        for file_name in files:
            if file_name.endswith('.py'):
                file_path = os.path.join(root, file_name)
                project_name = file_path  # Caminho completo até o arquivo

                # Try UTF-8 first, fallback to Latin1
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.readlines()
                    encoding_used = 'utf-8'
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding='latin1') as f:
                        content = f.readlines()
                    encoding_used = 'latin1'

                # Skip if already has CodeCarbon & psutil
                has_codecarbon = any('EmissionsTracker' in line for line in content)
                has_psutil = any('psutil' in line for line in content)
                if has_codecarbon and has_psutil:
                    print(f"[SKIP] {file_path} already contains CodeCarbon and psutil.")
                    continue

                # Find first non-comment, non-empty line
                insert_index = 0
                for i, line in enumerate(content):
                    if not line.strip().startswith('#') and line.strip():
                        insert_index = i
                        break

                code_to_insert = []
                if not has_codecarbon:
                    code_to_insert.append('from codecarbon import EmissionsTracker\n')
                if not has_psutil:
                    code_to_insert.append('import psutil\nimport csv\nimport os\n')
                if not has_codecarbon or not has_psutil:
                    code_to_insert.extend([
                        f'\n# Start CodeCarbon tracker with project name "{project_name}" and collect initial system metrics\n',
                        f'tracker = EmissionsTracker(project_name=r"{project_name}")\n',  # r"" para lidar com backslashes no Windows
                        'tracker.start()\n',
                        '\n',
                        'mem_start = psutil.virtual_memory().used / (1024**2)\n',
                        'cpu_start = psutil.cpu_percent(interval=None)\n',
                        '\n'
                    ])
                    content[insert_index:insert_index] = code_to_insert

                if not any('tracker.stop()' in line for line in content):
                    if not content[-1].endswith('\n'):
                        content[-1] += '\n'
                    content.extend(end_block_template)

                # Rewrite file
                with open(file_path, 'w', encoding=encoding_used) as f:
                    f.writelines(content)

                print(f"[OK] CodeCarbon/psutil added and CSV logging enabled: {file_path}")


if __name__ == "__main__":
    artifacts_path = os.path.abspath(
        r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos"
    )
    implement_codecarbon_and_psutil(artifacts_path)
