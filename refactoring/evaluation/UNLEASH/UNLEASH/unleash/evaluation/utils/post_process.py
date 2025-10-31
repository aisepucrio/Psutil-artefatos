from codecarbon import EmissionsTracker
import psutil
import csv
import os
import regex as re

def replace_consecutive_variables(template):
    while True:
        prev = template
        template = re.sub(r'<\*>\.<\*>', '<*>', template)
        if prev == template:
            break
    while True:
        prev = template
        template = re.sub(r'<\*><\*>', '<*>', template)
        if prev == template:
            break
    return template

def apply_replacements(template):
    replacements = {
        " #<*># ": " <*> ",
        " #<*> ": " <*> ",
        "<*>:<*>": "<*>",
        "<*>#<*>": "<*>",
        "<*>/<*>": "<*>",
        "<*>@<*>": "<*>",
        "<*>.<*>": "<*>",
        ' "<*>" ': ' <*> ',
        " '<*>' ": " <*> ",
        "<*><*>": "<*>"
    }
    for old, new in replacements.items():
        template = template.replace(old, new)
    return template

def tokenize_template(template, token_delimiters):
    tokens = re.split('(' + '|'.join(token_delimiters) + ')', template)
    return tokens

def process_tokens(tokens):
    new_tokens = []
    for token in tokens:
        if re.match(r'^\d+$', token):
            token = '<*>'
        if re.match(r'^[^\s\/]*<\*>[^\s\/]*$', token):
            if token != '<*>/<*>':
                token = '<*>'
        new_tokens.append(token)
    return new_tokens


def correct_single_template(template, user_strings=None):
    template = template.strip()
    template = re.sub(r'\s+', ' ', template)

    path_delimiters = {
        r'\s', r'\,', r'\!', r'\;', r'\:',
        r'\=', r'\|', r'\"', r'\'',
        r'\[', r'\]', r'\(', r'\)', r'\{', r'\}'
    }
    token_delimiters = path_delimiters.union({
        r'\.', r'\-', r'\+', r'\@', r'\#', r'\$', r'\%', r'\&',
    })

    tokens = tokenize_template(template, token_delimiters)
    new_tokens = process_tokens(tokens)
    template = ''.join(new_tokens)
    template = replace_consecutive_variables(template)
    template = apply_replacements(template)
    return template

tracker = EmissionsTracker(project_name=r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\UNLEASH\unleash\evaluation\utils\post_process.py")
tracker.start()

mem_start = psutil.virtual_memory().used / (1024**2)
cpu_start = psutil.cpu_percent(interval=None)

param_regex = [
    r'{([ :_#.\-\w\d]+)}',
    r'{}'
]

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
