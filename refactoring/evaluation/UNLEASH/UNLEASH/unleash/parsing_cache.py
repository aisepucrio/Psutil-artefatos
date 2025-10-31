from codecarbon import EmissionsTracker
import psutil
import csv
import os
import re
import sys
from collections import defaultdict, Counter, OrderedDict

sys.setrecursionlimit(1000000)
import multiprocessing as mp


def print_tree(move_tree, indent=' '):
    for key, value in move_tree.items():
        if isinstance(value, dict):
            print(f'{indent}|- {key}')
            print_tree(value, indent + '|  ')
        elif isinstance(value, tuple):
            print(f'{indent}|- {key}: tuple')
        else:
            print(f'{indent}|- {key}: {value}')


def lcs_similarity(X, Y):
    m, n = len(X), len(Y)
    c = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                c[i][j] = c[i - 1][j - 1] + 1
            else:
                c[i][j] = max(c[i][j - 1], c[i - 1][j])
    return 2 * c[m][n] / (m + n)


def post_process_tokens(tokens, punc):
    excluded_str = ['=', '|', '(', ')', ";"]
    for i in range(len(tokens)):
        if tokens[i].find("<*>") != -1:
            tokens[i] = "<*>"
        else:
            new_str = ""
            for s in tokens[i]:
                if (s not in punc and s != ' ') or s in excluded_str:
                    new_str += s
            tokens[i] = new_str
    return tokens


def message_split(message):
    punc = "!\"#$%&'()+,-/;:=?@.[\]^_`{|}~"
    splitters = "\s\\" + "\\".join(punc)
    splitter_regex = re.compile("([{}])".format(splitters))
    tokens = re.split(splitter_regex, message)

    tokens = list(filter(lambda x: x != "", tokens))
    tokens = post_process_tokens(tokens, punc)

    tokens = [
        token.strip()
        for token in tokens
        if token != "" and token != ' '
    ]
    tokens = [
        token
        for idx, token in enumerate(tokens)
        if not (token == "<*>" and idx > 0 and tokens[idx - 1] == "<*>")
    ]
    return tokens


def tree_match(match_tree, log_content):
    template, template_id, parameter_str = match_template(match_tree, log_content)
    if template:
        return (template, template_id, parameter_str)
    else:
        return ("NoMatch", "NoMatch", parameter_str)


def match_log(log, template):
    pattern_parts = template.split("<*>")
    pattern_parts_escaped = [re.escape(part) for part in pattern_parts]
    regex_pattern = "(.*?)".join(pattern_parts_escaped)
    regex = "^" + regex_pattern + "$"
    matches = re.search(regex, log)

    if matches is None:
        return False
    else:
        return True  # all(len(var.split()) == 1 for var in matches.groups())


def get_all_templates(move_tree):
    result = []
    for key, value in move_tree.items():
        if isinstance(value, tuple):
            result.append(value[2])
        else:
            result = result + get_all_templates(value)
    return result


class ParsingCache(object):
    def __init__(self):
        self.template_tree = {}
        self.template_list = []

    def add_templates(self, event_template, insert=True, relevant_templates=[]):
        template_tokens = message_split(event_template)
        if not template_tokens or event_template == "<*>":
            return -1

        if insert or not relevant_templates:
            template_id = self.insert(event_template, template_tokens, len(self.template_list))
            self.template_list.append(event_template)
            return template_id

        max_similarity, similar_template = self._find_similar_template(event_template, relevant_templates)

        if max_similarity > 0.8:
            success, template_id = self.modify(similar_template, event_template)
            if not success:
                template_id = self.insert(event_template, template_tokens, len(self.template_list))
                self.template_list.append(event_template)
            return template_id
        else:
            template_id = self.insert(event_template, template_tokens, len(self.template_list))
            self.template_list.append(event_template)
            return template_id

    def _find_similar_template(self, event_template, relevant_templates):
        max_similarity = 0
        similar_template = None
        for rt in relevant_templates:
            splited_template1, splited_template2 = rt.split(), event_template.split()
            if len(splited_template1) != len(splited_template2):
                continue
            similarity = lcs_similarity(splited_template1, splited_template2)
            if similarity > max_similarity:
                max_similarity = similarity
                similar_template = rt
        return max_similarity, similar_template

    def insert(self, event_template, template_tokens, template_id):
        move_tree = self._get_move_tree(template_tokens)
        move_tree["".join(template_tokens)] = (
            sum(1 for s in template_tokens if s != "<*>"),
            template_tokens.count("<*>"),
            event_template,
            template_id
        )
        return template_id

    def _get_move_tree(self, template_tokens):
        start_token = template_tokens[0]
        if start_token not in self.template_tree:
            self.template_tree[start_token] = {}
        move_tree = self.template_tree[start_token]
        for token in template_tokens[1:]:
            if token not in move_tree:
                move_tree[token] = {}
            move_tree = move_tree[token]
        return move_tree

    def modify(self, similar_template, event_template):
        merged_template = self._merge_templates(similar_template, event_template)
        if not merged_template:
            return False, -1

        success, old_ids = self.delete(similar_template)
        if not success:
            return False, -1

        self.insert(merged_template, message_split(merged_template), old_ids)
        self.template_list[old_ids] = merged_template
        return True, old_ids

    def _merge_templates(self, similar_template, event_template):
        similar_tokens = similar_template.split()
        event_tokens = event_template.split()

        if len(similar_tokens) != len(event_tokens):
            return None

        merged_template = []
        for i, token in enumerate(similar_tokens):
            if token == event_tokens[i]:
                merged_template.append(token)
            else:
                merged_template.append("<*>")
        return " ".join(merged_template)

    def delete(self, event_template):
        template_tokens = message_split(event_template)
        start_token = template_tokens[0]
        if start_token not in self.template_tree:
            return False, []
        move_tree = self.template_tree[start_token]

        for token in template_tokens[1:]:
            if token not in move_tree:
                return False, []
            move_tree = move_tree[token]
        old_id = move_tree["".join(template_tokens)][3]
        del move_tree["".join(template_tokens)]
        return True, old_id

    def match_event(self, log):
        return tree_match(self.template_tree, log)

    def _preprocess_template(self, template):
        return template


def match_template(match_tree, log_content):
    log_tokens = message_split(log_content)
    results = []
    find_results = find_template(match_tree, log_tokens, results, [], 1)
    relevant_templates = find_results[1]
    new_results = [result for result in results if all(result[i] is not None for i in range(1, 3))]
    if new_results:
        new_results.sort(key=lambda x: (-x[1][0], x[1][1]))
        return new_results[0][1][2], new_results[0][1][3], new_results[0][2]
    return False, False, relevant_templates


def _find_template_recursive(move_tree, log_tokens, result, parameter_list, depth):
    flag = 0
    if not log_tokens:
        for key, value in move_tree.items():
            if isinstance(value, tuple):
                result.append((key, value, tuple(parameter_list)))
                flag = 2
        if "<*>" in move_tree:
            parameter_list.append("")
            move_tree = move_tree["<*>"]
            if isinstance(move_tree, tuple):
                result.append(("<*>", None, None))
                flag = 2
            else:
                for key, value in move_tree.items():
                    if isinstance(value, tuple):
                        result.append((key, value, tuple(parameter_list)))
                        flag = 2
    else:
        token = log_tokens[0]
        relevant_templates = []
        if token in move_tree:
            find_result = _find_template_recursive(move_tree[token], log_tokens[1:], result, parameter_list, depth + 1)
            if find_result[0]:
                flag = 2
            elif flag != 2:
                flag = 1
                relevant_templates.extend(find_result[1])

        if "<*>" in move_tree:
            if isinstance(move_tree["<*>"], dict):
                next_keys = move_tree["<*>"].keys()
                next_continue_keys = [nk for nk in next_keys if not isinstance(move_tree["<*>"][nk], tuple)]
                idx = 0
                while idx < len(log_tokens):
                    token = log_tokens[idx]
                    if token in next_continue_keys:
                        parameter_list.append("".join(log_tokens[0:idx]))
                        find_result = _find_template_recursive(
                            move_tree["<*>"], log_tokens[idx:], result, parameter_list, depth + 1
                        )
                        if find_result[0]:
                            flag = 2
                        elif flag != 2:
                            flag = 1
                            relevant_templates.extend(find_result[1])
                        if parameter_list:
                            parameter_list.pop()
                        next_continue_keys.remove(token)
                    idx += 1
                if idx == len(log_tokens):
                    parameter_list.append("".join(log_tokens[0:idx]))
                    find_result = _find_template_recursive(
                        move_tree["<*>"], log_tokens[idx:], result, parameter_list, depth + 1
                    )
                    if find_result[0]:
                        flag = 2
                    else:
                        if flag != 2:
                            flag = 1
                    if parameter_list:
                        parameter_list.pop()
    if flag == 2:
        return True, []
    if flag == 1:
        return False, relevant_templates
    if flag == 0:
        if depth >= 2:
            return False, get_all_templates(move_tree)
        else:
            return False, []


def find_template(move_tree, log_tokens, result, parameter_list, depth):
    return _find_template_recursive(move_tree, log_tokens, result, parameter_list, depth)


tracker = EmissionsTracker(project_name=r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\UNLEASH\unleash\parsing_cache.py")
tracker.start()

mem_start = psutil.virtual_memory().used / (1024**2)
cpu_start = psutil.cpu_percent(interval=None)

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

mem_end = psutil.virtual_memory().used / (1024**2)
cpu_end = psutil.cpu_percent(interval=None)
tracker.stop()
