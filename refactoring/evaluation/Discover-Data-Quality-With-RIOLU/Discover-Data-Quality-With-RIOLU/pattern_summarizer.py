from codecarbon import EmissionsTracker
import psutil
import csv
import os
import re
import string
import numpy as np
from sklearn.cluster import KMeans
from utils import Utils


class PatternGenerator:

    def __init__(self, data, coverage_threshold):
        self.coverage_threshold = coverage_threshold
        self.sampling_size = int(0.2 * len(data))
        self.splits, self.indices_train = Utils.split_and_validate(data, self.sampling_size)
        self.test = self.splits[1]
        self.test_size = len(self.splits[1])
        self.digits = set(str(i) for i in range(10))
        self.upper_letters = set(string.ascii_uppercase)
        self.lower_letters = set(string.ascii_lowercase)
        self.type_mapping = {d: r'\d' for d in self.digits}
        self.type_mapping.update({l: '[A-Z]' for l in self.upper_letters})
        self.type_mapping.update({l: '[a-z]' for l in self.lower_letters})
        self.template_information = {}
        self.patterns = []
        self.pattern_coverage = {}

    def information_gathering(self, symbols, column, coverage):
        self.template_information = {}
        max_length = Utils.symbol_length(symbols, column, coverage)
        for i in range(len(column)):
            template, token_length_bag, token_char_bag, token_bag = Utils.token_info(
                symbols, column[i], max_length
            )
            if template not in self.template_information:
                self.template_information[template] = {}
            self._process_token_info(
                template, token_length_bag, token_char_bag, token_bag
            )

    def _process_token_info(self, template, token_length_bag, token_char_bag, token_bag):
        for token, length in token_length_bag.items():
            token_chars = token_char_bag[token]
            current_token = token_bag[token]
            if token not in self.template_information[template]:
                self.template_information[template][token] = {
                    'length': {},
                    'chars': {},
                    'token': {}
                }
            self._update_token_stats(
                template, token, length, current_token, token_chars
            )

    def _update_token_stats(self, template, token, length, current_token, token_chars):
        self._update_length_counts(template, token, length)
        self._update_token_counts(template, token, current_token)
        self._update_char_counts(template, token, token_chars)

    def _update_length_counts(self, template, token, length):
        if length in self.template_information[template][token]['length']:
            self.template_information[template][token]['length'][length] += 1
        else:
            self.template_information[template][token]['length'][length] = 1

    def _update_token_counts(self, template, token, current_token):
        if current_token in self.template_information[template][token]['token']:
            self.template_information[template][token]['token'][current_token] += 1
        else:
            self.template_information[template][token]['token'][current_token] = 1

    def _update_char_counts(self, template, token, token_chars):
        for j, char in enumerate(token_chars):
            pos_key = 'pos_%d' % j
            if pos_key not in self.template_information[template][token]['chars']:
                self.template_information[template][token]['chars'][pos_key] = {char: 1}
            else:
                if char not in self.template_information[template][token]['chars'][pos_key]:
                    self.template_information[template][token]['chars'][pos_key][char] = 1
                else:
                    self.template_information[template][token]['chars'][pos_key][char] += 1

    def pattern_generation(self, symbols, column):
        self.information_gathering(symbols, column, self.coverage_threshold)
        for template, token_stats in self.template_information.items():
            composed_template = template
            for _, stats in token_stats.items():
                composed_template = self._process_token_stats(
                    composed_template, stats, symbols, column
                )
            self.patterns.append(composed_template)

    def _process_token_stats(self, composed_template, stats, symbols, column):
        token_char = ''
        coverages = [value / sum(stats['token'].values()) for value in stats['token'].values()]
        categories = list(stats['token'].keys())
        composed_template = self._handle_token_categories(
            composed_template, coverages, categories, column
        )
        if 'TOKEN' not in composed_template:
            return composed_template
        filtered = Utils.rank_and_threshold(stats['length'], self.coverage_threshold)
        length_constraint, minimum_constraint = self._get_length_constraints(filtered)
        last_type = ''
        type_count = 0
        force_dumped = False
        for pos, char_stats in stats['chars'].items():
            if self._should_skip_char_processing(int(pos[4:]), minimum_constraint, length_constraint, force_dumped):
                continue
            token_char, last_type, type_count, force_dumped = self._process_char_stats(
                token_char, char_stats, symbols, column, last_type, type_count,
                minimum_constraint, force_dumped
            )
            if 'TOKEN' not in composed_template:
                return composed_template
        composed_template = self._finalize_token_char(composed_template, token_char, last_type, type_count, force_dumped)
        return composed_template

    def _handle_token_categories(self, composed_template, coverages, categories, column):
        if len(coverages) > 1:
            coverages.extend([1, 1 / len(column)])
            X = np.array(coverages).reshape(-1, 1)
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
            kmeans.fit(X)
            cluster_labels = kmeans.labels_
            label_select = np.argmax(kmeans.cluster_centers_)
            categories_selected = [categories[i] for i in range(len(categories)) if cluster_labels[i] == label_select]
            categories_coverage = sum(
                [coverages[i] for i in range(len(coverages) - 2) if cluster_labels[i] == label_select]
            )
            if categories_coverage >= self.coverage_threshold:
                token_list = sorted([re.escape(token) for token in categories_selected])
                if len(token_list) > 1:
                    composed_template = re.sub(r'TOKEN', '(' + '|'.join(token_list) + ')', composed_template, 1)
                elif len(token_list) == 1:
                    composed_template = re.sub(r'TOKEN', token_list[0], composed_template, 1)
        else:
            composed_template = re.sub(r'TOKEN', re.escape(categories[0]), composed_template, 1)
        return composed_template

    def _get_length_constraints(self, filtered):
        if len(filtered.keys()) == 1:
            length_constraint = list(filtered.keys())[0]
            minimum_constraint = length_constraint
        else:
            length_constraint = '+'
            minimum_constraint = min(list(filtered.keys()))
        return length_constraint, minimum_constraint

    def _should_skip_char_processing(self, pos_index, minimum_constraint, length_constraint, force_dumped):
        if pos_index >= minimum_constraint:
            if length_constraint != '+':
                return True
            elif not force_dumped:
                return False
        return False

    def _process_char_stats(self, token_char, char_stats, symbols, column, last_type, type_count, minimum_constraint, force_dumped):
        filtered = Utils.rank_and_threshold(char_stats, self.coverage_threshold)
        coverages = [value / sum(char_stats.values()) for value in char_stats.values()]
        chars = list(char_stats.keys())
        coverages.extend([1, 1 / len(column)])
        X = np.array(coverages).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
        kmeans.fit(X)
        cluster_labels = kmeans.labels_
        label_select = np.argmax(kmeans.cluster_centers_)
        chars_selected = [chars[i] for i in range(len(chars)) if cluster_labels[i] == label_select]
        chars_coverage = sum([coverages[i] for i in range(len(chars)) if cluster_labels[i] == label_select])
        if chars_coverage >= self.coverage_threshold and int(pos[4:]) < minimum_constraint:
            if last_type != '':
                token_char += '%s{%d}' % (last_type, type_count)
            char_list = sorted([re.escape(token) for token in chars_selected])
            if len(char_list) > 1:
                token_char += '(' + '|'.join(char_list) + ')'
            elif len(char_list) == 1:
                token_char += char_list[0]
            last_type = ''
        else:
            token_char, last_type, type_count = self._process_char_type(token_char, filtered, symbols, last_type, type_count, minimum_constraint)
        return token_char, last_type, type_count, force_dumped

    def _process_char_type(self, token_char, filtered, symbols, last_type, type_count, minimum_constraint):
        mapped = np.unique([self.type_mapping[key] for key in filtered.keys()])
        if len(mapped) == 1:
            if mapped[0] == last_type:
                type_count += 1
            else:
                if last_type != '':
                    if int(pos[4:]) > minimum_constraint:
                        token_char += '%s*' % (last_type)
                    else:
                        token_char += '%s{%d}' % (last_type, type_count)
                last_type = mapped[0]
                type_count = 1
        elif all(m for m in mapped if m not in symbols):
            current_type = '['
            for item in sorted(mapped):
                if item == r'\d':
                    current_type += '0-9'
                elif item == '[a-z]':
                    current_type += 'a-z'
                else:
                    current_type += 'A-Z'
            current_type += ']'
            if current_type == last_type:
                type_count += 1
            else:
                if last_type != '':
                    if int(pos[4:]) > minimum_constraint:
                        token_char += '%s*' % (last_type)
                    else:
                        token_char += '%s{%d}' % (last_type, type_count)
                last_type = current_type
                type_count = 1
        else:
            if '.' == last_type:
                type_count += 1
            else:
                if last_type != '':
                    if int(pos[4:]) > minimum_constraint:
                        token_char += '%s*' % (last_type)
                    else:
                        token_char += '%s{%d}' % (last_type, type_count)
                last_type = '.'
                type_count = 1
        return token_char, last_type, type_count

    def _finalize_token_char(self, composed_template, token_char, last_type, type_count, force_dumped):
        if last_type != '':
            if force_dumped:
                token_char += '%s*' % (last_type)
            else:
                token_char += '%s{%d}' % (last_type, type_count)
        composed_template = re.sub(r'TOKEN', r'%s' % token_char, composed_template, 1)
        return composed_template

    def pattern_coverage_statictics(self):
        train_data, test_data = self.splits
        boc_summary = Utils.bag_of_characters_summary(train_data)
        symbols = set(
            item
            for item in boc_summary.keys()
            if (
                item not in self.digits
                and item not in self.upper_letters
                and item not in self.lower_letters
            )
        )
        self.type_mapping.update({l: '[a-z]' for l in symbols})
        self.pattern_generation(symbols, train_data)
        for template in self.patterns:
            if template not in self.pattern_coverage:
                self.pattern_coverage[template] = {}
            pattern = re.compile(template)
            cov_whole = len(Utils.find_exact_match_elements(template, train_data + test_data))
            self.pattern_coverage[template] = cov_whole / (len(train_data) + len(test_data))
        print(self.pattern_coverage)


if __name__ == '__main__':
    tracker = EmissionsTracker(project_name=r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\Discover-Data-Quality-With-RIOLU\pattern_summarizer.py")
    tracker.start()
    mem_start = psutil.virtual_memory().used / (1024 ** 2)
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
            f"{mem_start:.2f}",
            f"{mem_start - mem_start:.2f}",
            f"{cpu_start:.2f}",
            f"{cpu_start:.2f}"
        ])

    tracker.stop()
