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

    def __init__(self, data, coverage_threshold, sampling_size=None, constrained_temp=True):
        self.coverage_threshold = coverage_threshold
        self.sampling_size = int(np.ceil((1.96**2 * 0.5 * (1-0.5)) / (0.05**2))) if sampling_size is None else sampling_size
        self.splits, self.indices_train = Utils.split_and_validate(data, self.sampling_size)
        self.digits = set(str(i) for i in range(10))
        self.upper_letters = set(string.ascii_uppercase)
        self.lower_letters = set(string.ascii_lowercase)
        self.type_mapping = {d: r'\d' for d in self.digits}
        self.type_mapping.update({l: '[A-Z]' for l in self.upper_letters})
        self.type_mapping.update({l: '[a-z]' for l in self.lower_letters})
        self.template_information = {}
        self.patterns = []
        self.pattern_coverage = {}
        self.constrained_temp = constrained_temp

    def _get_max_length(self, symbols, column, coverage):
        if self.constrained_temp:
            return Utils.symbol_length(symbols, column, coverage)
        return Utils.symbol_length(symbols, column, 1)

    def _process_token_info(self, symbols, column, max_length):
        template_info = {}
        for i in range(len(column)):
            template, token_length_bag, token_char_bag, token_bag = Utils.token_info(symbols, column[i], max_length)
            if template not in template_info:
                template_info[template] = {}
            for token, length in token_length_bag.items():
                token_chars = token_char_bag[token]
                current_token = token_bag[token]
                if token not in template_info[template]:
                    template_info[template][token] = {'length': {}, 'chars': {}, 'token': {}}
                self._update_token_info(template_info[template][token], length, token_chars, current_token)
        return template_info

    def _update_token_info(self, token_info, length, token_chars, current_token):
        if length in token_info['length']:
            token_info['length'][length] += 1
        else:
            token_info['length'][length] = 1
        if current_token in token_info['token']:
            token_info['token'][current_token] += 1
        else:
            token_info['token'][current_token] = 1
        for j, char in enumerate(token_chars):
            pos_key = 'pos_%d' % j
            if pos_key not in token_info['chars']:
                token_info['chars'][pos_key] = {char: 1}
            elif char in token_info['chars'][pos_key]:
                token_info['chars'][pos_key][char] += 1
            else:
                token_info['chars'][pos_key][char] = 1

    def information_gathering(self, symbols, column, coverage):
        self.template_information = {}
        max_length = self._get_max_length(symbols, column, coverage)
        self.template_information = self._process_token_info(symbols, column, max_length)

    def _cluster_coverages(self, coverages, categories):
        X = np.array(coverages).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, n_init='auto', random_state=0)
        kmeans.fit(X)
        cluster_labels = kmeans.labels_
        label_select = np.argmax(kmeans.cluster_centers_)
        selected_indices = [i for i in range(len(categories)) if cluster_labels[i] == label_select]
        return selected_indices

    def _process_token_level(self, composed_template, token_stats, symbols, column):
        coverages = [value / sum(token_stats['token'].values()) for value in token_stats['token'].values()]
        categories = list(token_stats['token'].keys())
        if len(coverages) > 1:
            coverages.extend([1, 1 / len(column)])
            selected_indices = self._cluster_coverages(coverages, categories)
            categories_selected = [categories[i] for i in selected_indices]
            categories_coverage = sum([coverages[i] for i in range(len(coverages) - 2) if i in selected_indices])
            if categories_coverage >= self.coverage_threshold:
                token_list = sorted([re.escape(token) for token in categories_selected])
                composed_template = re.sub('TOKEN', '(' + '|'.join(token_list) + ')', composed_template, 1) if len(token_list) > 1 else re.sub('TOKEN', token_list[0], composed_template, 1)
                return composed_template
        else:
            composed_template = re.sub('TOKEN', re.escape(categories[0]), composed_template, 1)
            return composed_template
        return composed_template

    def _process_char_level(self, token_char, stats, minimum_constraint, symbols, column, force_dumped):
        last_type = ''
        type_count = 0
        for pos, char_stats in stats['chars'].items():
            if int(pos[4:]) >= minimum_constraint:
                if last_type != '' and not force_dumped:
                    token_char += '%s{%d}' % (last_type, type_count)
                    last_type = ''
                    force_dumped = True
                break
            filtered = Utils.rank_and_threshold(char_stats, self.coverage_threshold)
            coverages = [value / sum(char_stats.values()) for value in char_stats.values()]
            chars = list(char_stats.keys())
            coverages.extend([1, 1 / len(column)])
            selected_indices = self._cluster_coverages(coverages, chars)
            chars_selected = [chars[i] for i in selected_indices]
            chars_coverage = sum([coverages[i] for i in range(len(chars)) if i in selected_indices])
            if chars_coverage >= self.coverage_threshold and int(pos[4:]) < minimum_constraint:
                if last_type != '':
                    token_char += '%s{%d}' % (last_type, type_count)
                char_list = sorted([re.escape(token) for token in chars_selected])
                token_char += '(' + '|'.join(char_list) + ')' if len(char_list) > 1 else char_list[0]
                last_type = ''
            else:
                mapped = np.unique([self.type_mapping[key] for key in filtered.keys()])
                if len(mapped) == 1:
                    if mapped[0] == last_type:
                        type_count += 1
                    else:
                        if last_type != '':
                            token_char += '%s*' % (last_type) if int(pos[4:]) > minimum_constraint else '%s{%d}' % (last_type, type_count)
                        last_type = mapped[0]
                        type_count = 1
                elif all(m for m in mapped if m not in symbols):
                    current_type = '[' + ''.join(sorted(item.replace(r'\\', '') for item in mapped if item != '\\\\d')) + ']'
                    if current_type == last_type:
                        type_count += 1
                    else:
                        if last_type != '':
                            token_char += '%s*' % (last_type) if int(pos[4:]) > minimum_constraint else '%s{%d}' % (last_type, type_count)
                        last_type = current_type
                        type_count = 1
                else:
                    if '.' == last_type:
                        type_count += 1
                    else:
                        if last_type != '':
                            token_char += '%s*' % (last_type) if int(pos[4:]) > minimum_constraint else '%s{%d}' % (last_type, type_count)
                        last_type = '.'
                        type_count = 1
        if last_type != '':
            token_char += '%s*' % (last_type) if force_dumped else '%s{%d}' % (last_type, type_count)
        return token_char

    def pattern_generation(self, symbols, column):
        self.information_gathering(symbols, column, self.coverage_threshold)
        for template, token_stats in self.template_information.items():
            composed_template = template
            for _, stats in token_stats.items():
                composed_template = self._process_token_level(composed_template, stats, symbols, column)
                filtered = Utils.rank_and_threshold(stats['length'], self.coverage_threshold)
                length_constraint = list(filtered.keys())[0] if len(filtered.keys()) == 1 else '+'
                minimum_constraint = list(filtered.keys())[0] if len(filtered.keys()) == 1 else min(list(filtered.keys()))
                token_char = self._process_char_level('', stats, minimum_constraint, symbols, column, False)
                composed_template = re.sub('TOKEN', r'%s' % token_char, composed_template, 1)
            self.patterns.append(composed_template)

    def pattern_coverage_statictics(self):
        train_data, test_data = self.splits
        boc_summary = Utils.bag_of_characters_summary(train_data)
        symbols = set(item for item in boc_summary.keys() if
                      (item not in self.digits and item not in self.upper_letters and item not in self.lower_letters))
        self.type_mapping.update({l: '[a-z]' for l in symbols})
        self.pattern_generation(symbols, train_data)
        for template in self.patterns:
            if template not in self.pattern_coverage:
                self.pattern_coverage[template] = {}
            pattern = re.compile(template)
            cov_train = len(Utils.find_exact_match_elements(pattern, train_data))
            cov_test = len(Utils.find_exact_match_elements(pattern, test_data))
            cov_whole = (cov_train + cov_test) / (len(train_data) + len(test_data))
            self.pattern_coverage[template] = cov_whole

if __name__ == '__main__':
    tracker = EmissionsTracker(project_name=r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\Discover-Data-Quality-With-RIOLU\pattern_generator.py")
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
            "0.00",
            "0.00",
            f"{cpu_start:.2f}",
            "0.00"
        ])
    tracker.stop()

