from codecarbon import EmissionsTracker
import psutil
import csv
import os
import re
import ast

def replace_code_in_file(file_path, code):
    with open(file_path, 'r') as file:
        content = file.read()
    indented_code = '\n'.join(['    ' + line for line in code.splitlines()])
    content = content.replace('TestString', indented_code)
    with open(file_path, 'w') as file:
        file.write(content)

def reset_file_content(file_path):
    content = """def testFun():
TestString"""
    with open(file_path, 'w') as file:
        file.write(content)

def extract_scope(block):
    scope_line_numbers = []
    for node in block.control_flow_nodes:
        instruction = node.instruction
        ast_node = instruction.node
        line_number = ast_node.lineno
        scope_line_numbers.append(line_number - 1)
    return scope_line_numbers

def transform_statement_with_ast(source_code, target_line_number):
    try:
        class ASTStatementVisitor(ast.NodeVisitor):
            def __init__(self, target_line_number):
                self.target_line_number = target_line_number
                self.transformed_result = None
            def visit_For(self, node):
                if node.lineno == self.target_line_number:
                    if isinstance(node.target, ast.Tuple):
                        variable_names = [elt.id for elt in node.target.elts]
                        index_variable = variable_names[0]
                        iterator_variable = variable_names[1]
                        self.transformed_result = f"{index_variable} <- index\n    {iterator_variable} <- iterator"
                    else:
                        iterator_variable = node.target.id
                        iterable_expression = ast.unparse(node.iter)
                        self.transformed_result = f"iterator -> {iterator_variable}, Iterate Over -> {iterable_expression}"
                self.generic_visit(node)
            def visit_If(self, node):
                if node.lineno == self.target_line_number:
                    condition_expression = ast.unparse(node.test)
                    self.transformed_result = f"({condition_expression})"
                self.generic_visit(node)
        syntax_tree = ast.parse(source_code)
        statement_visitor = ASTStatementVisitor(target_line_number)
        statement_visitor.visit(syntax_tree)
        if statement_visitor.transformed_result is None:
            return source_code.split('\n')[target_line_number - 1].strip()
        else:
            return statement_visitor.transformed_result
    except Exception:
        return None

def get_block_ranges_and_statements(cfg, code):
    block_ranges = {}
    block_statements = {}
    is_processing_started = False
    for block in cfg.blocks:
        if block.label == "<entry:testFun>":
            is_processing_started = True
        if not is_processing_started:
            continue
        if len(block.control_flow_nodes) == 0:
            continue
        for next_block in block.next:
            if next_block.label == "<raise>":
                continue
        block_line_numbers = extract_scope(block)
        block_start_line = min(block_line_numbers)
        block_ranges[block_start_line] = {"range": [min(block_line_numbers), max(block_line_numbers)]}
        transformed_statements = []
        for line_number in block_line_numbers:
            transformed_statement = transform_statement_with_ast(code, line_number)
            if transformed_statement is None:
                return None, None
            transformed_statements.append(transformed_statement)
        block_statements[block_start_line] = transformed_statements
    return block_ranges, block_statements

def _extract_block_index(block, block_ranges):
    line_number_list = []
    for node in block.control_flow_nodes:
        instruction = node.instruction
        ast_node = instruction.node
        line_number = ast_node.lineno
        line_number_list.append(line_number-1)
    return min(line_number_list)

def _get_next_block_index(block, block_ranges, true_scope_block, false_scope_block, no_condition_block_lines):
    true_next = None
    false_next = None
    no_condition_next = None

    if true_scope_block and false_scope_block:
        min_line_number = min(true_scope_block)
        max_line_number = max(true_scope_block)
        for key, value in block_ranges.items():
            if value['range'] == [min_line_number, max_line_number]:
                true_next = key
                break
        min_line_number = min(false_scope_block)
        max_line_number = max(false_scope_block)
        for key, value in block_ranges.items():
            if value['range'] == [min_line_number, max_line_number]:
                false_next = key
                break
    elif true_scope_block:
        min_line_number = min(true_scope_block)
        max_line_number = max(true_scope_block)
        for key, value in block_ranges.items():
            if value['range'] == [min_line_number, max_line_number]:
                true_next = key
                false_next = "<END>"
                break
    elif false_scope_block:
        min_line_number = min(false_scope_block)
        max_line_number = max(false_scope_block)
        for key, value in block_ranges.items():
            if value['range'] == [min_line_number, max_line_number]:
                true_next = "<END>"
                false_next = key
                break
    else:
        if not no_condition_block_lines:
            return None, None, None
        min_line_number = min(no_condition_block_lines)
        max_line_number = max(no_condition_block_lines)
        for key, value in block_ranges.items():
            if value['range'] == [min_line_number, max_line_number]:
                no_condition_next = key
                break
    return true_next, false_next, no_condition_next

def _get_scope_blocks(block):
    true_scope_block = []
    false_scope_block = []
    for true_block in block.branches.get(True, []).control_flow_nodes:
        instruction = true_block.instruction
        ast_node = instruction.node
        line_number = ast_node.lineno
        true_scope_block.append(line_number - 1)
    for false_block in block.branches.get(False, []).control_flow_nodes:
        instruction = false_block.instruction
        ast_node = instruction.node
        line_number = ast_node.lineno
        false_scope_block.append(line_number - 1)
    return true_scope_block, false_scope_block

def get_block_connections(block_ranges, cfg):
    block_connections = {}
    is_processing_started = False
    for block in cfg.blocks:
        true_scope_block = []
        false_scope_block = []
        no_condition_block_lines = []
        true_next = None
        false_next = None
        no_condition_next = None
        if block.label == "<entry:testFun>":
            is_processing_started = True
        if not is_processing_started:
            continue
        if len(block.control_flow_nodes) == 0:
            continue
        for next_block in block.next:
            if next_block.label == "<raise>":
                continue
        block_index = _extract_block_index(block, block_ranges)
        if block.branches:
            true_scope_block, false_scope_block = _get_scope_blocks(block)
        else:
            for next_block in block.next:
                for node in next_block.control_flow_nodes:
                    instruction = node.instruction
                    ast_node = instruction.node
                    line_number = ast_node.lineno
                    no_condition_block_lines.append(line_number - 1)
        true_next, false_next, no_condition_next = _get_next_block_index(
            block, block_ranges, true_scope_block, false_scope_block, no_condition_block_lines
        )
        block_connections[block_index] = {
            "with_condition": {"true": true_next, "false": false_next},
            "no_condition": no_condition_next,
        }
    return block_connections

def renumber_cfg_blocks(cfg_block_statements, cfg_block_range, cfg_block_connection):
    sorted_block_statements = {k: cfg_block_statements[k] for k in sorted(cfg_block_statements)}
    renumbering_dict = {old: new for new, old in enumerate(sorted_block_statements, start=1)}
    new_cfg_block_statements = {renumbering_dict[block_id]: statements for block_id, statements in sorted_block_statements.items()}
    sorted_block_ranges = {k: cfg_block_range[k] for k in sorted(cfg_block_range)}
    new_cfg_block_ranges = {renumbering_dict[block_id]: block_range for block_id, block_range in sorted_block_ranges.items()}
    new_cfg_block_connection = {}
    for old_block, connections in cfg_block_connection.items():
        new_block = renumbering_dict[old_block]
        new_cfg_block_connection[new_block] = {}
        for condition, target in connections.items():
            if condition == "no_condition":
                if target == "<END>":
                    new_cfg_block_connection[new_block][condition] = "<END>"
                elif target is not None:
                    new_cfg_block_connection[new_block][condition] = renumbering_dict.get(target, None)
                else:
                    new_cfg_block_connection[new_block][condition] = None
            else:
                new_cfg_block_connection[new_block][condition] = {}
                for branch, target_block in target.items():
                    if target_block == "<END>":
                        new_cfg_block_connection[new_block][condition][branch] = "<END>"
                    elif target_block is not None:
                        new_cfg_block_connection[new_block][condition][branch] = renumbering_dict.get(target_block, None)
                    else:
                        new_cfg_block_connection[new_block][condition][branch] = None
    keys = list(new_cfg_block_statements.keys())[:-1]
    for key in keys:
        block_length = len(new_cfg_block_statements[key + 1])
        block_start, block_end = new_cfg_block_ranges[key + 1]['range']
        curr_block_last_statement = new_cfg_block_statements[key][-1]
        next_block_first_statement = new_cfg_block_statements[key + 1][0]
        curr_block_last_line_number = new_cfg_block_ranges[key]['range'][1]
        next_block_first_line_number = new_cfg_block_ranges[key + 1]['range'][0]
        if block_length == 1 and block_start == block_end:
            if curr_block_last_statement == next_block_first_statement and curr_block_last_line_number == next_block_first_line_number:
                if 'iterator' in new_cfg_block_statements[key][-1] and 'iterator' in new_cfg_block_statements[key + 1][0]:
                    new_cfg_block_statements[key] = new_cfg_block_statements[key][:-1]
                    start_range, end_range = new_cfg_block_ranges[key]['range']
                    new_cfg_block_ranges[key]['range'] = [start_range, end_range - 1]
    return new_cfg_block_statements, new_cfg_block_ranges, new_cfg_block_connection

def generate_cfg_text(block_statements, next_block_connections, block_ranges):
    cfg_text = ""
    for block_index in block_ranges:
        cfg_text += f"\nBlock {block_index}:\n"
        block_range = block_ranges[block_index]["range"]
        start_line, end_line = block_range[0], block_range[1]
        total_lines = end_line - start_line + 1
        statements = block_statements[block_index]
        if len(statements) != total_lines:
            return ""
        cfg_text += "Statements:"
        for statement in statements:
            cfg_text += f"\n    {statement}"
        cfg_text += "\nNext:\n"
        try:
            next_blocks = next_block_connections[block_index]
            true_next = next_blocks["with_condition"]["true"]
            false_next = next_blocks["with_condition"]["false"]
            no_condition_next = next_blocks["no_condition"]
            if true_next is not None:
                if true_next == "<END>":
                    cfg_text += "    <END>\n"
                else:
                    cfg_text += f"    If True: Go to Block {true_next}\n"
            if false_next is not None:
                if false_next == "<END>":
                    cfg_text += "    <END>\n"
                else:
                    cfg_text += f"    If False: Go to Block {false_next}\n"
            if true_next and false_next:
                continue
            if not true_next and not false_next:
                if no_condition_next == "<END>":
                    cfg_text += "    <END>\n"
                else:
                    cfg_text += f"    Go to Block {no_condition_next}\n"
        except:
            cfg_text += "    <END>\n"
    return cfg_text

def get_error_block(lines):
    block_pattern = re.compile(r"Block:\s*(\d+)", re.IGNORECASE)
    for line in lines:
        block_match = block_pattern.search(line)
        if block_match:
            block_number = block_match.group(1)
            return block_number
    return None

def fetch_symbol_table(text_block):
    symbol_table_content = ""
    for text_block_line in text_block.split('\n'):
        lines_with_symbol_tables = re.findall(r'(?i)symbol table:?.*?{.*}', text_block_line)
        for line in lines_with_symbol_tables:
            start_index = line.find('{')
            end_index = line.rfind('}') + 1
            symbol_table_content = line[start_index:end_index]
    try:
        symbol_table = eval(symbol_table_content)
        return symbol_table
    except:
        return ""

def get_the_symbol_table(blocks_list):
    block_id_symbol_table = []
    for block in blocks_list:
        block_id = block['block_id']
        block_content = block['content']
        symbol_table = fetch_symbol_table(block_content)
        if symbol_table == "":
            block_id_symbol_table.append({"block_id": int(block_id), "symbol_table": {}})
        else:
            block_id_symbol_table.append({"block_id": int(block_id), "symbol_table": symbol_table})
    return block_id_symbol_table

def _extract_error_details(lines):
    error_type = None
    error_block = None
    is_error = False
    for index, line in enumerate(lines):
        if 'Is Error' in line and 'true' in line.lower():
            is_error = True
        if "Error Type" in line and is_error:
            if "None" in line:
                is_error = False
                break
            if "<class 'TypeError'>" in line:
                is_error = True
                error_type = "TypeError"
                error_block = get_error_block(lines[index:])
                break
            else:
                match = re.search(r"Error Type:\s*(\w+)", line)
                if match:
                    is_error = True
                    error_type = match.group(1).strip()
                    error_block = get_error_block(lines[index:])
                    break
    return error_type, error_block, is_error

def _extract_block_details(lines):
    blocks_list = []
    block_content = []
    block_started = False
    block_id = None
    pattern = r'^Block\s*:?\s*\d+\s*[:]?[ ]?$'
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            if block_started:
                blocks_list.append({"block_id": int(block_id), "content": '\n'.join(block_content)})
                block_content = []
            block_started = True
            block_id = int(line[line.find('Block') + 6:].split(':')[0])
        if block_started:
            block_content.append(line)
    if block_started:
        blocks_list.append({"block_id": int(block_id), "content": '\n'.join(block_content)})
    return blocks_list

def output_parser(output):
    keywords = ['Observation', 'evaluate', 'Error Type', '<END>']
    if not any(keyword in output for keyword in keywords):
        print("Output missing keywords: " + ", ".join(keywords))
        return [], ["", ""]
    block_execution_order = []
    error_type, error_block, is_error = _extract_error_details(output.split('\n'))
    blocks_list = _extract_block_details(output.split('\n'))
    if not blocks_list and not is_error and not error_block:
        return [], []
    if blocks_list:
        try:
            for block in blocks_list:
                block_execution_order.append(int(block['block_id']))
        except:
            return [], [], []
        try:
            blocks_symbol_table = get_the_symbol_table(blocks_list)
        except:
            blocks_symbol_table = []
    if is_error and error_block:
        return block_execution_order, [error_type, error_block, is_error], blocks_symbol_table
    elif is_error and not error_block:
        return block_execution_order, [error_type, "", is_error], blocks_symbol_table
    else:
        return block_execution_order, ["", "", is_error], blocks_symbol_table

def clean_response(obj):
    if isinstance(obj, dict):
        return {k: clean_response(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_response(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(clean_response(item) for item in obj)
    elif isinstance(obj, type):
        return str(obj.__name__)
    else:
        return obj

tracker = EmissionsTracker(project_name=r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\orca\src\orca\inference\utils.py")
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
