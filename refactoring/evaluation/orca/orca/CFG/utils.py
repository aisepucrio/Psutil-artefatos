from codecarbon import EmissionsTracker
import psutil
import csv
import os
import ast

def replace_code_in_file(filePath, code):
    with open(filePath, 'r') as file:
        content = file.read()
    indented_code = '\n'.join(['    ' + line for line in code.splitlines()])
    content = content.replace('TestString', indented_code)
    with open(filePath, 'w') as file:
        file.write(content)

def reset_file_content(filePath):
    content = """def testFun():
TestString"""
    with open(filePath, 'w') as file:
        file.write(content)

def get_block_ranges_and_statements(cfg, code):
    block_ranges = {}
    block_statements = {}
    is_processing_started = False

    for block in cfg.blocks:
        if block.label == "<entry:testFun>":
            is_processing_started = True
        if not is_processing_started:
            continue
        if not block.control_flow_nodes:
            continue
        if any(next_block.label == "<raise>" for next_block in block.next):
            continue

        block_line_numbers = extract_scope(block)
        if not block_line_numbers:
            continue

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

def extract_scope(block):
    scope_line_numbers = []
    for node in block.control_flow_nodes:
        instruction = node.instruction
        ast_node = instruction.node
        line_number = ast_node.lineno
        scope_line_numbers.append(line_number - 1)
    return scope_line_numbers

def _get_block_index(block, block_ranges):
    line_number_list = [node.instruction.node.lineno - 1 for node in block.control_flow_nodes]
    block_index = min(line_number_list)
    return block_index

def _get_next_block_index(block_ranges, target_block_lines):
    if not target_block_lines:
        return None
    min_line_number = min(target_block_lines)
    max_line_number = max(target_block_lines)
    for key, value in block_ranges.items():
        if value['range'] == [min_line_number, max_line_number]:
            return key
    return None

def _process_conditional_block(block, block_ranges):
    true_scope_block = []
    false_scope_block = []
    if block.branches:
        true_block = block.branches[True]
        false_block = block.branches[False]
        true_scope_block = [node.instruction.node.lineno - 1 for node in true_block.control_flow_nodes]
        false_scope_block = [node.instruction.node.lineno - 1 for node in false_block.control_flow_nodes]
    return true_scope_block, false_scope_block

def _process_no_condition_block(block):
    no_condition_block_lines = []
    for next_block in block.next:
        for node in next_block.control_flow_nodes:
            no_condition_block_lines.append(node.instruction.node.lineno - 1)
    return no_condition_block_lines

def get_block_connections(block_ranges, cfg):
    block_connections = {}
    is_processing_started = False

    for block in cfg.blocks:
        if block.label == "<entry:testFun>":
            is_processing_started = True
        if not is_processing_started:
            continue
        if not block.control_flow_nodes:
            continue
        if any(next_block.label == "<raise>" for next_block in block.next):
            continue

        block_index = _get_block_index(block, block_ranges)
        true_scope_block, false_scope_block = _process_conditional_block(block, block_ranges)
        no_condition_block_lines = _process_no_condition_block(block)

        true_next = _get_next_block_index(block_ranges, true_scope_block)
        false_next = _get_next_block_index(block_ranges, false_scope_block)
        no_condition_next = _get_next_block_index(block_ranges, no_condition_block_lines)

        if true_next and false_next:
            pass
        elif true_next and not false_next:
            false_next = "<END>"
        elif false_next and not true_next:
            true_next = "<END>"

        block_connections[block_index] = {
            "with_condition": {"true": true_next, "false": false_next},
            "no_condition": no_condition_next
        }

    return block_connections

def renumber_cfg_blocks(cfg_block_statements, cfg_block_range, cfg_block_connection):
    sorted_block_statements = {k: cfg_block_statements[k] for k in sorted(cfg_block_statements)}
    renumbering_dict = {old: new for new, old in enumerate(sorted_block_statements, start=1)}
    new_cfg_block_statements = {
        renumbering_dict[block_id]: statements for block_id, statements in sorted_block_statements.items()
    }

    sorted_block_ranges = {k: cfg_block_range[k] for k in sorted(cfg_block_range)}
    new_cfg_block_ranges = {
        renumbering_dict[block_id]: block_range for block_id, block_range in sorted_block_ranges.items()
    }

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

def map_execution_order_to_blockwise(cfg_block_range, execution_order, execution_trace):
    execution_blocks = []
    combined_execution_trace = []
    for index, line_number in enumerate(execution_order):
        try:
            trace_line_number = int(execution_trace[index]['line'])
            if trace_line_number == int(line_number):
                variable_state = execution_trace[index]['var_val']
                combined_execution_trace.append({"line_number": trace_line_number, "state": variable_state})
            else:
                combined_execution_trace.append({"line_number": int(line_number), "state": []})
        except Exception:
            combined_execution_trace.append({"line_number": int(line_number), "state": []})

    blocks_ranges = {
        block_name: [range_[0], range_[1]]
        for block_name, range_ in cfg_block_range.items()
    }
    last_block = None
    for entry in combined_execution_trace:
        line_number = entry['line_number']
        state = entry['state']
        for block_name, range_ in blocks_ranges.items():
            if range_[0] <= line_number <= range_[1]:
                if last_block == block_name:
                    execution_blocks[-1] = {"block": block_name, "state": state}
                else:
                    execution_blocks.append({"block": block_name, "state": state})
                    last_block = block_name
                break
    return execution_blocks

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

tracker = EmissionsTracker(project_name=r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\orca\dataset_builder\CFG\utils.py")
tracker.start()

mem_start = psutil.virtual_memory().used / (1024**2)
cpu_start = psutil.cpu_percent(interval=None)

csv_file = "psutil_data.csv"
file_exists = os.path.isfile(csv_file)
with open(csv_file, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(["file", "mem_start_MB", "mem_end_MB", "mem_diff_MB", "cpu_start_percent", "cpu_end_percent"])

    file_name = __file__
    mem_end = psutil.virtual_memory().used / (1024**2)
    cpu_end = psutil.cpu_percent(interval=None)

    writer.writerow([
        file_name,
        f"{mem_start:.2f}",
        f"{mem_end:.2f}",
        f"{mem_end - mem_start:.2f}",
        f"{cpu_start:.2f}",
        f"{cpu_end:.2f}"
    ])

tracker.stop()
