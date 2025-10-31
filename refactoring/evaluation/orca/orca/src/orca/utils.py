from codecarbon import EmissionsTracker
import psutil
import csv
import os
import re
import json
import ast

# Start CodeCarbon tracker
tracker = EmissionsTracker(project_name=r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\orca\src\orca\utils.py")
tracker.start()

# System Metrics
def get_system_metrics():
    mem_start = psutil.virtual_memory().used / (1024**2)
    cpu_start = psutil.cpu_percent(interval=None)
    return mem_start, cpu_start

mem_start, cpu_start = get_system_metrics()

# Token Tracker
tokens_path = './tokens.json'

def load_tokens(path):
    try:
        with open(path, 'r') as infile:
            return json.load(infile)
    except FileNotFoundError:
        return {"input": 0, "output": 0, "total_tokens": 0, "estimate_cost": 0}
    except json.JSONDecodeError:
        return {"input": 0, "output": 0, "total_tokens": 0, "estimate_cost": 0}

def save_tokens(path, tokens):
    with open(path, 'w') as outfile:
        json.dump(tokens, outfile, indent=4)

def token_tracker(response):
    '''Track the tokens used in the response and update the tokens.json file.
       It caches tokens used in 'input' and 'output' and updates the total tokens used.
       Also, it estimates the cost of the API call based on the tokens used.
    '''
    tokens = load_tokens(tokens_path)
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens
    estimate_cost = ((prompt_tokens * 0.0010)/1000) + ((completion_tokens * 0.0020)/1000)
    tokens['input'] += prompt_tokens
    tokens['output'] += completion_tokens
    tokens['total_tokens'] += total_tokens
    tokens['estimate_cost'] += estimate_cost
    save_tokens(tokens_path, tokens)

# Code Scope Extraction
def extract_scopes_with_line_numbers(code):
    tree = ast.parse(code)
    categorized_scopes = {
        'for': [],
        'while': [],
        'if': [],
        'simple_statement': []
    }

    class ScopeExtractor(ast.NodeVisitor):
        def extract_scope(self, node, scope_type):
            start_line = node.lineno
            end_line = node.end_lineno
            scope_code = ast.get_source_segment(code, node)
            categorized_scopes[scope_type].append((start_line, end_line, scope_code))

        def visit_For(self, node):
            self.extract_scope(node, 'for')

        def visit_While(self, node):
            self.extract_scope(node, 'while')

        def visit_If(self, node):
            self.extract_scope(node, 'if')

        def generic_visit(self, node):
            if isinstance(node, ast.Expr) or isinstance(node, ast.Assign) or isinstance(node, ast.AugAssign):
                start_line = node.lineno
                end_line = node.end_lineno
                stmt_code = ast.get_source_segment(code, node)
                categorized_scopes['simple_statement'].append((start_line, end_line, stmt_code))
            super().generic_visit(node)

    ScopeExtractor().visit(tree)
    return categorized_scopes

def get_scope(code):
    for_loop = []
    while_loop = []
    if_statement = []
    simple_statements = []
    categorized_scopes = extract_scopes_with_line_numbers(code)
    for scope_type, scopes in categorized_scopes.items():
        for start_line, end_line, scope in scopes:
            if scope_type == 'for':
                for_loop.append([start_line, end_line])
            elif scope_type == 'while':
                while_loop.append([start_line, end_line])
            elif scope_type == 'if':
                if_statement.append([start_line, end_line])
            else:
                simple_statements.append([start_line, end_line])
    return for_loop, while_loop, if_statement, simple_statements

# Error and Symbol Table Extraction
def get_error_block(lines):
    block_pattern = re.compile(r"Block:\s*(\d+)", re.IGNORECASE)
    for line in lines:
        block_match = block_pattern.search(line)
        if block_match:
            return block_match.group(1)
    return None

def extract_symbol_table_content(text_block_line):
    match = re.search(r'(?i)symbol table:?.*?({.*})', text_block_line)
    if match:
        return match.group(1)
    return None

def fetch_symbol_table(text_block):
    symbol_table_content = ""
    for text_block_line in text_block.split('\n'):
        symbol_table_content = extract_symbol_table_content(text_block_line)
        if symbol_table_content:
            break
    if not symbol_table_content:
        return ""
    try:
        symbol_table = eval(symbol_table_content)
        return symbol_table
    except:
        return ""

def get_the_symbol_table(blocks_list):
    block_id_symbol_table = []
    for block in blocks_list:
        block_id = block['block_id']
        symbol_table = fetch_symbol_table(block['content'])
        block_id_symbol_table.append({"block_id": int(block_id), "symbol_table": symbol_table if symbol_table else {}})
    return block_id_symbol_table

# Output Parsing
def extract_error_details(lines, is_error):
    error_type = None
    error_block = None
    if "Error Type" in lines[0] and is_error:
            if "None" in lines[0]:
                is_error = False
            elif "<class 'TypeError'>" in lines[0]:
                error_type = "TypeError"
                error_block = get_error_block(lines)
            else:
                match = re.search(r"Error Type:\s*(\w+)", lines[0])
                if match:
                    error_type = match.group(1).strip()
                    error_block = get_error_block(lines)
    return error_type, error_block, is_error

def extract_block_details(lines):
    blocks_list = []
    block_content = []
    block_started = False
    block_id = None
    for line in lines:
        pattern = r'^Block\s*:?\s*\d+\s*[:]?[ ]?$'
        matches = re.findall(pattern, line)
        if matches:
            if block_started:
                blocks_list.append({"block_id": int(block_id), "content": '\n'.join(block_content)})
                block_content = []
            block_started = True
            block_id = int(line.split(':')[0].split()[-1])
        if block_started:
            block_content.append(line)
    if block_started:
        blocks_list.append({"block_id": int(block_id), "content": '\n'.join(block_content)})
    return blocks_list

def parse_block_execution_order(blocks_list):
    block_execution_order = []
    try:
        for block in blocks_list:
            block_execution_order.append(int(block['block_id']))
    except:
        return []
    return block_execution_order

def output_parser(output, response):
    token_tracker(response)
    keywords = ['Observation', 'evaluate', 'Error Type', '<END>']
    if not any(keyword in output for keyword in keywords):
        print("Output missing keywords: " + ", ".join(keywords))
        return [], [], []

    is_error = 'Is Error' in output and 'true' in output.lower()
    lines = output.split('\n')
    error_type, error_block, is_error = extract_error_details(lines, is_error)
    blocks_list = extract_block_details(lines)
    block_execution_order = parse_block_execution_order(blocks_list)
    blocks_symbol_table = get_the_symbol_table(blocks_list)

    if is_error and error_block:
        return block_execution_order, [error_type, error_block, is_error], blocks_symbol_table
    elif is_error and not error_block:
        return block_execution_order, [error_type, "", is_error], blocks_symbol_table
    else:
        return block_execution_order, ["", "", is_error], blocks_symbol_table

# Block and Statement Order Conversion
def get_gt_block_execution(ground_truth_blocks):
    execution_order = []
    for line in ground_truth_blocks:
        block_number = line['block']
        execution_order.append(int(block_number))
    return execution_order

def blocks_to_statements(block_range, block_execution):
    statement_execution = []
    for each_block in block_execution:
        block_range_for_current_block = block_range[str(each_block)]['range']
        start_range = block_range_for_current_block[0]
        end_range = block_range_for_current_block[1]
        for i in range(start_range, end_range+1):
            statement_execution.append(i)
    return statement_execution

def get_statements_from_blocks(block_range, prediction_blocks, ground_truth_execution_trace):
    gt_block_exe = get_gt_block_execution(ground_truth_execution_trace)
    pd_block_exe = prediction_blocks['block_execution']
    gt_statement_execution = blocks_to_statements(block_range, gt_block_exe)
    pd_statement_execution = blocks_to_statements(block_range, pd_block_exe)
    return pd_statement_execution, gt_statement_execution

# Save System Metrics
def save_system_metrics(file_path, mem_start, mem_end, cpu_start, cpu_end):
    file_exists = os.path.isfile(file_path)
    with open(file_path, "a", newline="") as csvfile:
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
save_system_metrics("psutil_data.csv", mem_start, mem_end, cpu_start, cpu_end)

tracker.stop()
