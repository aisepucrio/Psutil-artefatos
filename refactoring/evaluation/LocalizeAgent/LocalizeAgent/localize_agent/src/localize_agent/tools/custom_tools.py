from codecarbon import EmissionsTracker
import psutil
import csv
import os
import javalang
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import re

# Start CodeCarbon tracker
tracker = EmissionsTracker(project_name=r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\LocalizeAgent\localize_agent\src\localize_agent\tools\custom_tools.py")
tracker.start()

mem_start = psutil.virtual_memory().used / (1024**2)
cpu_start = psutil.cpu_percent(interval=None)


class CodeAnalysisInput(BaseModel):
    """Input for the tools that analyze Java source code."""
    source_code: str = Field(..., description="The Java source code to analyze.")


def clean_java_source_code(source_code: str) -> str:
    """Removes comments and import statements from Java source code."""
    return re.sub(r"//.*?$|/\*.*?\*/|^\s*import\s+.*?;", "", source_code, flags=re.DOTALL | re.MULTILINE)


class CountMethods(BaseTool):
    name: str = "CountMethods"
    description: str = "Counts the number of methods in a given Java source code."
    args_schema: Type[BaseModel] = CodeAnalysisInput

    def _run(self, source_code: str) -> str:
        try:
            cleaned_source_code = clean_java_source_code(source_code)
            tree = javalang.parse.parse(cleaned_source_code)
            method_count = 0
            for type_decl in tree.types:
                if hasattr(type_decl, 'methods'):
                    method_count += len(type_decl.methods)
            return f"The source code contains {method_count} methods."
        except Exception as e:
            return f"Error processing Java source code: {e}"


class VariableUsage(BaseTool):
    name: str = "VariableUsage"
    description: str = "Analyzes variable usage (fields vs. local variables) in the given Java source code."
    args_schema: Type[BaseModel] = CodeAnalysisInput

    def _run(self, source_code: str) -> str:
        try:
            cleaned_source_code = clean_java_source_code(source_code)
            tree = javalang.parse.parse(cleaned_source_code)
            variable_usage = {"global": 0, "methods": {}}

            for type_decl in tree.types:
                if hasattr(type_decl, 'fields'):
                    for field_decl in type_decl.fields:
                        variable_usage["global"] += len(field_decl.declarators)

                if hasattr(type_decl, 'methods'):
                    for method in type_decl.methods:
                        method_name = method.name
                        method_var_count = 0
                        for _, node in method.filter(javalang.tree.LocalVariableDeclaration):
                            method_var_count += len(node.declarators)
                        variable_usage["methods"][method_name] = method_var_count

            return f"Variable Usage: {variable_usage}"
        except Exception as e:
            return f"Error processing Java source code: {e}"


class FanInFanOutAnalysis(BaseTool):
    name: str = "FanInFanOutAnalysis"
    description: str = (
        "Analyzes methods in the Java source code and computes a naive fan-in/fan-out."
        " Fan-out = number of distinct methods that a method calls; "
        " Fan-in = number of times a method is called by other methods in the same file."
    )
    args_schema: Type[BaseModel] = CodeAnalysisInput

    def _run(self, source_code: str) -> str:
        try:
            cleaned_source_code = clean_java_source_code(source_code)
            tree = javalang.parse.parse(cleaned_source_code)
            method_calls = self._extract_method_calls(tree)
            fan_metrics = self._calculate_fan_metrics(method_calls)
            return self._format_fan_metrics(fan_metrics)
        except Exception as e:
            return f"Error processing Java source code: {e}"

    def _extract_method_calls(self, tree):
        method_calls = {}
        for type_decl in tree.types:
            if hasattr(type_decl, 'methods'):
                for method in type_decl.methods:
                    class_and_method = f"{type_decl.name}.{method.name}"
                    method_calls[class_and_method] = set()
                    for _, node in method.filter(javalang.tree.MethodInvocation):
                        method_calls[class_and_method].add(node.member)
        return method_calls

    def _calculate_fan_metrics(self, method_calls):
        fan_metrics = {key: {"fanIn": 0, "fanOut": 0} for key in method_calls.keys()}
        for method_key, calls in method_calls.items():
            fan_metrics[method_key]["fanOut"] = len(calls)
        for caller, called_methods in method_calls.items():
            for called_m in called_methods:
                possible_targets = [k for k in method_calls.keys() if k.endswith("." + called_m)]
                if len(possible_targets) == 1:
                    callee_key = possible_targets[0]
                    fan_metrics[callee_key]["fanIn"] += 1
        return fan_metrics

    def _format_fan_metrics(self, fan_metrics):
        lines = ["Fan-In / Fan-Out Analysis:"]
        for m_key, data in fan_metrics.items():
            lines.append(f"Method {m_key}: fanIn={data['fanIn']}, fanOut={data['fanOut']}")
        return "\n".join(lines)


class ClassCouplingAnalysis(BaseTool):
    name: str = "ClassCouplingAnalysis"
    description: str = (
        "By assessing class dependencies, this analysis provides insights into system"
        " modularity and potential areas for improving design quality. This tool identifies"
        " each class in the source file and detects the other classes it references."
    )
    args_schema: Type[BaseModel] = CodeAnalysisInput

    def _run(self, source_code: str) -> str:
        try:
            cleaned_source_code = clean_java_source_code(source_code)
            tree = javalang.parse.parse(cleaned_source_code)
            class_references = self._extract_class_references(tree)
            return self._format_class_references(class_references)
        except Exception as e:
            return f"Error processing Java source code: {e}"

    def _extract_class_references(self, tree):
        class_references = {}
        for type_decl in tree.types:
            if not hasattr(type_decl, 'name'):
                continue
            current_class_name = type_decl.name
            class_references[current_class_name] = set()
            for _, node in type_decl.filter(javalang.tree.Type):
                if node.name and (node.name != current_class_name):
                    class_references[current_class_name].add(node.name)
        return class_references

    def _format_class_references(self, class_references):
        lines = ["Class Coupling Analysis:"]
        for class_name, references in class_references.items():
            coupling_count = len(references)
            lines.append(
                f"  - Class {class_name} references: {sorted(list(references)) or 'None'}"
                f" -> Coupling = {coupling_count}"
            )
        return "\n".join(lines)


# Collect final system metrics and stop tracker
mem_end = psutil.virtual_memory().used / (1024**2)
cpu_end = psutil.cpu_percent(interval=None)

# Save psutil data to psutil_data.csv
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
