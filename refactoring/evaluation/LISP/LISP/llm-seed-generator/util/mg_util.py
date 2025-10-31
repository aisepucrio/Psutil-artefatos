from codecarbon import EmissionsTracker
import psutil
import csv
import os
import json
import sys
from queue import Queue
from typing import Dict, Any, List

OBTAIN_ALL_PARAM_INFO = -1


def return_type(mg: dict) -> str:
    return mg["returnTypeName"]


def method_name(mg: dict) -> str:
    return mg["methodName"]


def code_block(mg: dict) -> str:
    return mg["code"]


def _process_node(
    types: Dict[str, Dict],
    typ: str,
    mg_nodes: dict,
    q2: Queue,
    current_layer: int,
    layer: int,
    max_sub_num: int,
    s: set,
) -> None:
    if typ not in s:
        s.add(typ)
        t_node = mg_nodes.get(typ)
        if not t_node:
            types[typ] = types.get(typ, {"__is_jdk_type__": True})
        else:
            types[typ] = types.get(typ, {})
            types[typ].update({"classType": t_node.get("classType")})
            node = types[typ]

            if t_node.get("classType") == "class":
                node["constructors"] = node.get("constructors", {})
                node["constructors"].update(t_node.get("constructors", {}))

            _process_sub_classes(node, t_node, "subClassName", q2, current_layer, layer, max_sub_num)
            _process_sub_classes(node, t_node, "implementedClassName", q2, current_layer, layer, max_sub_num)
            _process_sub_classes(node, t_node, "subInterfaceName", q2, current_layer, layer, max_sub_num)

            if t_node.get("fields"):
                for field_t in t_node["fields"].values():
                    q2.put_nowait(field_t)
            if t_node.get("constructors") and t_node.get("classType") == "class":
                for constructor_params in t_node["constructors"].values():
                    for param_t in constructor_params.values():
                        q2.put_nowait(param_t)


def _process_sub_classes(
    node: Dict[str, Any],
    t_node: Dict[str, Any],
    key: str,
    q2: Queue,
    current_layer: int,
    layer: int,
    max_sub_num: int,
) -> None:
    if t_node.get(key) and current_layer < layer:
        size = min(max_sub_num, len(t_node.get(key, [])))
        node[key] = node.get(key, {})
        for i, sub_class_t in enumerate(t_node[key]):
            if i >= max_sub_num:
                break
            if not node.get(key) or not node[key].get(f"{key[:-4]}{i}"):
                node[key][f"{key[:-4]}{max_sub_num - size + i}"] = sub_class_t
                q2.put_nowait(sub_class_t)


def parameters(mg: dict, layer: int = OBTAIN_ALL_PARAM_INFO, max_sub_num: int = sys.maxsize) -> Dict[str, Dict]:
    types: Dict[str, Dict] = {}
    q1, q2, current_layer = Queue(), Queue(), 0
    s = set()
    [q1.put_nowait(param_type) for param_type in mg["parameters"].values()]

    while not q1.empty():
        typ = q1.get_nowait()
        if typ in mg["nodes"]:
            _process_node(types, typ, mg["nodes"], q2, current_layer, layer, max_sub_num, s)
        if q1.empty():
            if layer != OBTAIN_ALL_PARAM_INFO and current_layer >= layer:
                return types
            current_layer += 1
            while not q2.empty():
                q1.put_nowait(q2.get_nowait())
    return types


def parameter_info(mg: dict, layer: int = OBTAIN_ALL_PARAM_INFO, max_sub_num: int = sys.maxsize) -> str:
    required_params = parameters(mg, layer=layer, max_sub_num=max_sub_num)
    result = ""
    for typ, infos in required_params.items():
        if infos.get("__is_jdk_type__"):
            result += f"  - {typ}: a jdk-builtin type or cannot be parsed\n"
        else:
            class_type = infos.get("classType", "Unknown")
            result += f"  - {class_type}: {typ}\n"
            for key in ["subClassName", "subInterfaceName"]:
                if infos.get(key):
                    result += f"    - {key.replace('Name', 's')} name:\n"
                    for name in infos[key].values():
                        result += f"        - {name}\n"
            if infos.get("constructors"):
                result += "    - Constructors:\n"
                for signature, params in infos["constructors"].items():
                    result += f"        - {signature}: {params}\n"
    return result


def _save_psutil_data(file_path: str, data: List[str]) -> None:
    with open(file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not os.path.isfile(file_path):
            writer.writerow(["file", "mem_start_MB", "mem_end_MB", "mem_diff_MB", "cpu_start_percent", "cpu_end_percent"])
        writer.writerow(data)


if __name__ == "__main__":
    tracker = EmissionsTracker(project_name=r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\LISP\llm-seed-generator\util\mg_util.py")
    tracker.start()

    mem_start = psutil.virtual_memory().used / (1024**2)
    cpu_start = psutil.cpu_percent(interval=None)

    with open("../graph.json", "r") as file:
        mg1 = json.load(file)
    result1 = parameter_info(mg1, layer=5, max_sub_num=3)
    print(result1)

    mem_end = psutil.virtual_memory().used / (1024**2)
    cpu_end = psutil.cpu_percent(interval=None)

    csv_file = "psutil_data.csv"
    _save_psutil_data(csv_file, [
        __file__,
        f"{mem_start:.2f}",
        f"{mem_end:.2f}",
        f"{mem_end - mem_start:.2f}",
        f"{cpu_start:.2f}",
        f"{cpu_end:.2f}"
    ])

    tracker.stop()
