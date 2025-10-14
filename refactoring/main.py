import os
import subprocess
import json
from dotenv import load_dotenv
from collections import defaultdict
import shutil

# === Configuração ===
IA_PROVIDER = "gemini"  # escolha: "openai" ou "gemini"

# === Carrega variáveis de ambiente ===
load_dotenv()

# Gemini
if IA_PROVIDER == "gemini":
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# OpenAI
elif IA_PROVIDER == "openai":
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def reset_experimento():
    base_path = r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\refactoring"
    pastas = ["filtered-dpy", "output-dpy", "saida_gemini"]

    for pasta in pastas:
        caminho_pasta = os.path.join(base_path, pasta)

        if os.path.exists(caminho_pasta):
            for item in os.listdir(caminho_pasta):
                caminho_item = os.path.join(caminho_pasta, item)
                try:
                    if os.path.isfile(caminho_item) or os.path.islink(caminho_item):
                        os.remove(caminho_item)
                    elif os.path.isdir(caminho_item):
                        shutil.rmtree(caminho_item)
                except Exception as e:
                    print(f"Erro ao excluir {caminho_item}: {e}")
        else:
            print(f"A pasta '{caminho_pasta}' não existe.")


def roda_dpy(path):
    print("Iniciando análises com o DPy...")
    diretorio_dpy = r"C:\Users\PUC\Documents\DPy"
    caminho_saida = r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\refactoring\output-dpy"

    comando = [
        ".\\dpy", "analyze",
        "-i", f'"{path}"',
        "-o", f'"{caminho_saida}"'
    ]

    subprocess.run(" ".join(comando), cwd=diretorio_dpy, shell=True)
    print("Todas as análises foram concluídas.")


def filtra_arquivos(pasta):
    arquivos = []
    for root, dirs, files in os.walk(pasta):
        for f in files:
            if f.endswith("_implementation_smells.json"):
                arquivos.append(f)
    return arquivos


def prompt_instructions(pasta_output, arquivos):
    smell_list = []
    files = []

    for arquivo in arquivos:
        json_path = os.path.join(pasta_output, arquivo)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        grouped = defaultdict(list)
        for item in data:
            file_path = item["File"]
            grouped[file_path].append({
                "smell": item.get("Smell", ""),
                "Line": item.get("Line no", ""),
                "description": item.get("Details", "")
            })

        for file_path, file_smells in grouped.items():
            files.append(file_path)

            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
                lines = file_content.splitlines()

            snippets = []
            for item in file_smells:
                smell_name = item.get('smell', 'Unknown smell')
                smell_desc = item.get('description', 'No description')
                smell_lines = item.get('Line', 'No line info')

                if "-" in smell_lines:
                    start, end = [int(x.strip()) for x in smell_lines.split("-")]
                else:
                    start = end = int(smell_lines.strip())

                snippet = "\n".join(lines[start-1:end])

                snippets.append(
                    f"- Smell Type: {smell_name}\n"
                    f"- Description: {smell_desc}\n"
                    f"- Snippet of the Smell:\n{snippet}\n\n"
                )

            smell_report = f"""## Original Code\n{file_content}\n\n---\n\n## Detected Code Smells\n{''.join(snippets)}"""
            smell_list.append(smell_report)

    return files, smell_list


def gera_prompt(smell_list):
    prompts = []
    for smell in smell_list:
        prompt = f"""
        You are an advanced model specialized in **Python code refactoring**.

        Your goal is to transform the given code into a **cleaner, more maintainable, and more readable version** 
        while ensuring its **behavior remains identical**.

        ---

        {smell}

        ---

        ## Refactoring Guidelines
        - **Long method** → Extract methods to reduce complexity.

        ---

        ### Output Instructions
        1. Refactor the **entire codebase**, not just snippets.  
        2. Apply the most appropriate refactoring techniques from the list above.  
        3. Preserve the **original functionality** exactly.  
        4. Output **only the refactored code** — no explanations, no comments.  
        5. **DON'T EVER CREATE NEW CODE SMELLS AFTER REFACTORING THE CODE**
        6. Use meaningful names, consistent style, and keep functions/classes concise.  
        7. Treat this as production-quality Python code.  
        """
        prompts.append(prompt)
    return prompts


def remove_linhas(file):
    with open(file, "r", encoding="utf-8") as f:
        linhas = f.readlines()
    linhas = linhas[1:-1]
    with open(file, "w", encoding="utf-8") as f:
        f.writelines(linhas)


def filtrar_por_smell(caminho_arquivo, smell_tipo):
    with open(caminho_arquivo, "r", encoding="utf-8") as f:
        data = json.load(f)
    filtrado = [item for item in data if item.get("Smell") == smell_tipo]
    caminho = rf"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\refactoring\filtered-dpy\{os.path.basename(caminho_arquivo)}"
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(filtrado, f, indent=4, ensure_ascii=False)


# === MAIN PIPELINE ===
pasta_output = r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\refactoring\output-dpy"
nome_artefato = "dlisa"
artefato = fr"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\{nome_artefato}"
saida_gemini = r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\refactoring\saida_gemini"
filtered_output = r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\refactoring\filtered-dpy"

reset_experimento()
roda_dpy(artefato)

filtrar_por_smell(
    rf"{pasta_output}\{os.path.basename(artefato)}_implementation_smells.json",
    "Long method"
)

arquivos = filtra_arquivos(filtered_output)
files, smell_list = prompt_instructions(filtered_output, arquivos)
prompts = gera_prompt(smell_list)

total = len(prompts)
for i, prompt in enumerate(prompts, start=1):
    progresso = int((i / total) * 100)
    print(f"LLM PENSANDO com {IA_PROVIDER.upper()}... ({progresso}%)")

    if IA_PROVIDER == "gemini":
        resposta = genai.GenerativeModel("gemini-2.0-flash-lite").generate_content(prompt)
        output_text = resposta.text

    elif IA_PROVIDER == "openai":
        response = client.responses.create(
            model="gpt-5-mini",
            input=prompt
        )
        output_text = response.output_text

    # === CORREÇÃO: salva na estrutura relativa ao artefato ===
    caminho_original = files[i - 1]
    caminho_relativo = os.path.relpath(caminho_original, start=artefato)
    caminho_novo = os.path.join(saida_gemini, caminho_relativo)
    os.makedirs(os.path.dirname(caminho_novo), exist_ok=True)

    with open(caminho_novo, "w", encoding="utf-8") as f:
        f.write(output_text)
        print(f"Arquivo salvo em: {caminho_novo}")

    remove_linhas(caminho_novo)

roda_dpy(saida_gemini)
