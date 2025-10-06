import os
import psutil
import subprocess
import time
from datetime import datetime

def info_sistema():
    linhas = []
    linhas.append(" Informações do Sistema:")
    linhas.append(f" RAM disponível: {psutil.virtual_memory().available / (1024**2):.2f} MB")
    linhas.append(f" Espaço em disco disponível: {psutil.disk_usage('/').free / (1024**3):.2f} GB")
    linhas.append(f" CPU atual: {psutil.cpu_percent(interval=1)}%")
    linhas.append(f" Núcleos físicos: {psutil.cpu_count(logical=False)} | Lógicos: {psutil.cpu_count()}")
    return "\n".join(linhas)

def explorar_estrutura_projeto(caminho_projeto):
    total_arquivos = sum(len(files) for _, _, files in os.walk(caminho_projeto))
    total_tamanho = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(caminho_projeto)
        for filename in filenames
    )
    linhas = []
    linhas.append(f" Total de arquivos: {total_arquivos}")
    linhas.append(f" Tamanho total: {total_tamanho / 1024:.2f} KB")
    return "\n".join(linhas)

def monitorar_execucao(script, caminho_projeto):
    uso_cpu = []
    uso_memoria = []

    print(f"\n============================================================")
    print(f" Analisando: {os.path.basename(caminho_projeto)}")
    print(f"============================================================")

    try:
        processo = subprocess.Popen(
            ["python", script],
            cwd=caminho_projeto,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        while processo.poll() is None:
            uso_cpu.append(psutil.cpu_percent(interval=0.5))
            uso_memoria.append(psutil.virtual_memory().used / (1024**2))
        stdout, stderr = processo.communicate()

        if len(uso_cpu) == 0:
            raise RuntimeError("O script não executou corretamente ou foi muito rápido.")

        desempenho = [
            f" Duração: {len(uso_cpu)*0.5:.2f} segundos",
            f" CPU médio: {sum(uso_cpu)/len(uso_cpu):.2f}%",
            f" Memória média usada: {sum(uso_memoria)/len(uso_memoria):.2f} MB",
        ]

        if stderr:
            desempenho.append(f"Erro:\n{stderr.decode()}")

    except Exception as e:
        desempenho = [f"Falha na execução: {str(e)}"]

    return "\n".join(desempenho)

def salvar_relatorio(nome_arquivo, conteudo):
    caminho = os.path.join("relatorios", nome_arquivo)
    os.makedirs("relatorios", exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(conteudo)
    print(f"==> Relatório salvo em: {caminho}")

def main():
    projetos = [
        {
            "nome": "clozeMaster",
            "caminho": "clozeMaster",
            "script": "main.py"
        },
        {
            "nome": "Discover-Data-Quality-With-RIOLU",
            "caminho": "Discover-Data-Quality-With-RIOLU",
            "script": "src/riolu_example.py"
        },
        {
            "nome": "dlisa",
            "caminho": "dlisa",
            "script": "examples/1_classification/train_classification.py"
        },
        {
            "nome": "UNLEASH",
            "caminho": "UNLEASH",
            "script": "example/example.py"
        }
    ]

    for projeto in projetos:
        estrutura = explorar_estrutura_projeto(projeto["caminho"])
        desempenho = monitorar_execucao(projeto["script"], projeto["caminho"])
        conteudo = f" Projeto: {projeto['nome']}\n\n{info_sistema()}\n\n{estrutura}\n\n{desempenho}"
        nome_arquivo = f"{projeto['nome']}_relatorio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        salvar_relatorio(nome_arquivo, conteudo)

if __name__ == "__main__":
    main()
