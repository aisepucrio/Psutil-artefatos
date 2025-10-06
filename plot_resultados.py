import os
import re
import matplotlib.pyplot as plt

pasta_relatórios = "relatorios"

regex_cpu = re.compile(r'CPU médio:\s*([\d.]+)%')
regex_memoria = re.compile(r'Memória média usada:\s*([\d.]+)\s*MB')
regex_duracao = re.compile(r'Duração:\s*([\d.]+)\s*segundos')

cpu_medios = {}
memorias = {}
duracoes = {}

for nome_arquivo in os.listdir(pasta_relatórios):
    if nome_arquivo.endswith('.txt'):
        caminho_arquivo = os.path.join(pasta_relatórios, nome_arquivo)
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            conteudo = f.read()

            nome_projeto = nome_arquivo.replace(".txt", "").split('_')[0]

            match_cpu = regex_cpu.search(conteudo)
            match_mem = regex_memoria.search(conteudo)
            match_dur = regex_duracao.search(conteudo)

            if match_cpu and match_mem and match_dur:
                cpu_medios[nome_projeto] = float(match_cpu.group(1))
                memorias[nome_projeto] = float(match_mem.group(1))
                duracoes[nome_projeto] = float(match_dur.group(1))

def plot_bar(titulo, dados, ylabel):
    plt.figure()
    plt.bar(dados.keys(), dados.values())
    plt.title(titulo)
    plt.ylabel(ylabel)
    plt.xlabel("Projetos")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.grid(True)
    plt.show()

plot_bar(" CPU Médio por Projeto", cpu_medios, "Uso de CPU (%)")
plot_bar(" Memória Média Usada por Projeto", memorias, "Memória (MB)")
plot_bar(" Duração de Execução por Projeto", duracoes, "Tempo (segundos)")
