import os
import subprocess
from codecarbon import EmissionsTracker

tracker = EmissionsTracker()
tracker.start()

# Caminho para o dpy
diretorio_dpy = r"C:\Users\guicu\OneDrive\Documentos\prog\ecosustain\DPy\DPy\windows"

# Pasta onde estão os artefatos
pasta_base = r"C:\Users\guicu\OneDrive\Documentos\prog\ecosustain\artifact\artifacts"

# Pasta base para saída
pasta_saida_base = r"C:\Users\guicu\OneDrive\Documentos\prog\ecosustain\output-dpy"

subpastas = [p for p in os.listdir(pasta_base) if os.path.isdir(os.path.join(pasta_base, p))]
total = len(subpastas)

for i, nome_pasta in enumerate(subpastas, start=1):
    caminho_entrada = os.path.join(pasta_base, nome_pasta)
    caminho_saida = os.path.join(pasta_saida_base, nome_pasta)

    # Cria diretório de saída
    os.makedirs(caminho_saida, exist_ok=True)

    print(f"Analisando: {nome_pasta}")
    print(f"Progresso: ({i}/{total}) {int(i/total*100)}%")

    comando = [
        ".\\dpy", "analyze",
        "-i", f'"{caminho_entrada}"',
        "-o", f'"{caminho_saida}"'
    ]

    subprocess.run(" ".join(comando), cwd=diretorio_dpy, shell=True)

print("Todas as análises foram concluídas.")

tracker.stop()
