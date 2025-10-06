import os

def implementa_codecarbon(caminho_pasta):
    import_codecarbon = [
        'from codecarbon import EmissionsTracker\n',
        '\n',
        'tracker = EmissionsTracker()\n',
        'tracker.start()\n',
        '\n'
    ]
    tracker_stop = 'tracker.stop()\n'

    for raiz, _, arquivos in os.walk(caminho_pasta):
        for nome_arquivo in arquivos:
            if nome_arquivo.endswith('.py'):
                caminho_arquivo = os.path.join(raiz, nome_arquivo)

                # TENTA LER COM UTF-8, CAI PARA LATIN1 SE DER ERRO
                try:
                    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                        conteudo = f.readlines()
                    encoding_usado = 'utf-8'
                except UnicodeDecodeError:
                    with open(caminho_arquivo, 'r', encoding='latin1') as f:
                        conteudo = f.readlines()
                    encoding_usado = 'latin1'

                if any('EmissionsTracker' in linha for linha in conteudo):
                    print(f"[SKIP] {caminho_arquivo} já contém o CodeCarbon.")
                    continue

                indice_insercao = 0
                for i, linha in enumerate(conteudo):
                    if not linha.strip().startswith('#') and linha.strip():
                        indice_insercao = i
                        break

                conteudo[indice_insercao:indice_insercao] = import_codecarbon

                if not any('tracker.stop()' in linha for linha in conteudo):
                    if not conteudo[-1].endswith('\n'):
                        conteudo[-1] += '\n'
                    conteudo.append('\n' + tracker_stop)

                # ESCREVE USANDO A MESMA CODIFICAÇÃO QUE FOI LIDA
                with open(caminho_arquivo, 'w', encoding=encoding_usado) as f:
                    f.writelines(conteudo)

                print(f"[OK] CodeCarbon adicionado em: {caminho_arquivo}")

# CAMINHO BASE PARA A PASTA COM OS ARTEFATOS
pasta_artefatos = os.path.abspath(r"C:\Users\guicu\OneDrive\Documentos\prog\ecosustain\artifact\artifacts")
implementa_codecarbon(pasta_artefatos)
