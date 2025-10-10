import os
import pandas as pd

# === CONFIGURAÇÃO ===
artefato = "Discover-Data-Quality-With-RIOLU"
base_path = r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\refactoring\output-dpy"

# Caminhos dos arquivos JSON
path_original = os.path.join(base_path, f"{artefato}_implementation_smells.json")
path_refatorado = os.path.join(base_path, "saida_gemini_implementation_smells.json")

# Verificação de existência
for path in [path_original, path_refatorado]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Arquivo não encontrado: {path}")

# === LEITURA DOS JSONS ===
df_original = pd.read_json(path_original)
df_refatorado = pd.read_json(path_refatorado)

# Adiciona coluna "Package" como Original ou Refatorado
df_original["Package"] = "Original"
df_refatorado["Package"] = "Refatorado"

# === COMBINA OS DATAFRAMES ===
df = pd.concat([df_original, df_refatorado], ignore_index=True)

# === CONTAGEM DE SMELLS POR PACKAGE ===
print("\n📊 Contagem total de code smells por versão:")
print(df["Package"].value_counts(), "\n")

# === TABELA COMPARATIVA DE SMELLS ===
tabela = (
    df.groupby(["Smell", "Package"])
      .size()
      .unstack(fill_value=0)
      .reset_index()
)

print("📋 Tabela comparativa de code smells:")
print(tabela)

# === SALVA RESULTADO COMO CSV ===
output_csv = os.path.join(base_path, f"{artefato}_comparacao.csv")
tabela.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"\n✅ Tabela comparativa salva em: {output_csv}")
