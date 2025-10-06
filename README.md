# EcoSustain Replication Study

Este repositório contém o estudo de replicação do EcoSustain, permitindo rodar e medir emissões de carbono e uso de recursos do sistema em artefatos Python.

## Estrutura do Repositório

- `scrapping-icse/`: contém o script `scrapping_icse.py`, que gera um CSV com os arquivos ICSE e se possuem artefato ou não.
- `run-dpy/`: scripts para rodar os artefatos com DPY.
- `implement-tools-into-artifact/`: scripts para instrumentar os artefatos com CodeCarbon e psutil.
- `artefatos/`: pasta onde você deve colocar todos os artefatos baixados ou clonados.

## Passo a Passo para Replicação

### 1. Gerar CSV dos Artefatos ICSE

Dentro da pasta `scrapping-icse`:

python scrapping_icse.py


Isso criará um CSV indicando todos os arquivos ICSE e se possuem artefato.

### 2. Organizar os Artefatos

Coloque todos os artefatos baixados ou clonados em uma pasta no seu computador.  
Ex.: `~/EcoSustain/artefatos/`.

### 3. Configurar DPY

Baixe o DPY no seu computador.  

Entre na pasta `run-dpy` e configure:

- `diretorio_dpy` → caminho para o DPY  
- `pasta_base` → caminho para a pasta com os artefatos  
- `pasta_saida_base` → pasta de saída para resultados

### 4. Instrumentar Artefatos com CodeCarbon e Psutil

Abra `implement-tools-into-artifact/implement-tools.py`.  

Altere a variável `artifacts_path` para o caminho da pasta com os artefatos.  

Execute o script:

python implement-tools.py


Isso vai adicionar CodeCarbon e psutil a todos os arquivos `.py` dentro de cada artefato.

### 5. Rodar os Artefatos

Execute qualquer arquivo `.py` dentro de um artefato.  

**IMPORTANTE:** Toda execução cria dois arquivos na pasta do artefato:

- `emissions.csv` → saída do CodeCarbon  
- `psutil_data.csv` → saída do Psutil

⚠️ Renomeie esses arquivos imediatamente após cada execução para evitar sobrescrita ou mistura com outros artefatos.

### 6. Refatorando Artefatos

README DESSA PARTE EM BREVE