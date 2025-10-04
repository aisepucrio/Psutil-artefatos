# Monitoramento de Projetos Python com PSUTIL

Ferramenta para analisar scripts Python, monitorando uso de CPU, memória, estrutura de arquivos e desempenho, gerando relatórios automáticos.

## Funcionalidades

**Informações do Sistema**

   * RAM disponível
   * Espaço em disco livre
   * Uso de CPU atual
   * Núcleos físicos e lógicos
   * Outros ...


## Como Usar

1. **Instale a dependência**

```bash
pip install psutil matplotlib
```

2. **Coloque os projetos na pasta desejada**
   Cada projeto deve conter o arquivo Python que será monitorado.

3. **Edite na função `def main` se necessário**
   Ajuste caminhos e scripts de cada projeto.

4. **Execute o script principal**

```bash
python ferramenta.py
```



## Saída no Terminal

Durante a execução, o terminal mostrará mensagens de acompanhamento para cada artefato:

```
============================================================
 Analisando: clozeMaster
============================================================
==> Relatório salvo em: relatorios\clozeMaster_relatorio_20251004_170242.txt

============================================================
 Analisando: Discover-Data-Quality-With-RIOLU
============================================================
==> Relatório salvo em: relatorios\Discover-Data-Quality-With-RIOLU_relatorio_20251004_170244.txt

============================================================
 Analisando: dlisa
============================================================
==> Relatório salvo em: relatorios\dlisa_relatorio_20251004_170743.txt

============================================================
 Analisando: UNLEASH
============================================================
==> Relatório salvo em: relatorios\UNLEASH_relatorio_20251004_170745.txt
```
* Indica o artefato sob análise 
* Informa o caminho do relatório gerado armazenado em `relatorios\`


## Saída no Arquivo de Relatório

O conteúdo completo do relatório (`.txt`) inclui informações do sistema, estrutura do projeto e desempenho do script. 

**Exemplo: Artefato `clozeMaster`**
```
 Projeto: clozeMaster

 Informações do Sistema:
 RAM disponível: 3395.36 MB
 Espaço em disco disponível: 381.47 GB
 CPU atual: 18.5%
 Núcleos físicos: 10 | Lógicos: 12

 Total de arquivos: 30476
 Tamanho total: 1762879.83 KB

 Duração: 0.50 segundos
 CPU médio: 20.30%
 Memória média usada: 12712.42 MB
Erro:
Traceback (most recent call last):
  File "C:\Users\gabriele\Desktop\artefatos\clozeMaster\main.py", line 7, in <module>
    from tqdm import tqdm
ModuleNotFoundError: No module named 'tqdm'

```

* Permite verificar métricas completas e erros
* Útil para auditoria ou comparação de desempenho entre projetos


5. **Execute o script de visualização**

```bash
python plot_resultados.py
```

## Visualização de Dados com Gráficos

Um script adicional lê todos os relatórios na pasta `relatorios/` e gera gráficos comparativos para cada projeto:

* **CPU Médio por Projeto**
* **Memória Média Usada por Projeto**
* **Duração de Execução por Projeto**


## Observações

* Verifique se os scripts existem nos caminhos especificados; caso contrário, o relatório registrará erro.
* Projetos grandes podem gerar relatórios extensos.
* Ferramenta adaptável a qualquer projeto Python.

