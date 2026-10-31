# Calibração 1X2 e incerteza — Brasileirão 2025

## Objetivo

Auditar probabilidades de vitória da casa, empate e vitória de fora usando as 380 partidas de 2025, preservadas fora do ajuste de parâmetros. O modelo Davidson usa K, mando e peso de empate selecionados somente em 2024.

## Diagrama de confiabilidade

Para cada classe separadamente, agrupar as probabilidades em cinco intervalos fixos: [0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1]. Em cada faixa informar quantidade de partidas, média da probabilidade prevista e proporção empírica de ocorrências. Faixas vazias são omitidas. A frequência observada pode diferir da prevista por ruído amostral; com poucos jogos o resultado é impreciso.

**Não ajustar a calibração usando os mesmos resultados de 2025** e depois apresentar a mesma temporada como teste independente.

## Intervalos de confiança

Definir ganho individual como perda do baseline de frequências históricas 2023–24 **menos** perda do Davidson. Ganho positivo favorece o modelo. Reamostrar **datas** com reposição (e levar todas as partidas do dia escolhido), preservando uma parte da dependência entre jogos no mesmo dia; calcular médias ponderadas pela quantidade de partidas nos blocos sorteados. Usar 5.000 réplicas, semente 2025 e intervalo percentil bilateral de 95%.

O procedimento estima a incerteza empírica da diferença média de perdas em 2025, condicionado aos parâmetros já selecionados. Não modela de forma completa dependência temporal entre diferentes dias, clubes compartilhados, nem incerteza da etapa de treinamento. Um intervalo que exclua zero é evidência exploratória, **não garantia de vantagem futura**.

## Execução

```bash
python -m pip install -e ".[dev]"
python -m brasileirao538.calibration
python -m pytest -q
```

O código foi separado em `src/brasileirao538/calibration.py`; testes verificam cobertura de 380 confrontos por classe e reprodutibilidade do bootstrap.

## Próxima validação

Adicionar rolling-origin com mais temporadas, bootstrap temporal com blocos de dias consecutivos, reliability diagrams e análise de empates. Considerar métricas de calibração por classe e por faixa acompanhadas de contagens e incerteza; evitar leituras causais.
