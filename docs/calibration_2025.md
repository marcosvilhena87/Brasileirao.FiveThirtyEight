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

## Diagnóstico numérico preliminar (2025)

Nas 380 partidas, o ganho médio de perda do Davidson em relação às frequências históricas 2023–24 foi de **0.022121 no Brier multiclasses** e **0.031780 no log loss** (baseline menos Davidson). A auditoria por blocos identificou **110 datas** com jogos.

| Evento e faixa de probabilidade | n | Previsto (média) | Observado |
|---|---:|---:|---:|
| Casa, 20–40% | 123 | 34.0% | 37.4% |
| Casa, 40–60% | 230 | 48.5% | 54.3% |
| Casa, 60–80% | 26 | 63.9% | 76.9% |
| Empate, 20–40% | 378 | 26.2% | 26.2% |
| Fora, 0–20% | 51 | 16.0% | 7.8% |
| Fora, 20–40% | 284 | 29.0% | 23.9% |
| Fora, 40–60% | 45 | 44.3% | 40.0% |

Faixas com apenas 1 ou 2 jogos foram omitidas desta tabela e permanecem no relatório do programa. O peso de empate concentra quase todas as previsões dessa classe numa só faixa: isso limita a resolução da análise de calibração.

Como verificação independente, um bootstrap por 110 blocos de datas, 10.000 réplicas e gerador pseudoaleatório separado indicou **intervalos percentis aproximados** de **[0.0069, 0.0372] para o ganho em Brier** e **[0.0110, 0.0526] para o ganho em log loss**. Esses intervalos são aproximações de conferência; a implementação Python usa 5.000 réplicas e outra sequência aleatória, de modo que seus limites exatos podem diferir.

**Interpretação:** a vantagem média contra o baseline histórico parece positiva também após a reamostragem de datas, mas há indícios de subestimação de mandantes e superestimação de visitantes. Não aplicar pós-calibração nos resultados de 2025 e continuar chamando a mesma amostra de teste independente.
