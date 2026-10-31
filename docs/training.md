# Protocolo experimental Elo: 2023–2025

## Separacao temporal

- **2023 (treino/inicializacao):** construir os ratings finais a partir das partidas da temporada.
- **2024 (validacao):** selecionar `K` e vantagem de mando pelo erro quadratico medio de pontuacao Elo esperado antes dos jogos, usando os ratings herdados de 2023 com regressao a media.
- **2025 (holdout):** aplicar **sem reajustar parametros**, com ratings herdados de 2024, e medir o erro fora da amostra.

Avaliacao 2025 e *prequential*: o modelo prediz cada partida com dados anteriores e, depois que o resultado e conhecido, atualiza para a proxima data. Isso nao constitui previsao de toda a temporada em janeiro nem Brier score 1X2.

## Fonte

OpenFootball (arquivos `brazil/2023_br1.txt`, `2024_br1.txt` e `2025_br1.txt`), com resultados, mandos e datas reais. CSVs padronizados em `data/processed/`. Normalizacao dos identificadores mantida entre temporadas.

## Parametros e metrica

Grade: `K=(10,20,30,40)`, `home_advantage=(0,50,80,100)`; `initial_rating=1500`, `scale=400`, regressao anual `0.75`. Sao escolhas iniciais de modelagem. Objetivo na validacao:

```text
MSE = media[(S_mandante - E_mandante)^2]
S = 1 vitoria, 0.5 empate, 0 derrota
```

Nao equivale a acuracia de 1X2 nem a probabilidade de titulo. Times novos na Serie A entram com 1500, sem informacao de sua campanha anterior. Jogos do mesmo dia usam os ratings pre-dia por ausencia de horarios confiaveis em todos os jogos.

## Reproduzir

```bash
python -m pip install -e ".[dev]"
python -m brasileirao538.training
python -m pytest -q
```

## Limites e evolucoes

Apenas duas temporadas de treinamento/validacao: estimativas de parametros tem alta incerteza. Nao usar resultados de 2025 para alterar parametros, nem selecionar modelo depois de ver o teste. Expandir historico, usar validacao rolling-origin, avaliar previsao de empates e calibrar probabilidades 1X2 antes de simular Monte Carlo.
