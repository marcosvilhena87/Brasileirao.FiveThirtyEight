# Brasileirão 2025 — dados e baseline Elo

## Origem e uso

Dataset: [OpenFootball / South America — Brazil 2025](https://github.com/openfootball/south-america/blob/master/brazil/2025_br1.txt). Fonte publica declarada como CC0 pelo projeto OpenFootball; conferir os termos da fonte ao redistribuir. Arquivo derivado: `data/processed/brasileirao_2025.csv`.

Foram extraidos 380 placares, 20 clubes e 38 rodadas; os nomes foram normalizados em portugues (por exemplo, `CR Flamengo` -> `Flamengo` e `SC Recife` -> `Sport`). O CSV conserva data real, numero original da rodada, gols e mando.

**Ressalva:** o arquivo de origem nao apresenta horario individual em todas as partidas. Por isso, a atualizacao Elo aplica a mesma fotografia de ratings a todos os jogos de cada data e soma seus deltas antes de passar ao dia seguinte. Evita usar o resultado de uma partida como informacao supostamente previa de outra no mesmo dia.

## Execucao

```bash
python -m pip install -e ".[dev]"
python -m brasileirao538.season2025
python -m pytest -q
```

A classificacao tem pontos, vitorias, saldo e gols marcados. Criterios de desempate adicionais (como confronto direto) ainda nao foram implementados. O procedimento reproduz pontuacoes finais esperadas (Flamengo 79; Palmeiras 76).

## Interpretacao

Ratings no fim de 2025 sao **retrospectivos**, porque utilizam os resultados da temporada. Como todos os times iniciam com 1.500, nao ha priors historicos. O K=20 e a vantagem de mando de 80 pontos sao hipoteses e nao calibracao. Esta analise nao estima chances de titulo em datas passadas nem validacao fora da amostra.

## Proximos passos

- Importar temporadas anteriores para estabelecer priors.
- Calibrar K, mando e regressao a media em janelas temporais separadas.
- Medir escore pre-jogo, Brier e log loss quando houver um modelo 1X2.
- Incluir identificador e horario oficiais para tratar jogos simultaneos com maior precisao.
