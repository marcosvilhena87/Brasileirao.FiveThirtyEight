# Modelo de probabilidades 1X2 — Davidson

Para a diferença de rating ajustada pelo mando `d = R_casa - R_fora + H`, construímos `x = d * ln(10) / 400`. Os três pesos do modelo Davidson são `exp(x/2)`, `nu` e `exp(-x/2)`, correspondendo a vitória da casa, empate e vitória de fora. Após normalização, somam exatamente 1 (dentro de precisão numérica).

O modelo usa **rating Elo** como variável explicativa. Suas probabilidades 1X2 não são a mesma coisa que a pontuação Elo esperada usada para atualizar ratings.

## Protocolo

- 2023: estabelece rating inicial da temporada seguinte.
- 2024: `K` e vantagem de mando previamente selecionados pelo protocolo Elo; peso de empate `nu` escolhido por menor log loss na grade `0.25, 0.5, 0.75, 1, 1.25, 1.5`.
- 2025: teste fora da amostra; sem selecionar ou ajustar hiperparâmetros com seus resultados.
- Jogos na mesma data utilizam um snapshot de ratings anterior aos placares daquele dia.
- Baselines: frequências de 1X2 de 2023 e 2024 combinadas, e frequência uniforme de 1/3 por resultado.

## Métricas

**Brier multiclasses:** média da soma dos três erros quadráticos `(p_i - y_i)^2`, escala de 0 a 2. Esta convenção não divide por três.

**Log loss:** média de `-ln(p_resultado_real)`. Menor valor indica desempenho melhor para ambas as métricas. Log loss pune especialmente previsões excessivamente confiantes quando estão erradas.

## Reproduzir

```bash
python -m pip install -e ".[dev]"
python -m brasileirao538.experiment1x2
python -m pytest -q
```

Amostra pequena para calibração: verificar estabilidade por outros anos e calibração probabilística por faixas antes de usar como motor de Monte Carlo.
