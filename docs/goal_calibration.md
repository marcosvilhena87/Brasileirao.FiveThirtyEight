# Calibração das intensidades de gols (2023–2025)

A metodologia mantém 2025 como teste fora da amostra. Para os 760 jogos de 2023–2024, ajusta dois lambdas positivos (mandante e visitante) por máxima verossimilhança **condicional ao resultado 1X2**. As probabilidades de resultado vêm do Elo-Davidson; a calibração de gols usa somente os placares passados.

O modelo toma o produto Poisson e normaliza *dentro de cada classe* (vitória da casa, empate, vitória de fora), de forma que cada classe preserve a massa 1X2 original. Otimização L-BFGS-B com limites de 0,2 a 5 gols e suporte truncado de 0 a 10.

## Avaliação de 2025

- **Log loss de placar exato:** média de `-ln P(gols_casa,gols_fora)`, usando a matriz condicional por confronto.
- **MAE gols casa/fora:** erro absoluto entre gols reais e esperança de gols do modelo.
- **MAE e viés da diferença de gols:** diferença real menos esperada; viés reportado como previsão menos realidade.
- **Variância da diferença de gols:** comparação entre variância preditiva agregada (variância condicional média + variância dos saldos esperados) e variância empírica.

Como referência, comparar a dois Poisson independentes fixos, estimados por médias de gols por mando dos jogos de 2023–2024. Esses valores não podem usar 2025.

O resultado de 2025 é um teste *prequential* (ratings atualizados apenas após os resultados de cada data). O modelo de intensidades **não é refitado** durante 2025.

## Reproduzir

```bash
python -m pip install -e ".[dev]"
python -m brasileirao538.goal_calibration
python -m pytest -q
```

## Limitações

Dois lambdas globais para todos os confrontos simplificam demais o padrão de gols, mesmo após a condição 1X2. A vantagem do modelo não é garantida: comparar numericamente com o baseline. Os testes garantem integridade técnica, não significância estatística. Para regressões futuras, experimentar intensidades ligadas à diferença de rating e parâmetros temporais, com validação separada. O motor Monte Carlo atual possui parâmetros default ilustrativos; a integração dos lambdas calibrados como defaults deve ocorrer somente após avaliação explícita.
