# Teste de intensidades de gols dependentes da forca relativa

A diferenca Elo ajustada ao mando e expressa por `x = ln(P_1/P_2)`, onde `P_1` e `P_2` sao as probabilidades de vitoria mandante e visitante do Davidson. Definimos:

```text
lambda_casa(jogo) = lambda_casa_base * exp(beta * x)
lambda_fora(jogo) = lambda_fora_base * exp(-beta * x)
```

A inclinacao `beta` controla o impacto da forca relativa no numero de gols. `beta=0` reproduz o Poisson global anterior; as marginais 1X2 permanecem **exatamente iguais** no modelo condicional. Assim, o teste compara especificamente a qualidade das distribuicoes de placares e saldo, sem alterar o modelo de vencedor.

## Validacao temporal

1. **2023:** estimar lambdas globais e beta somente com placares de 2023.
2. **2024:** avaliar ambos os modelos em 380 jogos com os parametros de 2023 fixos. Nao usar os gols de 2024 para ajustar esse teste.
3. **2023+2024:** reestimar os lambdas e beta com 760 jogos.
4. **2025:** aplicar os parametros finais sem reajuste, para avaliar NLL exato, MAE de gols, MAE e vies do saldo e variancia preditiva do saldo versus observada.

```bash
python -m brasileirao538.relative_goals
python -m pytest -q
```

**Cuidado:** as probabilidades Elo-Davidson por temporada seguem o protocolo existente, cuja calibracao de K/mando/empate usa a temporada 2024. Portanto, a comparacao de 2024 para o componente de gols nao e uma validacao integralmente independente de todo o pipeline. O holdout 2025 permanece protegido. Os parametros do simulador so devem mudar se a nova versao superar o Poisson global de forma robusta no teste; resultados estatisticos ainda devem ser interpretados com cautela.

## Resultados observados no GitHub Actions

Inclinação estimada com 2023: `beta=0.153205`. Inclinação reestimada em 2023–2024: `beta=0.127548`. Lambdas base em 2023–2024: casa `1.29610`, fora `1.15824`.

| Ano e métrica | Poisson global | Poisson variável Elo |
|---|---:|---:|
| 2024: log loss placar exato ↓ | 2.72197 | **2.72106** |
| 2024: MAE saldo ↓ | 1.13936 | **1.13369** |
| 2025: log loss placar exato ↓ | 2.84516 | **2.82744** |
| 2025: MAE gols casa ↓ | 0.95770 | **0.95323** |
| 2025: MAE gols fora ↓ | 0.79219 | **0.79022** |
| 2025: MAE saldo ↓ | 1.23931 | **1.23072** |
| 2025: viés saldo (previsto − observado) | −0.24197 | **−0.20629** |
| 2025: variância saldo prevista | 2.44509 | **2.53213** |
| 2025: variância saldo observada | 2.88600 | 2.88600 |

Em 2025, melhoria relativa de **0,62% no log loss exato** e **0,69% no MAE saldo**. Ganho pequeno; não há intervalo de confiança da diferença. Ainda existe subdispersão do saldo. O modelo será mantido **experimental**, sem substituir automaticamente os lambdas globais do motor de Monte Carlo.

[Execução reproduzível](https://github.com/marcosvilhena87/Brasileirao.FiveThirtyEight/actions/runs/41223386346).
