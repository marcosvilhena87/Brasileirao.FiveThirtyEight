# Auditoria Elo — avaliação temporal 2025

## Protocolo

Partidas: 380 de 2023, 380 de 2024 e 380 de 2025. Modelo conforme `src/brasileirao538/training.py` na main anterior à auditoria.

- 2023: construir ratings iniciando em 1500.
- 2024: selecionar K (10/20/30/40) e vantagem de mando (0/50/80/100) pelo MSE da pontuação Elo prevista.
- 2025: *holdout* sem reajustar hiperparâmetros; incorporar cada resultado **depois** da previsão, processando jogos simultâneos por data.
- Regressão anual: retenção de 75% do desvio em relação a 1500; recém-promovidos entram com 1500.
- O alvo é **escore Elo de 0, 0,5 ou 1**, não probabilidade de resultado 1X2.

## Resultado 2025 (380 partidas)

| Modelo | MSE | MAE |
|---|---:|---:|
| Elo selecionado em 2024 (K=30, mando=80) | 0.155209 | 0.345953 |
| Referência fixa histórica 2023–2024 (esperança 0.601974) | 0.168164 | 0.369200 |
| Referência neutra fixa (esperança 0.5) | 0.184868 | 0.369737 |

Redução relativa de MSE: **7,70% ante a referência histórica** e **16,04% ante 0,5**. Comparação é descritiva: ainda não foi estimada incerteza inferencial nem significância estatística.

## Estabilidade da grade em 2024 (MSE)

| K | Mando 0 | Mando 50 | Mando 80 | Mando 100 |
|---|---:|---:|---:|---:|
| 10 | 0.176202 | 0.165891 | 0.164689 | 0.165800 |
| 20 | 0.174471 | 0.163945 | 0.162544 | 0.163497 |
| 30 | 0.174476 | 0.163767 | **0.162181** | 0.162981 |
| 40 | 0.175259 | 0.164378 | 0.162619 | 0.163279 |

A região K=20–40, mando=80 produz resultados próximos. Mando=0 apresenta erro muito superior nessa grade. **Não se deve selecionar a melhor configuração por 2025**, pois isso contaminaria o holdout.

## Erros por mês em 2025

| Mês | Jogos | MSE Elo | MSE neutro 0,5 |
|---|---:|---:|---:|
| Março | 10 | 0.079025 | 0.125000 |
| Abril | 50 | 0.140136 | 0.170000 |
| Maio | 41 | 0.158223 | 0.195122 |
| Junho | 15 | 0.215602 | 0.200000 |
| Julho | 42 | 0.154307 | 0.184524 |
| Agosto | 51 | 0.170780 | 0.186275 |
| Setembro | 34 | 0.168122 | 0.176471 |
| Outubro | 54 | 0.154353 | 0.203704 |
| Novembro | 63 | 0.149133 | 0.174603 |
| Dezembro | 20 | 0.141193 | 0.212500 |

A exceção de junho inclui somente 15 jogos. Não interpretar comparações mensais como significância estatística.

## Oito maiores erros individuais (módulo S-E)

| Data | Partida | Placar | E mandante | Erro S-E |
|---|---|---|---:|---:|
| 27/04 | Palmeiras x Bahia | 0–1 | 0.801 | -0.801 |
| 27/07 | Cruzeiro x Ceará | 1–2 | 0.772 | -0.772 |
| 10/08 | Cruzeiro x Santos | 1–2 | 0.765 | -0.765 |
| 08/11 | Vasco x Juventude | 1–3 | 0.738 | -0.738 |
| 25/10 | Fortaleza x Flamengo | 1–0 | 0.274 | +0.726 |
| 10/08 | Grêmio x Sport | 0–1 | 0.726 | -0.726 |
| 08/11 | São Paulo x Bragantino | 0–1 | 0.724 | -0.724 |
| 20/11 | Bahia x Fortaleza | 2–3 | 0.710 | -0.710 |

O erro é grande quando ocorre resultado oposto à vantagem esperada. **E=0.801 não significa 80,1% de chance de vitória**; a quantidade equivale a P(vitória)+0,5P(empate).

## Cautelas e próximos experimentos

1. A calibração de mando e K usa uma única temporada de validação (2024), o que limita conclusões sobre estabilidade interanual.
2. A nota de desempenho de 2025 é *prequential*, pois os resultados anteriores de 2025 atualizam o Elo; não é previsão de toda a temporada em sua abertura.
3. O próximo teste deve estimar intervalos de incerteza para a diferença emparelhada de perdas, idealmente com reamostragem por blocos temporais e dados históricos adicionais.
4. Para comparar probabilidades 1X2, implementar um modelo que estime separadamente vitória, empate e derrota e então reportar Brier multiclasses e log loss.
5. Reexecutar 2023–2025 em diferentes esquemas de inicialização e validar a estabilidade no tempo antes de aplicação prospectiva.

## Reproduzir

```bash
python -m pip install -e ".[dev]"
python -m brasileirao538.training
python -m pytest -q
```

Os números desta nota foram reproduzidos sobre os CSVs versionados em `data/processed/` usando as mesmas fórmulas e critérios de desempate de hiperparâmetros de `training.py`. A implementação das análises detalhadas em script automatizado é parte da etapa seguinte.
