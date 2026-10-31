# Motor Monte Carlo 1X2 — primeira versão

## Escopo

O módulo `brasileirao538.simulation` executa até **300.000 temporadas (padrão)** em lotes de 10.000, sem materializar a matriz completa de partidas × temporadas na memória. Usa gerador `numpy.random.default_rng(seed)` e amostra uma classe **1 / X / 2** para cada partida restante.

Entradas:
- `TeamState(name, points, wins, goal_difference, goals_for)`: estado conhecido na data do corte.
- `Fixture(home, away, probabilities)`: jogo futuro e probabilidades Davidson fixas.
- `fixtures_from_elo(..., ratings, draw_weight=0.75, config=EloConfig(...))`: conversão opcional do Elo de clubes em probabilidades 1X2. Os ratings são **congelados** durante cada simulação. Não há evolução de Elo nos cenários gerados.
- `simulate(..., simulations=300_000, seed=2026, relegation_slots=4, batch_size=10_000)`.

Saídas: probabilidades por colocação, título (1º), rebaixamento (últimas quatro posições, se configurado) e pontuação final média.

## Limite importante: gols e desempates

As probabilidades 1X2 **não informam placares ou saldo de gols**. Portanto esta versão simula **pontos e número de vitórias**, mas usa `goal_difference` e `goals_for` da classificação inicial como **proxies congeladas**. A ordenação é pontos > vitórias > saldo inicial > gols marcados iniciais > ordem de entrada para empates residuais. **Não representa integralmente os critérios oficiais do Brasileirão**, que incluem saldo/gols finais e critérios adicionais. O desempate por ordem de entrada é apenas uma convenção técnica, não um critério oficial.

A metodologia é adequada para verificar a infraestrutura de simulação, mas probabilidades de campeão/rebaixado de uma temporada real devem ser rotuladas **provisórias** até a implementação de placares, saldo e desempates oficiais. Não reivindicar resultados oficiais ou exatidão onde houver empate.

## Exemplo mínimo

```python
from brasileirao538.simulation import TeamState, fixtures_from_elo, simulate

teams = [TeamState("Flamengo", points=70, wins=21), TeamState("Palmeiras", points=68, wins=20)]
ratings = {"Flamengo": 1650, "Palmeiras": 1620}
remaining = [("Palmeiras", "Flamengo")]
fixtures = fixtures_from_elo(remaining, ratings)
results = simulate(teams, fixtures, simulations=300_000, seed=2026, relegation_slots=0)
print(results.title)
print(results.expected_points)
```

Estes números são **fictícios**, não a tabela ou os ratings oficiais de qualquer edição.

## Reprodutibilidade e hipóteses

- Fixar semente, ordem da lista de partidas, número de simulações e tamanho do lote para resultados idênticos.
- A distribuição 1X2 é obtida do Davidson (`nu=0.75`, `K=30` e vantagem de mando de 80 pontos por padrão), mas as probabilidades individuais são tomadas como **independentes e fixas**; o modelo não incorpora escalações, suspensões ou calendário de outras competições.
- Simulador não baixa calendário ou classificação do Brasileirão 2026: o chamador precisa fornecer dados atuais e verificados.
- Não há risco de uso de placares futuros na inicialização, desde que o chamador respeite a data de corte e ratings pré-jogo.

## Próximas etapas

Adicionar distribuição de gols condicionada a 1X2, critérios de desempate por temporada, importação de calendário, validação retrospectiva do simulador e análise de sensibilidade às probabilidades 1X2.
