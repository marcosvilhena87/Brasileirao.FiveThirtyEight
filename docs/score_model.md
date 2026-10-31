# Placar Poisson condicionado ao Elo-Davidson 1X2

O modelo `conditioned_poisson` cria uma distribuição de placares `P(gols_casa, gols_fora)`. A massa inicial de cada placar é o produto de duas Poisson, truncadas em 10 gols por time. Em seguida, **cada uma das três categorias** (vitória da casa, empate, vitória de fora) é normalizada separadamente e multiplicada pela probabilidade 1X2 Davidson correspondente.

Assim, a soma de todas as células de vitória/empate/derrota **coincide com as probabilidades 1X2 de entrada** (salvo arredondamento numérico). Em particular, a frequência de 1X2 do simulador não muda ao introduzir o placar.

## Integração ao motor

O parâmetro `simulate_scores=True` (padrão) passa a gerar placares e atualizar pontos, vitórias, gols marcados e saldo em cada temporada. O modo legado `simulate_scores=False` continua disponível para comparar com a versão anterior, quando saldo e gols eram congelados.

A ordenação aplica pontos, vitórias, saldo e gols marcados. Empates restantes ainda são resolvidos pela **ordem de entrada**; confrontos diretos, cartões e sorteios regulamentares ainda não foram implementados.

### Exemplo

```python
from brasileirao538.probabilities import Probabilities
from brasileirao538.scores import conditioned_poisson

p = Probabilities(0.5, 0.25, 0.25)
distribution = conditioned_poisson(p, home_lambda=1.45, away_lambda=1.15)
assert abs(sum(distribution.outcome_probabilities().as_tuple()) - 1) < 1e-10
```

## Limitações de validade

**As intensidades 1,45 e 1,15 gols são valores iniciais ilustrativos, NÃO calibrados.** A distribuição de placares condicionada ao 1X2 ajusta perfeitamente os resultados 1X2 por construção, mas não prova a adequação das margens de vitória, saldo de gols, nem correlação de baixas contagens.

O truncamento em até 10 gols redistribui a pequena cauda Poisson dentro de cada categoria. Amostras com jogos futuros precisam de validação de gols e placares em temporadas fora da amostra; uma alternativa posterior é Dixon-Coles, especialmente para placares baixos.

```bash
python -m pytest -q
```
