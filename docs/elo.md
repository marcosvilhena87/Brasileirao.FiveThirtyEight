# Rating Elo — v1

O motor esta em `src/brasileirao538/elo.py` e permite atualizar ratings a partir de jogos encerrados.

## Equacoes

Para ratings `R_casa` e `R_fora`, vantagem `H` (zero em campo neutro) e escala `D`:

```text
E_casa = 1 / (1 + 10 ** ((R_fora - R_casa - H) / D))
S_casa = 1 (vitoria), 0.5 (empate), 0 (derrota)
delta = K * (S_casa - E_casa)
R_casa_novo = R_casa + delta
R_fora_novo = R_fora - delta
```

`E_casa` mede **pontuacao Elo esperada**, nao probabilidade de vitoria: `E_casa = P(vitoria) + 0.5 * P(empate)`. O calculo de probabilidades separadas 1X2 depende de outro modelo, ainda a implementar.

## Parametros iniciais (hipoteses, nao calibrados)

- Rating inicial: 1500
- Fator K: 20
- Vantagem do mandante: 80 pontos Elo
- Escala: 400 pontos Elo

Esses parametros sao configuraveis com `EloConfig`. **Nao sao estimativas empiricas validadas para o Brasileirao.** Calibrar em dados historicos com separacao cronologica antes de emitir previsoes.

## Uso

```python
from brasileirao538.elo import EloConfig, EloRatings

model = EloRatings(EloConfig(initial_rating=1500, k_factor=20, home_advantage=80))
before = model.expected("Flamengo", "Palmeiras")
result = model.record("Flamengo", "Palmeiras", 2, 1)
print(before, result.home_after, result.away_after)
print(model.ratings)
```

Para jogo em campo neutro: `model.record("A", "B", 1, 1, neutral=True)`.

### Regras da interface

- `record` recebe placar final; jogos futuros nao devem ser registrados.
- A ordem de alimentacao deve ser cronologica, sem usar resultados futuros na avaliacao.
- Identificadores dos clubes devem ser estaveis: `"Flamengo"` e `"CR Flamengo"` contam como dois clubes distintos.
- Mandante e visitante devem ser diferentes; gols precisam ser inteiros nao negativos.
- Ratings de clubes novos comecam no valor padrao sem modificar o dicionario ate um jogo ser registrado.
- Atualizacoes conservam a soma dos ratings dos dois clubes em cada jogo (sem regressao a media nesta versao).
- Empate pode fazer o mandante perder pontos Elo quando tem expectativa superior a 0,5.
- O estado em memoria nao e persistido automaticamente; persistencia e tratamento de temporadas ficarao para etapa futura.

## Testes

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
ruff check .
ruff format --check .
```

## Limitacoes

Sem ajuste de margem de gols, importancia da partida, inatividade, promocao/rebaixamento, informacao de mercado, estimacao de 1X2 ou recalibracao. O modelo serve de base reprodutivel para evolucoes futuras.
