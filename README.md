# Brasileirão FiveThirtyEight

Projeto independente de previsão probabilística do Campeonato Brasileiro, inspirado em conceitos de ratings Elo e simulação Monte Carlo. **Não possui afiliação com o FiveThirtyEight.**

## Objetivos

- Estimar a força dos clubes com ratings Elo.
- Calcular probabilidades de vitória, empate e derrota.
- Simular a classificação final e estimar chances de título, vagas continentais e rebaixamento.
- Avaliar a calibração das previsões por backtesting histórico.
- Produzir resultados reproduzíveis, incluindo semente aleatória e versão dos dados.

> Estado atual: motor Elo v1 implementado (na branch de desenvolvimento), com testes. Probabilidades 1X2 e simulações ainda não foram implementadas; os parâmetros Elo não foram calibrados com dados reais.

## Requisitos

- Python 3.11 ou superior
- Git

## Instalação

```bash
git clone https://github.com/marcosvilhena87/Brasileirao.FiveThirtyEight.git
cd Brasileirao.FiveThirtyEight
python -m venv .venv
```

Ative o ambiente virtual:

- Windows (PowerShell): `.venv\Scripts\Activate.ps1`
- Linux/macOS: `source .venv/bin/activate`

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

## Estrutura

```text
.
├── .github/workflows/ci.yml    # Integração contínua
├── data/
│   ├── raw/                    # Dados originais (não versionados)
│   └── processed/              # Dados tratados (não versionados)
├── docs/                       # Documentação técnica
├── notebooks/                  # Exploração e experimentos
├── src/brasileirao538/
│   ├── __init__.py
│   ├── elo.py                  # Rating Elo com mando de campo
│   ├── probabilities.py        # Futuro modelo de partidas
│   └── simulation.py           # Futuro simulador Monte Carlo
├── tests/
├── .gitignore
└── pyproject.toml
```

## Roadmap

1. Definir o contrato de dados de partidas e classificação, com identificadores consistentes de clubes e temporada.
2. Implementar rating Elo com mando de campo e atualização por partida. **Concluído (v1).**
3. Definir probabilidades 1X2 por um modelo calibrável.
4. Implementar simulação Monte Carlo com desempates oficiais por temporada.
5. Adicionar validação temporal (sem vazamento de dados) e métricas como log loss e Brier score.
6. Criar CLI para previsão e exportação CSV/Parquet.

## Qualidade e reprodução

Execute `python -m pytest` para os testes, `ruff check .` para lint e `ruff format --check .` para formatação. Fixe as versões em um arquivo de lock antes de publicar resultados. Preserve a fonte, a data de coleta, a temporada, os jogos adiados e o estado do calendário em cada execução.

## Licença

Ainda não definida. Antes de distribuir ou reutilizar código externamente, escolha uma licença adequada.

## Exemplo Elo v1

```python
from brasileirao538.elo import EloRatings

model = EloRatings()
print(model.expected("Flamengo", "Palmeiras"))
match = model.record("Flamengo", "Palmeiras", 2, 1)
print(match.home_after, match.away_after)
```

O valor retornado por `expected` é a **pontuação Elo esperada**, não a probabilidade isolada de vitória. Consulte [a documentação do Elo](docs/elo.md) para fórmulas, parâmetros, campo neutro, limitações e testes.
