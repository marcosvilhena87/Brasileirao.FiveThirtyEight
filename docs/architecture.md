# Arquitetura proposta

Fluxo: dados brutos -> validacao e limpeza -> ratings Elo -> probabilidades 1X2 -> simulacao Monte Carlo -> agregacoes e relatorios.

## Contratos a definir

- Partidas: temporada, identificador, rodada, data, mandante, visitante, gols, estado (previsto/finalizado).
- Classificacao: pontos, jogos, vitorias, empates, derrotas, gols pro/contra e desempates oficiais.
- Previsoes: id do jogo, timestamp de geracao, versao do modelo, p_mandante, p_empate, p_visitante.
- Simulacao: semente, numero de repeticoes, versao do calendario e probabilidade por posicao.

## Cuidados metodologicos

Usar somente informacoes disponiveis na data de cada previsao; nao incorporar resultados futuros ao estimar parametros. As regras de vagas, rebaixamento e desempate podem variar por edicao e devem ser parametrizadas. Calibrar o modelo com validacao cronologica.

## Proximas implementacoes

Definir interfaces de entrada/saida, invariantes de classificacao e testes de determinismo antes de incluir dados historicos reais.
