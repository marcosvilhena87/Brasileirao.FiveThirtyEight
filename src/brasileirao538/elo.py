"""Motor Elo deterministico para resultados de futebol.

O valor esperado representa pontuacao Elo (vitoria=1, empate=0.5,
derrota=0), e NAO a probabilidade isolada de vitoria.
"""

from dataclasses import dataclass, field
from math import isfinite


@dataclass(frozen=True)
class EloConfig:
    """Parametros de rating, expressos em pontos Elo."""

    initial_rating: float = 1500.0
    k_factor: float = 20.0
    home_advantage: float = 80.0
    scale: float = 400.0

    def __post_init__(self) -> None:
        if not all(isfinite(x) for x in (
            self.initial_rating, self.k_factor, self.home_advantage, self.scale
        )):
            raise ValueError("Parametros Elo devem ser finitos")
        if self.k_factor <= 0 or self.scale <= 0:
            raise ValueError("k_factor e scale devem ser positivos")
        if self.home_advantage < 0:
            raise ValueError("home_advantage nao pode ser negativo")


@dataclass(frozen=True)
class EloResult:
    """Resultado completo de uma atualizacao de ratings."""

    expected_home: float
    score_home: float
    home_before: float
    away_before: float
    home_after: float
    away_after: float


def expected_home_score(
    home_rating: float,
    away_rating: float,
    config: EloConfig | None = None,
    *,
    neutral: bool = False,
) -> float:
    """Retorna o escore Elo esperado do mandante (entre 0 e 1)."""
    cfg = config or EloConfig()
    if not isfinite(home_rating) or not isfinite(away_rating):
        raise ValueError("Ratings devem ser finitos")
    advantage = 0.0 if neutral else cfg.home_advantage
    exponent = (away_rating - home_rating - advantage) / cfg.scale
    # Forma estavel numericamente, mesmo para diferencas extremas.
    if exponent >= 0:
        factor = 10.0 ** (-exponent) if exponent < 300 else 0.0
        return factor / (1.0 + factor)
    factor = 10.0 ** exponent if exponent > -300 else 0.0
    return 1.0 / (1.0 + factor)


def update_ratings(
    home_rating: float,
    away_rating: float,
    home_goals: int,
    away_goals: int,
    config: EloConfig | None = None,
    *,
    neutral: bool = False,
) -> EloResult:
    """Atualiza simultaneamente dois ratings apos partida finalizada."""
    if (
        isinstance(home_goals, bool) or isinstance(away_goals, bool)
        or not isinstance(home_goals, int) or not isinstance(away_goals, int)
        or home_goals < 0 or away_goals < 0
    ):
        raise ValueError("Gols devem ser inteiros nao negativos")
    cfg = config or EloConfig()
    expected = expected_home_score(home_rating, away_rating, cfg, neutral=neutral)
    score = 1.0 if home_goals > away_goals else 0.0 if home_goals < away_goals else 0.5
    delta = cfg.k_factor * (score - expected)
    return EloResult(
        expected_home=expected,
        score_home=score,
        home_before=float(home_rating),
        away_before=float(away_rating),
        home_after=home_rating + delta,
        away_after=away_rating - delta,
    )


@dataclass
class EloRatings:
    """Estado por clube. Partidas devem ser fornecidas em ordem cronologica."""

    config: EloConfig = field(default_factory=EloConfig)
    ratings: dict[str, float] = field(default_factory=dict)

    def rating(self, team: str) -> float:
        """Retorna rating existente ou inicial sem alterar o estado."""
        if not isinstance(team, str) or not team.strip():
            raise ValueError("Identificador de clube deve ser texto nao vazio")
        return self.ratings.get(team, self.config.initial_rating)

    def expected(self, home: str, away: str, *, neutral: bool = False) -> float:
        """Calcula expectativa sem modificar os ratings."""
        self._validate_teams(home, away)
        return expected_home_score(
            self.rating(home), self.rating(away), self.config, neutral=neutral
        )

    def record(
        self,
        home: str,
        away: str,
        home_goals: int,
        away_goals: int,
        *,
        neutral: bool = False,
    ) -> EloResult:
        """Registra uma partida e atualiza os dois clubes atomicamente."""
        self._validate_teams(home, away)
        result = update_ratings(
            self.rating(home),
            self.rating(away),
            home_goals,
            away_goals,
            self.config,
            neutral=neutral,
        )
        self.ratings[home] = result.home_after
        self.ratings[away] = result.away_after
        return result

    @staticmethod
    def _validate_teams(home: str, away: str) -> None:
        if not isinstance(home, str) or not home.strip():
            raise ValueError("Mandante invalido")
        if not isinstance(away, str) or not away.strip():
            raise ValueError("Visitante invalido")
        if home == away:
            raise ValueError("Mandante e visitante devem ser diferentes")
