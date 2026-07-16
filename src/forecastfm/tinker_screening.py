"""Conservative text checks applied before a Tinker upload."""

import re
from collections.abc import Iterable

from forecastfm.models import ForecastCase, TrainingExample

_BLOCKED_HEALTH_PATTERN = re.compile(
    r"\b("
    r"concussion|diagnosis|disease|fracture|health|illness|injured|injury|"
    r"medical|questionable|rehab|recovery|soreness|sprain|surgery"
    r")\b",
    flags=re.IGNORECASE,
)


class TinkerScreeningError(ValueError):
    """Raised when a basic health-term screen flags a training example."""


def flagged_health_terms(example: TrainingExample) -> tuple[str, ...]:
    """Return known health terms found anywhere in a training example."""
    return flagged_health_terms_in_texts((*_case_texts(example.case), example.target_method))


def flagged_case_health_terms(case: ForecastCase) -> tuple[str, ...]:
    """Return known health terms found in a model input case."""
    return flagged_health_terms_in_texts(_case_texts(case))


def flagged_health_terms_in_texts(texts: Iterable[str]) -> tuple[str, ...]:
    """Return known health terms found in arbitrary model-facing text."""
    matches = {
        match.group(0).lower() for text in texts for match in _BLOCKED_HEALTH_PATTERN.finditer(text)
    }
    return tuple(sorted(matches))


def require_health_screen_passes(example: TrainingExample) -> None:
    """Reject flagged terms; this screen does not establish policy compliance."""
    _require_no_terms(flagged_health_terms(example))


def require_case_health_screen_passes(case: ForecastCase) -> None:
    """Reject flagged terms before sending an inference case to Tinker."""
    _require_no_terms(flagged_case_health_terms(case))


def require_text_health_screen_passes(texts: Iterable[str]) -> None:
    """Reject flagged terms in model-facing text before a Tinker call."""
    _require_no_terms(flagged_health_terms_in_texts(texts))


def _case_texts(case: ForecastCase) -> tuple[str, ...]:
    question = case.question
    texts = [
        question.question_id,
        question.text,
        question.resolution_rule,
        question.resolution_source,
        case.prior_source,
        *question.outcomes,
    ]
    for card in case.evidence:
        texts.extend((card.text, card.source))
    return tuple(texts)


def _require_no_terms(terms: tuple[str, ...]) -> None:
    if terms:
        joined_terms = ", ".join(terms)
        raise TinkerScreeningError(f"case contains flagged health terms: {joined_terms}")
