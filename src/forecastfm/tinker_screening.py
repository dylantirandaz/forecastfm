"""Conservative text checks applied before a Tinker upload."""

import re

from forecastfm.models import TrainingExample

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
    texts = [
        example.case.question.question_id,
        example.case.question.text,
        example.case.question.resolution_rule,
        example.case.question.resolution_source,
        example.case.prior_source,
        example.target_method,
        *example.case.question.outcomes,
    ]
    for card in example.case.evidence:
        texts.extend((card.text, card.source))
    matches = {
        match.group(0).lower() for text in texts for match in _BLOCKED_HEALTH_PATTERN.finditer(text)
    }
    return tuple(sorted(matches))


def require_health_screen_passes(example: TrainingExample) -> None:
    """Reject flagged terms; this screen does not establish policy compliance."""
    terms = flagged_health_terms(example)
    if terms:
        joined_terms = ", ".join(terms)
        raise TinkerScreeningError(f"case contains flagged health terms: {joined_terms}")
