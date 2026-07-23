"""Goal attainment probability calculations from simulated terminal values."""

from __future__ import annotations

import numpy as np

from .schemas import GoalProbabilityResult


def calculate_goal_probability(final_values: np.ndarray, goal_amount: float) -> GoalProbabilityResult:
    """Return educational probabilities of reaching, missing, or exceeding a goal."""
    if goal_amount <= 0:
        raise ValueError("goal_amount must be greater than zero")
    if final_values.size == 0:
        raise ValueError("final_values cannot be empty")

    success = float(np.mean(final_values >= goal_amount) * 100.0)
    shortfall = float(np.mean(final_values < goal_amount) * 100.0)
    exceeding = float(np.mean(final_values > goal_amount) * 100.0)
    return GoalProbabilityResult(
        goal_amount=float(goal_amount),
        probability_of_success=round(success, 2),
        probability_of_shortfall=round(shortfall, 2),
        probability_of_exceeding=round(exceeding, 2),
    )
