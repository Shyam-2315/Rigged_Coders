"""Dataset validation primitives for the TrustVest preprocessing pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from ml.config import FeatureEngineeringConfig


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """Result of one business-rule check."""

    name: str
    invalid_rows: int
    description: str

    @property
    def passed(self) -> bool:
        return self.invalid_rows == 0


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Serializable validation summary for a particular pipeline stage."""

    stage: str
    row_count: int
    column_count: int
    generated_at_utc: str
    checks: tuple[ValidationCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "generated_at_utc": self.generated_at_utc,
            "passed": self.passed,
            "checks": [
                {**asdict(check), "passed": check.passed} for check in self.checks
            ],
        }


class ValidationEngine:
    """Validate the source data against configured credit-data constraints."""

    def __init__(self, config: FeatureEngineeringConfig) -> None:
        self.config = config

    def validate(self, frame: pd.DataFrame, stage: str) -> ValidationReport:
        """Return all validation results without mutating ``frame``."""
        missing_columns = sorted(set(self.config.required_columns) - set(frame.columns))
        checks: list[ValidationCheck] = [
            ValidationCheck(
                name="required_columns_present",
                invalid_rows=len(missing_columns),
                description=(
                    "All configured source columns are present"
                    if not missing_columns
                    else f"Missing columns: {', '.join(missing_columns)}"
                ),
            )
        ]
        if missing_columns:
            return ValidationReport(
                stage=stage,
                row_count=len(frame),
                column_count=len(frame.columns),
                generated_at_utc=self._timestamp(),
                checks=tuple(checks),
            )

        checks.extend(
            (
                ValidationCheck(
                    name="no_missing_values",
                    invalid_rows=int(frame.isna().any(axis=1).sum()),
                    description="Every row contains values for all source columns",
                ),
                ValidationCheck(
                    name="no_duplicate_users",
                    invalid_rows=int(frame[self.config.user_id_column].duplicated().sum()),
                    description="Each user identifier is unique",
                ),
                self._numeric_check(
                    frame,
                    "income_positive",
                    "monthly_income",
                    lambda value: value <= self.config.min_income - 0.01,
                    "Monthly income is greater than zero",
                ),
                self._numeric_check(
                    frame,
                    "expenses_non_negative",
                    "monthly_expenses",
                    lambda value: value < self.config.min_expenses,
                    "Monthly expenses are non-negative",
                ),
                self._numeric_check(
                    frame,
                    "savings_non_negative",
                    "monthly_savings",
                    lambda value: value < self.config.min_savings,
                    "Monthly savings are non-negative",
                ),
                self._savings_not_above_income(frame),
                self._numeric_check(
                    frame,
                    "age_in_range",
                    "age",
                    lambda value: (value < self.config.min_age) | (value > self.config.max_age),
                    f"Age is between {self.config.min_age} and {self.config.max_age}",
                ),
                self._numeric_check(
                    frame,
                    "credit_likelihood_in_range",
                    self.config.target_column,
                    lambda value: (value < self.config.min_credit_likelihood)
                    | (value > self.config.max_credit_likelihood),
                    "Credit likelihood is between 0 and 100",
                ),
            )
        )
        checks.extend(self._ratio_checks(frame))
        return ValidationReport(
            stage=stage,
            row_count=len(frame),
            column_count=len(frame.columns),
            generated_at_utc=self._timestamp(),
            checks=tuple(checks),
        )

    def _ratio_checks(self, frame: pd.DataFrame) -> list[ValidationCheck]:
        return [
            self._numeric_check(
                frame,
                f"{column}_in_range",
                column,
                lambda value: (value < self.config.min_ratio) | (value > self.config.max_ratio),
                f"{column} is between 0 and 1",
            )
            for column in self.config.payment_ratio_columns
        ]

    def _savings_not_above_income(self, frame: pd.DataFrame) -> ValidationCheck:
        income = pd.to_numeric(frame["monthly_income"], errors="coerce")
        savings = pd.to_numeric(frame["monthly_savings"], errors="coerce")
        invalid = income.isna() | savings.isna() | (savings > income)
        return ValidationCheck(
            name="savings_not_above_income",
            invalid_rows=int(invalid.sum()),
            description="Monthly savings do not exceed monthly income",
        )

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _numeric_check(
        frame: pd.DataFrame,
        name: str,
        column: str,
        invalid_predicate: object,
        description: str,
    ) -> ValidationCheck:
        values = pd.to_numeric(frame[column], errors="coerce")
        # ``invalid_predicate`` is kept generic to make configuration-driven
        # validators concise while preserving a small, testable public surface.
        invalid = values.isna() | invalid_predicate(values)  # type: ignore[operator]
        return ValidationCheck(name, int(invalid.sum()), description)


def write_validation_report(
    path: Path,
    reports: dict[str, ValidationReport],
) -> None:
    """Persist validation evidence from each pipeline stage as JSON."""
    payload = {stage: report.to_dict() for stage, report in reports.items()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
