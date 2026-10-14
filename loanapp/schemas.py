"""Request validation.

Pydantic v2 does the parsing and range checking so that everything downstream
of this module can assume well-formed input. Validation errors are converted to
a flat ``{field: message}`` shape that both the JSON API and the HTML forms
render directly.

Validation here is about *shape and plausible range*. Business judgement
(is this instalment affordable?) lives in :mod:`loanapp.rules`, where it can be
explained to the applicant rather than rejected as a 400.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .features import LoanApplication

USER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{3,50}$")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$")


class SchemaError(ValueError):
    """Validation failed. ``errors`` maps field name to a readable message."""

    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))


def _flatten(exc: ValidationError) -> dict[str, str]:
    errors: dict[str, str] = {}
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "_"
        message = error["msg"]
        # Pydantic prefixes custom ValueError messages; strip the noise.
        errors.setdefault(location, message.removeprefix("Value error, "))
    return errors


def parse(model: type[BaseModel], payload: Any) -> BaseModel:
    """Validate ``payload`` against ``model``, raising :class:`SchemaError`."""
    try:
        return model.model_validate(dict(payload or {}))
    except ValidationError as exc:
        raise SchemaError(_flatten(exc)) from exc


class RegistrationRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    user_id: str
    email: str
    full_name: str = Field(min_length=2, max_length=100)
    password: str = Field(min_length=12, max_length=200)
    confirm_password: str

    @field_validator("user_id")
    @classmethod
    def check_user_id(cls, value: str) -> str:
        if not USER_ID_PATTERN.match(value):
            raise ValueError(
                "Username must be 3-50 characters, using letters, numbers, "
                "dot, dash or underscore only"
            )
        return value.lower()

    @field_validator("email")
    @classmethod
    def check_email(cls, value: str) -> str:
        if not EMAIL_PATTERN.match(value):
            raise ValueError("Enter a valid email address")
        return value.lower()

    @field_validator("password")
    @classmethod
    def check_password(cls, value: str) -> str:
        # Length beats composition rules for real-world strength, so we ask for
        # 12 characters and only reject the genuinely obvious.
        if value.lower() in {
            "password1234",
            "123456789012",
            "qwertyuiop12",
            "letmein12345",
        }:
            raise ValueError("Choose a less predictable password")
        if len(set(value)) < 5:
            raise ValueError("Password must use at least 5 distinct characters")
        return value

    @model_validator(mode="after")
    def check_match(self) -> RegistrationRequest:
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


class LoginRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    user_id: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=1, max_length=200)

    @field_validator("user_id")
    @classmethod
    def normalise(cls, value: str) -> str:
        return value.lower()


class ApplicationRequest(BaseModel):
    """A loan application submitted for assessment.

    Ranges are wide on purpose: the job of this layer is to reject impossible
    input, not to pre-judge the applicant. An unaffordable request is valid
    input that the rules layer will explain.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    label: str = Field(default="", max_length=80)
    applicant_income: float = Field(ge=0, le=100_000_000, description="Monthly")
    coapplicant_income: float = Field(default=0.0, ge=0, le=100_000_000)
    loan_amount: float = Field(gt=0, le=1_000_000_000, description="Total principal")
    loan_term_months: int = Field(ge=6, le=480)
    credit_history: int = Field(ge=0, le=1)
    dependents: int = Field(default=0, ge=0, le=10)
    education: Literal["graduate", "not_graduate"] = "graduate"
    self_employed: bool = False
    property_area: Literal["rural", "semiurban", "urban"] = "urban"
    cibil_score: int | None = Field(default=None, ge=300, le=900)

    @field_validator("self_employed", mode="before")
    @classmethod
    def coerce_checkbox(cls, value: Any) -> Any:
        # HTML forms send "on"/"yes"/"1"/absent rather than JSON booleans.
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return value

    @field_validator("cibil_score", mode="before")
    @classmethod
    def blank_is_none(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    def to_application(self) -> LoanApplication:
        return LoanApplication(
            applicant_income=self.applicant_income,
            coapplicant_income=self.coapplicant_income,
            loan_amount=self.loan_amount,
            loan_term_months=self.loan_term_months,
            credit_history=self.credit_history,
            dependents=self.dependents,
            education=self.education,
            self_employed=self.self_employed,
            property_area=self.property_area,
            cibil_score=self.cibil_score,
        )


class CompareRequest(BaseModel):
    """Two to five scenarios scored side by side."""

    scenarios: list[ApplicationRequest] = Field(min_length=2, max_length=5)
