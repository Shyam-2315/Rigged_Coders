"""Human-readable behavioral personas returned with risk predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class BehavioralPersona:
    """A concise, product-ready description of an investor behavioral segment."""

    name: str
    description: str
    investment_philosophy: str
    strengths: tuple[str, ...]
    potential_risks: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


PERSONAS: dict[str, tuple[BehavioralPersona, BehavioralPersona]] = {
    "Low": (
        BehavioralPersona(
            name="Secure Saver",
            description="Prioritises capital stability, emergency access, and gradual wealth accumulation.",
            investment_philosophy="Protect the downside first; invest only when the financial cushion is secure.",
            strengths=("Disciplined cash reserves", "Strong focus on financial security", "Low exposure to volatility"),
            potential_risks=("Inflation may erode purchasing power", "May delay long-term wealth creation", "Can hold excess cash"),
        ),
        BehavioralPersona(
            name="Conservative Planner",
            description="Plans ahead and values predictability while taking measured, well-understood steps.",
            investment_philosophy="Use diversified, understandable investments to pursue goals without large drawdowns.",
            strengths=("Goal-oriented planning", "Patient decision-making", "Preference for predictable outcomes"),
            potential_risks=("May under-allocate to growth assets", "Can react cautiously after market volatility", "Needs return expectations aligned to goals"),
        ),
    ),
    "Medium": (
        BehavioralPersona(
            name="Balanced Builder",
            description="Comfortable combining stability and growth through steady, diversified investing.",
            investment_philosophy="Balance near-term resilience with long-term compounding and periodic rebalancing.",
            strengths=("Moderate loss tolerance", "Practical saving-and-investing habit", "Balanced long-term outlook"),
            potential_risks=("May drift from allocation during market swings", "Needs adequate emergency reserves", "Can overestimate diversification"),
        ),
        BehavioralPersona(
            name="Strategic Investor",
            description="Uses investment knowledge and planning to pursue growth with deliberate risk controls.",
            investment_philosophy="Take selective risk when it is supported by research, a time horizon, and financial readiness.",
            strengths=("Informed decision-making", "Structured investment process", "Comfort with calibrated risk"),
            potential_risks=("May become overconfident", "Requires periodic portfolio review", "Complex products can obscure risk"),
        ),
    ),
    "High": (
        BehavioralPersona(
            name="Growth Explorer",
            description="Seeks long-term growth and remains comparatively composed through market volatility.",
            investment_philosophy="Accept short-term fluctuations in pursuit of long-duration compounding opportunities.",
            strengths=("Long investment horizon", "High tolerance for temporary losses", "Openness to growth assets"),
            potential_risks=("Drawdowns can be substantial", "Needs diversification discipline", "Liquidity needs must remain funded"),
        ),
        BehavioralPersona(
            name="Aggressive Wealth Seeker",
            description="Actively pursues high-growth opportunities and is willing to accept meaningful volatility.",
            investment_philosophy="Seek upside while setting clear loss limits, liquidity reserves, and concentration controls.",
            strengths=("High return orientation", "Strong market engagement", "Willingness to stay invested"),
            potential_risks=("May chase returns", "Concentration and leverage can magnify losses", "Confidence can exceed experience"),
        ),
    ),
}


def select_persona(risk_profile: str, features: Mapping[str, float]) -> BehavioralPersona:
    """Choose one of the profile's two personas from observable behavior."""
    if risk_profile not in PERSONAS:
        raise ValueError(f"Unsupported risk profile: {risk_profile}")
    primary, alternate = PERSONAS[risk_profile]
    if risk_profile == "Low":
        return primary if float(features["liquidity_preference_index"]) < 48 else alternate
    if risk_profile == "Medium":
        return alternate if float(features["investment_experience_index"]) >= 62 else primary
    return alternate if float(features["risk_tolerance_score"]) >= 82 else primary


def recommendation_summary(risk_profile: str, persona: BehavioralPersona) -> str:
    """Return an intentionally non-prescriptive integration summary."""
    summaries = {
        "Low": "You favour financial stability and lower volatility; recommendations should emphasise liquidity, diversification, and goal-aligned pacing.",
        "Medium": "You appear comfortable taking measured investment risk while maintaining financial stability and diversification.",
        "High": "You show a higher tolerance for volatility and long-term growth, provided emergency liquidity and diversification remain protected.",
    }
    if risk_profile not in summaries:
        raise ValueError(f"Unsupported risk profile: {risk_profile}")
    return f"{summaries[risk_profile]} Persona: {persona.name}."
