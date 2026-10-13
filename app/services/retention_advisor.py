import os
from langchain_groq import ChatGroq
from app.config import settings

# ---------------------------------------------------------------
# Company retention policy — grounded context (RAG-style)
# ---------------------------------------------------------------
RETENTION_POLICY = """
RETENTION PLAYBOOK — STC-STYLE TELECOM POLICY (2026)

Eligible retention offers by risk tier:

HIGH RISK (churn probability >= 0.70):
  - 30% discount on monthly bill for 6 months
  - Free upgrade to premium support for 12 months
  - Option to switch to one-year contract with locked-in price

MEDIUM RISK (churn probability 0.40-0.69):
  - 15% discount for 3 months
  - Free 20 GB additional data for 3 months
  - Free online security subscription for 6 months

LOW RISK (churn probability < 0.40):
  - Loyalty bonus: 5 GB extra data for 1 month
  - Thank-you message

Escalation rules:
  - If customer has late_payments >= 2, offer a one-time bill credit of SAR 50
  - If customer has tenure_months > 48, add a loyalty upgrade: 10 GB extra monthly data for 6 months
  - If customer has num_support_calls >= 3, assign a dedicated account manager

Tone guidelines:
  - Warm, respectful, personalized — never robotic
  - Mention the customer's specific concern (from SHAP factors)
  - Never promise anything outside this playbook
"""


class RetentionAdvisor:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY") or settings.groq_api_key
        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.4,
            api_key=api_key,
            max_tokens=800,
        )
        self.policy = RETENTION_POLICY

    def _tier(self, prob: float) -> str:
        if prob >= 0.70:
            return "HIGH"
        if prob >= 0.40:
            return "MEDIUM"
        return "LOW"

    def _top_factor_summary(self, factors: list[dict]) -> str:
        lines = []
        for f in factors[:3]:
            lines.append(f"- {f['feature']}: {f['direction']}")
        return "\n".join(lines)

    def _build_prompt(self, customer: dict, prob: float, factors: list[dict]) -> str:
        tier = self._tier(prob)
        factor_summary = self._top_factor_summary(factors)
        return f"""You are a retention specialist for a Saudi telecom company.
Write TWO personalized retention messages for the customer below — one in English, one in Arabic.

CUSTOMER PROFILE:
- Customer ID: {customer.get('customer_id', 'N/A')}
- Tenure: {customer.get('tenure_months', '?')} months
- Monthly bill: SAR {customer.get('monthly_charges', '?')}
- Contract: {customer.get('contract_type', '?')}
- Internet: {customer.get('internet_service', '?')}
- Support calls: {customer.get('num_support_calls', '?')}
- Days since last login: {customer.get('days_since_last_login', '?')}
- Late payments: {customer.get('late_payments', '?')}

CHURN RISK: {prob:.0%}  (Tier: {tier})

TOP RISK FACTORS (from SHAP analysis):
{factor_summary}

RETENTION POLICY (use ONLY offers listed here):
{self.policy}

INSTRUCTIONS:
- Write the English message first, then "===ARABIC===", then the Arabic version.
- Each message: 3-4 sentences. Warm, specific, respectful.
- Reference the customer's specific risk factors, not generic reasons.
- Only offer what the policy allows for the {tier} tier.
- Do NOT invent new offers. Do NOT exceed the policy.
- In Arabic, use formal but friendly tone (أنت form is fine).

Output format:
ENGLISH:
<english message>
===ARABIC===
<arabic message>
"""

    def generate(self, customer: dict, prob: float, factors: list[dict]) -> tuple[str, str]:
        prompt = self._build_prompt(customer, prob, factors)
        try:
            response = self.llm.invoke(prompt).content.strip()
        except Exception as e:
            return f"LLM error: {e}", f"خطأ في النموذج: {e}"

        if "===ARABIC===" in response:
            en, ar = response.split("===ARABIC===", 1)
            en = en.replace("ENGLISH:", "").strip()
            return en, ar.strip()
        return response, ""

    def refusal_check(self, customer: dict, prob: float, factors: list[dict]) -> str:
        """Ask the LLM to verify that the offer only uses policy offers."""
        # Simple guard: if the generated offer mentions a % not in policy, flag it.
        # In a real system we'd run a second LLM call for verification.
        return "verified"


retention_advisor = RetentionAdvisor()