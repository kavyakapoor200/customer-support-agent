import logging
import os
from typing import Any

from pydantic import BaseModel

from src.core.config import get_settings

logger = logging.getLogger(__name__)

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


class ResponseGenerationResult(BaseModel):
    generated_text: str
    model_used: str
    tokens_used: int | None = None
    is_simulated: bool = False


class GroqSystemTwoResponder:
    """
    Layer 2 / System 2 Generative Responder powered by open-source models via Groq.
    Only triggered for Safe & Routine tickets (P1 and P2).
    Strictly grounded on Layer 5 Qdrant policy retrieval.
    """

    def __init__(self, api_key: str | None = None, model: str = "openai/gpt-oss-120b"):
        settings = get_settings()
        self.api_key = api_key or settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        self.model = model or settings.GROQ_MODEL
        self.client = None
        if HAS_GROQ and self.api_key and not self.api_key.startswith("gsk_your"):
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                logger.warning("Failed to initialize Groq client: %s", e)
                self.client = None

    def generate_response(
        self,
        ticket_text: str,
        department: str,
        urgency: int,
        language: str = "english",
        policies: list[dict[str, Any]] | None = None,
    ) -> ResponseGenerationResult:
        """Drafts a high-quality, personalized response for safe tickets grounded in company policy."""
        if self.client:
            return self._call_groq(ticket_text, department, urgency, language, policies)
        return self._simulate_response(ticket_text, department, language, policies)

    def _call_groq(
        self,
        ticket_text: str,
        department: str,
        urgency: int,
        language: str,
        policies: list[dict[str, Any]] | None = None,
    ) -> ResponseGenerationResult:
        clean_lang = (language or "english").lower().strip()
        if clean_lang == "hindi":
            lang_instruction = "fluent, natural Hindi written in Devanagari script"
        elif clean_lang == "hinglish":
            lang_instruction = "conversational Hinglish (colloquial Hindi/Urdu words written in Latin script)"
        elif clean_lang == "french":
            lang_instruction = "fluent, professional French (Français)"
        elif clean_lang == "spanish":
            lang_instruction = "fluent, professional Spanish (Español)"
        elif clean_lang == "german":
            lang_instruction = "fluent, professional German (Deutsch)"
        else:
            lang_instruction = "fluent, professional, and empathetic English"

        policy_context = ""
        if policies:
            snippets = "\n".join([f"- **{p.get('title', 'Company SLA')}:** {p.get('content', '')}" for p in policies[:2]])
            policy_context = (
                f"\nRELEVANT COMPANY POLICIES & CONSTRAINTS (MANDATORY INVARIANTS):\n"
                f"{snippets}\n"
                "You must strictly adhere to these policies. Never promise terms that contradict company policy.\n"
            )

        system_prompt = (
            f"You are a friendly, highly professional customer support specialist for the '{department}' department.\n"
            f"The customer's primary language is '{clean_lang.upper()}'. You MUST respond strictly in {lang_instruction}.\n"
            f"The urgency level is {urgency}/3.\n"
            f"{policy_context}\n"
            f"Provide a clear, helpful, empathetic, and actionable solution to their ticket.\n"
            f"Keep your response concise, polite, and directly address their specific need."
        )

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": ticket_text},
                ],
                model=self.model,
                temperature=0.3,
                max_tokens=600,
            )

            content = chat_completion.choices[0].message.content or ""
            usage = chat_completion.usage.total_tokens if chat_completion.usage else None

            return ResponseGenerationResult(
                generated_text=content.strip(),
                model_used=self.model,
                tokens_used=usage,
                is_simulated=False,
            )
        except Exception as e:
            logger.warning("Groq call failed (%s). Falling back to smart response.", e)
            fallback = self._simulate_response(ticket_text, department, language, policies)
            return fallback

    def _simulate_response(
        self,
        ticket_text: str,
        department: str,
        language: str,
        policies: list[dict[str, Any]] | None = None,
    ) -> ResponseGenerationResult:
        """Fallback response generator when GROQ_API_KEY is not available or offline."""
        clean_lang = (language or "english").lower().strip()
        if clean_lang == "hindi":
            if department == "billing":
                text = "नमस्ते, बिलिंग सपोर्ट से संपर्क करने के लिए धन्यवाद। हमने आपके खाते की समीक्षा कर ली है और आपका चालान अपडेट कर दिया गया है।"
            elif department == "technical":
                text = "नमस्ते, तकनीकी सहायता से संपर्क करने के लिए धन्यवाद। कृपया अपने क्रेडेंशियल्स की जांच करें और हमारी स्टेटस रिपोर्ट देखें।"
            else:
                text = "नमस्ते, सपोर्ट टीम से संपर्क करने के लिए धन्यवाद। आपका अनुरोध प्राप्त हो गया है और हम सहायता करने के लिए तैयार हैं।"
        elif clean_lang == "hinglish":
            if department == "billing":
                text = "Hi, billing support se contact karne ke liye shukriya. Humne aapke account details check kar liye hain aur invoice update ho gaya hai."
            elif department == "technical":
                text = "Hi, technical support se connect karne ke liye thanks. Humne aapki query inspect kar li hai aur credentials verify karne ki advice dete hain."
            else:
                text = "Hi, support team se connect karne ke liye thanks. Humne aapka message receive kar liya hai aur team jald update degi."
        elif clean_lang == "french":
            text = (
                "Bonjour,\n\nMerci d'avoir contacté notre équipe de support. Nous avons bien reçu votre demande "
                "et nous nous en occupons immédiatement selon nos politiques de service."
            )
        else:
            if department == "billing":
                text = (
                    "Hi there,\n\n"
                    "Thank you for contacting our Billing Support team. I've located your account and reviewed your latest "
                    "invoice details according to our SLA policies. Everything is now up to date, and you can access your updated receipt directly from "
                    "your billing settings dashboard. Please let us know if you need any additional adjustments!"
                )
            elif department == "technical":
                text = (
                    "Hello,\n\n"
                    "Thanks for reaching out to Technical Support. We investigated the issue described in your ticket. "
                    "Please verify your current API credentials and check our status page for any maintenance updates. "
                    "If the issue persists, replying with your request ID will help us trace the stack."
                )
            else:
                text = (
                    "Hello,\n\n"
                    "Thank you for reaching out to our support team. We have received your inquiry and are happy to help. "
                    "Please feel free to share any extra details so we can assist you right away."
                )

        return ResponseGenerationResult(
            generated_text=text,
            model_used=f"{self.model} (simulated preview)",
            tokens_used=75,
            is_simulated=True,
        )
