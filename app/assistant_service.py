import os
from typing import Any

from dotenv import load_dotenv

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - handled at runtime by health route
    genai = None


DERMALENS_SAFETY_WARNING = (
    "DermaLens provides AI-assisted visual skin insights for educational support. "
    "It is not a medical diagnosis. Please consult a qualified healthcare professional "
    "for medical concerns."
)


class GeminiSkinAssistantService:
    def __init__(self) -> None:
        load_dotenv(override=True)
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        self._model = None

        if self.api_key and genai is not None:
            genai.configure(
                api_key=self.api_key,
                transport="rest",
            )
            self._model = genai.GenerativeModel(self.model_name)

    def is_available(self) -> bool:
        return bool(self.api_key and self._model is not None)

    def get_health_status(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "assistant_available": self.is_available(),
            "provider": "gemini",
            "model": self.model_name,
            "api_key_configured": bool(self.api_key),
        }

        if not self.api_key:
            status["reason"] = "GEMINI_API_KEY missing"

        return status

    def ask_assistant(
        self,
        predicted_class: str,
        confidence: float,
        top_k: list[Any],
        user_question: str,
    ) -> dict[str, str]:
        if not self.is_available():
            raise RuntimeError("Gemini API key is not configured on the server.")

        safe_question = user_question.strip()[:500]
        top_k_text = self._format_top_k(top_k)
        prompt = f"""
You are DermaLens Skin Assistant.
You provide general educational skincare information only.
You do not diagnose.
You do not claim the user has a condition.
You do not prescribe medication.
You do not recommend dangerous treatments.
You do not replace a dermatologist or healthcare professional.
You encourage professional consultation for symptoms that are concerning, painful, bleeding, rapidly changing, worsening, spreading, infected-looking, or persistent.
You use calm, simple wording.
You avoid scary language.
You explain that the visual result is a possible match, not a confirmed condition.
You do not over-trust the confidence score.
You avoid technical terms like Grad-CAM, EfficientNet, logits, classifier, neural network, PyTorch, model internals.
If the user asks for diagnosis, medication, treatment certainty, or urgent medical judgment, refuse gently and redirect to safe educational guidance and professional care.

Answer style:
- Short direct answer first.
- Then 2 to 4 concise bullet points maximum.
- End with a safety reminder.
- Avoid long medical explanations.
- Keep language suitable for general users.

Skin insight context:
Possible match: {predicted_class}
Confidence score: {confidence:.4f}
Top possible matches:
{top_k_text}

User question:
{safe_question}
""".strip()

        try:
            response = self._model.generate_content(prompt)
            answer = getattr(response, "text", "").strip()
        except Exception as error:
            raise RuntimeError("Gemini assistant request failed.") from error

        if not answer:
            raise RuntimeError("Gemini assistant returned an empty response.")

        return {
            "answer": answer,
            "warning": DERMALENS_SAFETY_WARNING,
        }

    @staticmethod
    def _format_top_k(top_k: list[Any]) -> str:
        rows: list[str] = []

        for item in top_k[:5]:
            if hasattr(item, "class_name"):
                class_name = item.class_name
                confidence = item.confidence
            else:
                class_name = item.get("class_name", "Unknown")
                confidence = item.get("confidence", 0)

            rows.append(f"- {class_name}: {float(confidence) * 100:.2f}%")

        return "\n".join(rows) if rows else "- No additional matches provided."


assistant_service = GeminiSkinAssistantService()
