from ..logging_setup import get_logger
from .base import PipelineStep
from .models import LLMNormInput, LLMNormOutput, TranscribedSegment

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "Bạn là chuyên gia chính tả tiếng Việt. "
    "Sửa lỗi chính tả, dấu câu và viết hoa của câu được cung cấp. "
    "Giữ nguyên nghĩa gốc, tên riêng và thuật ngữ chuyên môn. "
    "Chỉ trả về câu đã sửa, không giải thích thêm."
)


class LLMNormStep(PipelineStep[LLMNormInput, LLMNormOutput]):
    """Corrects Vietnamese spelling and punctuation for each ASR segment."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: str = "",
    ):
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._base_url = base_url

    async def run(self, input: LLMNormInput) -> LLMNormOutput:
        if not input.segments:
            return LLMNormOutput(segments=[])

        corrected: list[TranscribedSegment] = []
        for seg in input.segments:
            corrected_text = await self._correct_segment(seg.text)
            corrected.append(seg.model_copy(update={"text": corrected_text}))

        return LLMNormOutput(segments=corrected)

    async def _correct_segment(self, text: str) -> str:
        """Correct a single segment. Returns the original text on any error."""
        try:
            if self._provider == "anthropic":
                return await self._call_anthropic(text)
            return await self._call_openai_compatible(text)
        except Exception as exc:
            logger.warning("LLMNormStep correction failed, keeping original | error=%s", exc)
            return text

    async def _call_anthropic(self, text: str) -> str:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        response = await client.messages.create(
            model=self._model,
            max_tokens=512,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": text}],
        )
        return response.content[0].text.strip()

    async def _call_openai_compatible(self, text: str) -> str:
        """Supports both OpenAI cloud and local servers (OpenAI-compatible API)."""
        import openai

        client_kwargs: dict = {"api_key": self._api_key or "local"}
        if self._base_url:
            client_kwargs["base_url"] = self._base_url

        client = openai.AsyncOpenAI(**client_kwargs)
        response = await client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        return (response.choices[0].message.content or text).strip()
