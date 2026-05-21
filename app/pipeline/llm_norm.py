import json

from .base import PipelineStep
from .models import LLMNormInput, LLMNormOutput, TranscribedSegment

_SYSTEM_PROMPT = """Bạn là chuyên gia chuẩn hoá văn bản tiếng Việt. Nhiệm vụ của bạn là:
1. Sửa lỗi chính tả và dấu câu tiếng Việt
2. Chuẩn hoá định dạng (viết hoa đầu câu, dấu chấm cuối câu khi cần)
3. Không thay đổi ý nghĩa gốc
4. Giữ nguyên tên riêng, thuật ngữ chuyên môn
5. Trả về JSON với cùng cấu trúc đầu vào

Quy tắc đầu ra:
- Chỉ trả về JSON thuần tuý, không có markdown code block
- Cấu trúc: {"segments": [{"start": ..., "duration": ..., "text": "...", "speaker": "..."}]}"""


class LLMNormStep(PipelineStep[LLMNormInput, LLMNormOutput]):
    def __init__(self, provider: str, model: str, api_key: str):
        self._provider = provider
        self._model = model
        self._api_key = api_key

    async def run(self, input: LLMNormInput) -> LLMNormOutput:
        if not input.segments:
            return LLMNormOutput(segments=[])

        payload = {
            "segments": [s.model_dump() for s in input.segments]
        }
        user_text = json.dumps(payload, ensure_ascii=False)

        if self._provider == "anthropic":
            return await self._run_anthropic(user_text, input.segments)
        return await self._run_openai(user_text, input.segments)

    async def _run_anthropic(
        self, user_text: str, fallback: list[TranscribedSegment]
    ) -> LLMNormOutput:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        response = await client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_text}],
        )
        raw = response.content[0].text
        return self._parse_response(raw, fallback)

    async def _run_openai(
        self, user_text: str, fallback: list[TranscribedSegment]
    ) -> LLMNormOutput:
        import openai
        client = openai.AsyncOpenAI(api_key=self._api_key)
        response = await client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        return self._parse_response(raw, fallback)

    @staticmethod
    def _parse_response(
        raw: str, fallback: list[TranscribedSegment]
    ) -> LLMNormOutput:
        try:
            data = json.loads(raw)
            segments = [TranscribedSegment(**s) for s in data["segments"]]
            return LLMNormOutput(segments=segments)
        except Exception:
            return LLMNormOutput(segments=fallback)
