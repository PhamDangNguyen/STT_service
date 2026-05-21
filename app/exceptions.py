class NoteNotFoundError(Exception):
    def __init__(self, note_id: str):
        super().__init__(f"Note not found: {note_id}")
        self.note_id = note_id


class AudioFetchError(Exception):
    def __init__(self, url: str, reason: str = ""):
        super().__init__(f"Failed to fetch audio from {url}: {reason}")
        self.url = url


class UnsupportedAudioFormatError(Exception):
    SUPPORTED = frozenset({".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".opus"})

    def __init__(self, extension: str):
        super().__init__(
            f"Unsupported audio format: {extension}. Supported: {self.SUPPORTED}"
        )
        self.extension = extension


class PipelineStepError(Exception):
    def __init__(self, step_name: str, cause: Exception):
        super().__init__(f"Pipeline step '{step_name}' failed: {cause}")
        self.step_name = step_name
        self.cause = cause


class ProgressNotFoundError(Exception):
    def __init__(self, note_id: str):
        super().__init__(f"No progress record found for note: {note_id}")
        self.note_id = note_id
