import torch
import whisper

class SpeechToText:
    def __init__(self, device=None):
        """
        Initializes the Whisper model for speech transcription.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = whisper.load_model("small").to(self.device)  # 'small' or 'base' version

    def transcribe_audio(self, audio_path):
        """
        Transcribes the given audio file to text.
        """
        result = self.model.transcribe(audio_path)
        return result["text"]