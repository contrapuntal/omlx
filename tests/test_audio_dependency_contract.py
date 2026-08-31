import re
import tomllib
from pathlib import Path


def _name(requirement: str) -> str:
    return re.split(r"[\s\[<>=@;]", requirement, maxsplit=1)[0].lower()


def test_audio_extra_owns_kokoro_and_runtime_dependencies():
    data = tomllib.loads(Path("pyproject.toml").read_text())
    audio = data["project"]["optional-dependencies"]["audio"]
    names = {_name(requirement) for requirement in audio}
    assert {
        "misaki",
        "num2words",
        "spacy",
        "phonemizer-fork",
        "espeakng-loader",
        "python-multipart",
        "mistral-common",
        "torch",
        "librosa",
    } <= names


def test_audio_extra_uses_immutable_companion_fork_pin():
    data = tomllib.loads(Path("pyproject.toml").read_text())
    audio = data["project"]["optional-dependencies"]["audio"]
    pin = next(req for req in audio if req.startswith("mlx-audio["))
    assert re.fullmatch(
        r"mlx-audio\[tts,stt,sts\] @ git\+https://github\.com/contrapuntal/mlx-audio@[0-9a-f]{40}",
        pin,
    )
