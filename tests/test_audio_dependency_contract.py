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
    # 2026-10-05 rebase: the companion fork pin (a266d8db) fell 120 commits
    # behind upstream v0.5.7, so the audio extra tracks the immutable upstream
    # pin until the companion is rebased onto >=94c7716. Either way the pin
    # must be a full 40-hex SHA, never a moving branch.
    data = tomllib.loads(Path("pyproject.toml").read_text())
    audio = data["project"]["optional-dependencies"]["audio"]
    pin = next(req for req in audio if req.startswith("mlx-audio["))
    assert re.fullmatch(
        r"mlx-audio\[tts,stt,sts\] @ git\+https://github\.com/(Blaizzy|contrapuntal)/mlx-audio@[0-9a-f]{40}",
        pin,
    )
