import os
import sys
import pytest

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from camera_framing import CameraZone, CameraFramingConfig
from director import TVDirector

@pytest.fixture
def sample_config():
    z_host = CameraZone("zone_host", "Host", 0, 0, 960, 1080, speaker_label="SPEAKER_00")
    z_guest = CameraZone("zone_guest", "Guest", 960, 0, 960, 1080, speaker_label="SPEAKER_01")
    z_wide = CameraZone("zone_wide", "Wide", 0, 0, 1920, 1080, is_wide=True)
    return CameraFramingConfig(
        file_name="podcast.mp4",
        source_width=1920,
        source_height=1080,
        zones=[z_host, z_guest, z_wide]
    )

def test_filler_word_suppression(sample_config):
    director = TVDirector(sample_config, min_shot_duration=2.0)
    # Dialogue with a quick 0.5s filler "yeah" from Guest in between Host's speech
    words = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "word": "Why did you build"},
        {"start": 2.0, "end": 4.5, "speaker": "SPEAKER_00", "word": "this distributed database architecture?"},
        {"start": 4.6, "end": 5.0, "speaker": "SPEAKER_01", "word": "yeah"}, # Filler word < 1.8s
        {"start": 5.1, "end": 8.0, "speaker": "SPEAKER_00", "word": "Because latency was the bottleneck."},
    ]

    shots = director.direct_clip(words, clip_start=0.0, clip_end=8.0, use_ollama=False)
    assert len(shots) > 0
    # Because "yeah" is a filler word under 1.8s, the camera should NOT jitter cut to Guest for 0.4s
    assert len(shots) == 1
    assert shots[0]["camera"] == "zone_host"
    assert shots[0]["start"] == 0.0
    assert shots[0]["end"] == 8.0

def test_minimum_shot_duration_hysteresis(sample_config):
    director = TVDirector(sample_config, min_shot_duration=2.0)
    # Turn of 1.0 second should not produce a 1.0s cut
    words = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "word": "Hello"},
        {"start": 1.0, "end": 10.0, "speaker": "SPEAKER_01", "word": "This is a detailed explanation of our algorithms."}
    ]

    shots = director.direct_clip(words, clip_start=0.0, clip_end=10.0, use_ollama=False)
    for shot in shots:
        dur = shot["end"] - shot["start"]
        assert dur >= 1.95, f"Shot duration {dur}s is less than minimum 2.0s"

def test_split_stack_for_rapid_banter(sample_config):
    director = TVDirector(sample_config, min_shot_duration=2.0)
    # Rapid alternating banter
    words = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "word": "I completely disagree with that point."},
        {"start": 2.2, "end": 4.0, "speaker": "SPEAKER_01", "word": "No way, look at the benchmark numbers!"},
        {"start": 4.2, "end": 6.0, "speaker": "SPEAKER_00", "word": "The benchmarks are cherry-picked!"},
    ]

    shots = director.direct_clip(words, clip_start=0.0, clip_end=6.0, use_ollama=False)
    has_split = any(s["type"] == "split_stack" for s in shots)
    assert has_split, "Expected split_stack cut for rapid conversational debate"

def test_timeline_coverage_zero_gaps(sample_config):
    director = TVDirector(sample_config, min_shot_duration=2.0)
    words = [
        {"start": 2.0, "end": 5.0, "speaker": "SPEAKER_00", "word": "Starting at 2s"},
        {"start": 10.0, "end": 15.0, "speaker": "SPEAKER_01", "word": "Continuing after a pause"},
    ]

    shots = director.direct_clip(words, clip_start=0.0, clip_end=20.0, use_ollama=False)
    assert shots[0]["start"] == 0.0
    assert shots[-1]["end"] == 20.0
    for k in range(len(shots) - 1):
        assert shots[k]["end"] == shots[k + 1]["start"], f"Gap detected between shot {k} and {k+1}"

def test_undiarized_or_solo_dialogue_never_triggers_split_stack(sample_config):
    director = TVDirector(sample_config, min_shot_duration=2.0)
    # Words with no speaker tags (or all same speaker)
    words = [
        {"start": 0.0, "end": 5.0, "speaker": None, "word": "Talking about foundation models and AI."},
        {"start": 5.1, "end": 12.0, "speaker": None, "word": "It should always remain on a single focal camera."}
    ]
    shots = director.direct_clip(words, clip_start=0.0, clip_end=15.0, use_ollama=False)
    assert len(shots) == 1
    assert shots[0]["type"] == "single"
    assert shots[0]["camera"] == "zone_host"

