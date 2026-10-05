import os
import sys
import pytest

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from camera_framing import CameraZone, CameraFramingConfig, suggest_camera_zones

def test_camera_zone_initialization_and_clamp():
    zone = CameraZone(
        id="zone_host",
        label="Host",
        x=-10,
        y=-5,
        width=1001,  # Odd width
        height=959,   # Odd height
        speaker_label="Host"
    )
    assert zone.x == 0
    assert zone.y == 0
    # Must be even integers for FFmpeg
    assert zone.width % 2 == 0
    assert zone.height % 2 == 0
    assert zone.width >= 1000
    assert zone.aspect_ratio > 0

def test_camera_framing_config_serialization(tmp_path):
    z1 = CameraZone("zone_host", "Host", 0, 0, 960, 1080, speaker_label="SPEAKER_00")
    z2 = CameraZone("zone_guest", "Guest", 960, 0, 960, 1080, speaker_label="SPEAKER_01")
    z3 = CameraZone("zone_wide", "Wide", 0, 0, 1920, 1080, is_wide=True)

    config = CameraFramingConfig(
        file_name="podcast_ep1.mp4",
        source_width=1920,
        source_height=1080,
        mode="podcast",
        split_preference="split_stack",
        zones=[z1, z2, z3]
    )

    save_path = config.save(config_dir=str(tmp_path))
    assert os.path.isfile(save_path)

    loaded = CameraFramingConfig.load("podcast_ep1.mp4", config_dir=str(tmp_path))
    assert loaded is not None
    assert loaded.file_name == "podcast_ep1.mp4"
    assert len(loaded.zones) == 3
    assert loaded.get_zone("zone_host").label == "Host"
    assert loaded.get_zone_by_speaker("SPEAKER_01").id == "zone_guest"
    assert loaded.get_wide_zone().id == "zone_wide"

def test_suggest_camera_zones_fallback():
    # Calling with non-existent file returns default podcast zones without crashing
    config = suggest_camera_zones("non_existent_file.mp4")
    assert config is not None
    assert len(config.zones) >= 2
    assert config.get_host_zone() is not None
    assert config.get_guest_zone() is not None
