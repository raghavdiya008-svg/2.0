import os
import sys
import pytest
from fastapi.testclient import TestClient

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from server import app

client = TestClient(app)

def test_camera_zones_save_and_get():
    save_payload = {
        "file_name": "test_video.mp4",
        "source_width": 1920,
        "source_height": 1080,
        "mode": "podcast",
        "split_preference": "split_stack",
        "zones": [
            {
                "id": "zone_host",
                "label": "Host",
                "x": 0,
                "y": 0,
                "width": 960,
                "height": 1080,
                "speaker_label": "SPEAKER_00",
                "color": "#3b82f6"
            },
            {
                "id": "zone_guest",
                "label": "Guest",
                "x": 960,
                "y": 0,
                "width": 960,
                "height": 1080,
                "speaker_label": "SPEAKER_01",
                "color": "#10b981"
            }
        ]
    }

    res_save = client.post("/api/camera_zones/save", json=save_payload)
    assert res_save.status_code == 200
    data_save = res_save.json()
    assert data_save["status"] == "ok"
    assert len(data_save["config"]["zones"]) == 2

    res_get = client.get("/api/camera_zones/test_video.mp4")
    assert res_get.status_code == 200
    data_get = res_get.json()
    assert data_get["file_name"] == "test_video.mp4"
    assert len(data_get["zones"]) == 2
    assert data_get["zones"][0]["label"] == "Host"

def test_director_plan_endpoint():
    plan_payload = {
        "file_name": "test_video.mp4",
        "start": 0.0,
        "end": 10.0,
        "words": [
            {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00", "word": "Welcome to our technical podcast."},
            {"start": 4.5, "end": 9.0, "speaker": "SPEAKER_01", "word": "Thanks for having me today."}
        ],
        "use_ollama": False
    }

    res = client.post("/api/director/plan", json=plan_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "shots" in data
    assert len(data["shots"]) >= 1
    assert data["shots"][0]["start"] == 0.0
    assert data["shots"][-1]["end"] == 10.0
