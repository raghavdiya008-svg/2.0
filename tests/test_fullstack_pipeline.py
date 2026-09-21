import os
import sys
import unittest
import tempfile
import json
from unittest.mock import patch, MagicMock

# Ensure src is in sys.path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import dependency_manager
import media_downloader
import curation_engine
import pipeline
from server import app
from fastapi.testclient import TestClient


class TestFullstackPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_dependency_manager(self):
        """Verifies dependency checking and individual package lookup."""
        status = dependency_manager.check_dependencies()
        self.assertIsInstance(status, dict)
        self.assertIn("torch", status["modules"])
        self.assertIn("yt-dlp", status["modules"])
        self.assertIn("gdown", status["modules"])
        
        if dependency_manager.is_installed("torch"):
            self.assertTrue(dependency_manager.is_installed("torch"))
        else:
            self.assertTrue(dependency_manager.is_installed("fastapi"))
        self.assertFalse(dependency_manager.is_installed("non_existent_pkg_xyz_12345"))

    def test_02_media_downloader_url_detection(self):
        """Verifies URL classification for YouTube and Google Drive."""
        yt_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://youtube.com/shorts/dQw4w9WgXcQ",
        ]
        for url in yt_urls:
            self.assertTrue(media_downloader.is_youtube_url(url), f"Failed for {url}")
            self.assertFalse(media_downloader.is_gdrive_url(url))

        gdrive_urls = [
            "https://drive.google.com/file/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs/view",
            "https://drive.google.com/open?id=1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs",
        ]
        for url in gdrive_urls:
            self.assertTrue(media_downloader.is_gdrive_url(url), f"Failed for {url}")
            self.assertFalse(media_downloader.is_youtube_url(url))

        self.assertFalse(media_downloader.is_youtube_url("https://example.com/video.mp4"))
        self.assertFalse(media_downloader.is_gdrive_url("https://example.com/video.mp4"))

    def test_03_media_downloader_local_file(self):
        """Verifies local file pass-through in download_media."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_path = f.name
            f.write(b"mock video data")

        try:
            result = media_downloader.download_media(temp_path)
            self.assertEqual(os.path.abspath(result["file_path"]), os.path.abspath(temp_path))
            self.assertEqual(result["file_name"], os.path.basename(temp_path))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_04_md5_audio_stream_cache(self):
        """Verifies compute_audio_stream_md5 and global cache read/write."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_wav = f.name
            f.write(b"RIFF mock audio data header and pcm stream")

        try:
            md5_hash = pipeline.compute_audio_stream_md5(temp_wav)
            self.assertEqual(len(md5_hash), 32)

            test_key = f"test_key_{md5_hash[:8]}"
            test_data = {"words": [{"word": "hello", "start": 0.0, "end": 0.5}], "audio_md5": md5_hash}
            pipeline._save_audio_cache(test_key, test_data)

            loaded = pipeline._load_audio_cache(test_key)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["audio_md5"], md5_hash)
            self.assertEqual(len(loaded["words"]), 1)
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

    def test_05_curation_high_volume_and_durations(self):
        """Verifies duration constraints (30s-50s), volume targeting, and scores >= 75."""
        # 1. Short clip (<= 45s) returns 1 full cut
        short_cuts = curation_engine.get_viral_cuts([], [], 30.0)
        self.assertEqual(len(short_cuts), 1)
        self.assertEqual(short_cuts[0]["start"], 0.0)
        self.assertEqual(short_cuts[0]["end"], 30.0)
        self.assertGreaterEqual(short_cuts[0]["virality_score"], 75)

        # 2. 180s clip returns 3 cuts, each between 30s and 50s
        cuts_180 = curation_engine.get_viral_cuts([], [], 180.0)
        self.assertEqual(len(cuts_180), 3)
        for cut in cuts_180:
            dur = cut["end"] - cut["start"]
            self.assertGreaterEqual(dur, 30.0)
            self.assertLessEqual(dur, 50.0)
            self.assertGreaterEqual(cut["virality_score"], 75)

        # 3. 1 hour clip targets 40 clips
        curation_1hr = curation_engine.curate_video_with_warning([], [], 3600.0)
        self.assertEqual(curation_1hr["target_clips"], 40)
        self.assertEqual(len(curation_1hr["clips"]), 40)
        self.assertFalse(curation_1hr["low_yield_warning"])
        for cut in curation_1hr["clips"]:
            dur = cut["end"] - cut["start"]
            self.assertGreaterEqual(dur, 30.0)
            self.assertLessEqual(dur, 50.0)
            self.assertGreaterEqual(cut["virality_score"], 75)

        # 4. Low yield warning triggering on empty duration
        warn_result = curation_engine.curate_video_with_warning([], [], 0.0)
        self.assertTrue(warn_result["low_yield_warning"])

        # 5. Quality Fallback Message: "Only [X] high-quality viral hooks were found."
        # When words is empty on a 180s video and only 3 cuts are yielded vs expected target,
        # or when min_score filters out clips:
        with patch.object(curation_engine, "get_viral_cuts", return_value=[{"start": 0.0, "end": 35.0, "virality_score": 90}]):
            low_res = curation_engine.curate_video_with_warning([], [], 180.0)
            self.assertTrue(low_res["low_yield_warning"])
            self.assertEqual(low_res["message"], "Only 1 high-quality viral hooks were found.")

    def test_06_fastapi_server_endpoints(self):
        """Verifies core FastAPI endpoints: index, health, dependencies, and ingest validation."""
        # GET /
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers.get("content-type", ""))

        # GET /health
        res_health = self.client.get("/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json()["status"], "ok")

        # POST /api/check_dependencies
        res_deps = self.client.post("/api/check_dependencies", json={"auto_install": False})
        self.assertEqual(res_deps.status_code, 200)
        self.assertIn("dependencies", res_deps.json())
        self.assertIn("all_installed", res_deps.json())
        self.assertIn("missing", res_deps.json())

        # POST /api/ingest validation
        res_ingest_bad = self.client.post("/api/ingest", json={})
        self.assertEqual(res_ingest_bad.status_code, 400)

        # GET /list_assets
        res_assets = self.client.get("/list_assets")
        self.assertEqual(res_assets.status_code, 200)
        self.assertIn("videos", res_assets.json())

    def test_07_fastapi_batch_render_workflow(self):
        """Verifies batch submission, batch status polling, and zip download endpoint."""
        import server
        dummy_video = os.path.join(server.INPUTS_DIR, "batch_test_vid.mp4")
        os.makedirs(server.INPUTS_DIR, exist_ok=True)
        with open(dummy_video, "w") as f:
            f.write("mock video data")

        try:
            sample_clips = [
                {"start": 0.0, "end": 35.0, "hook_sentence": "Hook 1", "virality_score": 90},
                {"start": 35.0, "end": 70.0, "hook_sentence": "Hook 2", "virality_score": 85},
            ]
            batch_req = {
                "file_name": "batch_test_vid.mp4",
                "clips": sample_clips,
                "speed": 1.12
            }
            res_batch = self.client.post("/api/render_batch", json=batch_req)
            self.assertEqual(res_batch.status_code, 200)
            data = res_batch.json()
            self.assertEqual(data["status"], "queued")
            self.assertEqual(data["total_clips"], 2)
            batch_id = data["batch_id"]

            # Query status
            res_status = self.client.get(f"/api/batch_status/{batch_id}")
            self.assertEqual(res_status.status_code, 200)
            status_data = res_status.json()
            self.assertEqual(status_data["total"], 2)
            self.assertIn(status_data["status"], ["processing", "completed", "failed"])

            # Query progress
            res_prog = self.client.get(f"/api/progress/test_proc_123")
            self.assertEqual(res_prog.status_code, 200)
            self.assertEqual(res_prog.json()["stage"], "idle")
        finally:
            if os.path.exists(dummy_video):
                try:
                    os.remove(dummy_video)
                except OSError:
                    pass

    def test_08_shutil_make_archive_zip_export(self):
        """Verifies create_zip_archive uses shutil.make_archive to produce final_reels.zip."""
        import engine_ffmpeg
        import zipfile
        with tempfile.TemporaryDirectory() as td:
            file1 = os.path.join(td, "clip_1.mp4")
            file2 = os.path.join(td, "clip_2.mp4")
            with open(file1, "w") as f:
                f.write("clip1 data")
            with open(file2, "w") as f:
                f.write("clip2 data")

            out_zip = os.path.join(td, "final_reels.zip")
            generated_zip = engine_ffmpeg.create_zip_archive([file1, file2], output_zip_path=out_zip)
            self.assertTrue(os.path.isfile(generated_zip))
            self.assertTrue(generated_zip.endswith("final_reels.zip"))

            with zipfile.ZipFile(generated_zip, "r") as zf:
                names = zf.namelist()
                self.assertIn("clip_1.mp4", names)
                self.assertIn("clip_2.mp4", names)

    def test_09_dependency_streaming_endpoint(self):
        """Verifies GET /api/dependencies/stream returns an SSE streaming response."""
        res = self.client.get("/api/dependencies/stream?packages=nonexistent_test_pkg")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/event-stream", res.headers.get("content-type", ""))

    def test_10_cache_bypass_whisperx_and_vad(self):
        """Verifies cache hit skips WhisperX alignment and Silero VAD during /api/curate."""
        import server
        dummy_vid = os.path.join(server.INPUTS_DIR, "cached_sample_vid.mp4")
        with open(dummy_vid, "w") as f:
            f.write("dummy video data")

        mock_hash = "mock_md5_hash_1234567890abcdef"
        mock_cached_data = {
            "words": [{"word": "cached", "start": 0.0, "end": 1.0}],
            "silence_gaps": [],
            "audio_md5": mock_hash
        }
        pipeline._save_audio_cache(mock_hash, mock_cached_data)

        try:
            with patch("subprocess.run") as mock_subproc:
                mock_subproc.return_value = MagicMock(
                    returncode=0,
                    stdout=json.dumps({"format": {"duration": "35.0"}})
                )
                with patch("audio_intelligence.run_whisperx_alignment") as mock_whisperx:
                    res = self.client.post("/api/curate", json={
                        "file_name": "cached_sample_vid.mp4",
                        "audio_md5": mock_hash,
                        "min_score": 75
                    })
                    self.assertEqual(res.status_code, 200)
                    data = res.json()
                    self.assertTrue(data["cache_hit"])
                    # WhisperX should NOT have been called due to cache hit
                    mock_whisperx.assert_not_called()
        finally:
            if os.path.exists(dummy_vid):
                os.remove(dummy_vid)

    def test_11_memory_sync_endpoint_and_vault(self):
        """Verifies GET and POST /api/memory/sync updates unified memory vault."""
        res = self.client.get("/api/memory/sync")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertIn("system_audit", data)
        self.assertIn("pipeline_db", data)
        self.assertIn("licensing", data)

        import memory_sync
        snapshot = memory_sync.compile_full_memory_snapshot()
        self.assertIn("tracker_id", snapshot)
        self.assertIn("system_audit", snapshot)
        self.assertIn("phases_compliance", snapshot["system_audit"])


if __name__ == "__main__":
    unittest.main()

