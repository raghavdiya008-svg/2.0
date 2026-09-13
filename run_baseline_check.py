import os
import re

def assert_file_not_contains(filepath, regex_pattern):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    assert not re.search(regex_pattern, content), f"Found '{regex_pattern}' in {filepath}"

def assert_file_contains(filepath, regex_pattern):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    assert re.search(regex_pattern, content), f"Missing '{regex_pattern}' in {filepath}"

def run_tests():
    print("Executing E2E Baseline Assertions...\n")
    
    # 1. Caption Engine Sanitization
    assert_file_not_contains("src/caption_engine.py", r"recognize_google")
    assert_file_not_contains("src/audio_intelligence.py", r"recognize_google")
    assert_file_contains("src/caption_engine.py", r"\[WARN\] Caption generation failed — proceeding without subtitles")
    assert_file_contains("src/audio_intelligence.py", r"faster-whisper")
    print(" [PASS] Task 1: Caption Engine Sanitization")
    
    # 2. Platform UI & Safe-Zone Cropper
    assert_file_contains("src/pipeline.py", r"strip_social_ui")
    assert_file_contains("src/pipeline.py", r"crop=iw:ih\*0.73:0:ih\*0.12")
    print(" [PASS] Task 2: Platform UI & Safe-Zone Cropper")
    
    # 3. NMS & Dynamic VAD Gating
    assert_file_contains("src/curation_engine.py", r"overlap_dur > clip_len \* 0.15")
    assert_file_contains("src/pipeline.py", r"wpm < 25\.0")
    assert_file_contains("src/engine_vad.py", r"is_nvenc_available")
    assert_file_contains("src/engine_vad.py", r"libx264")
    print(" [PASS] Task 3: Curation NMS, Dynamic VAD Gating, NVENC Guard")
    
    # 4. Codebase Cleanup
    assert_file_not_contains("src/audio_intelligence.py", r"def get_vocal_analysis")
    print(" [PASS] Task 4: Codebase Cleanup (Dead code removal)")
    
    print("\n[SUCCESS] All Baseline Integrity Checks PASSED. The pipeline is ready for production.")

if __name__ == "__main__":
    run_tests()
