import sys
import os

# Add src to sys.path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_path)

def test_imports():
    print("Testing imports...")
    try:
        from videocuts.config import Config
        from videocuts.utils.system import format_timestamp
        from videocuts.audio.transcription import parse_srt_to_segments
        from videocuts.video.tracking import analyze_clip_faces
        from videocuts.caption.generators import write_clip_ass
        from videocuts.main import run_pipeline
        print("✅ All imports successful!")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        sys.exit(1)

def test_utils():
    print("Testing utils...")
    from videocuts.utils.system import format_timestamp
    ts = format_timestamp(3661.123)
    if ts == "01:01:01,123":
        print("✅ format_timestamp - OK")
    else:
        print(f"❌ format_timestamp - FAILED (got {ts})")
        sys.exit(1)

def test_config():
    print("Testing config...")
    from videocuts.config import Config
    cfg = Config()
    if cfg.video.target_width == 1080:
        print("✅ Config - OK")
    else:
        print("❌ Config - FAILED")
        sys.exit(1)

def test_project_paths():
    print("Testing project paths...")
    from videocuts.config import Config
    cfg = Config()
    cfg.paths.project_name = "test_project"
    
    if "test_project" in cfg.paths.srt_file and "test_project" in cfg.paths.output_dir:
        print("✅ Project Paths - OK")
    else:
        print(f"❌ Project Paths - FAILED (srt={cfg.paths.srt_file}, out={cfg.paths.output_dir})")
        sys.exit(1)

if __name__ == "__main__":
    print("--- VideoCuts Refactoring Verification ---")
    test_imports()
    test_utils()
    test_config()
    test_project_paths()
    print("--- Verification Complete ---")
