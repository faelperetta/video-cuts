import argparse
import logging
from videocuts.config import Config
from videocuts.main import run_pipeline

logger = logging.getLogger(__name__)
from videocuts.llm.selector import detect_llm_availability

def main():
    parser = argparse.ArgumentParser(description="Video Highlights Generator")
    parser.add_argument("--hook", action="store_true", help="Enable auto-hook overlay")
    parser.add_argument("--fast", action="store_true", help="Fast mode for development")
    parser.add_argument("--use-llm", action="store_true", help="Use OpenAI for clip selection")
    parser.add_argument("--llm-model", type=str, help="Override LLM model")
    parser.add_argument("--layout", choices=["auto", "single", "split", "wide"], help="Override layout mode")
    parser.add_argument("--content-type", choices=["coding", "fitness", "gaming"], help="Override content type")
    parser.add_argument("--input", type=str, help="Override input video path")
    parser.add_argument("--project", type=str, help="Project name (creates a project folder)")
    parser.add_argument("--num-clips", type=int, help="Number of highlights")
    parser.add_argument("--language", type=str, help="Override language detection")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    cfg = Config()
    
    if args.debug: cfg.debug = True
    if args.hook: cfg.hook.enabled = True
    if args.fast: cfg.enable_fast_mode()
    if args.layout: cfg.layout.mode = args.layout
    if args.content_type: cfg.content_type = args.content_type
    if args.input: cfg.paths.input_video = args.input
    if args.project: cfg.paths.project_name = args.project
    if args.num_clips: cfg.highlight.num_highlights = args.num_clips
    if args.language: cfg.model.whisper_language = args.language
    
    if args.use_llm or detect_llm_availability():
        if detect_llm_availability():
            cfg.llm.enabled = True
            if args.llm_model: cfg.llm.model = args.llm_model
        # The original `elif args.use_llm:` block is replaced by the following `if` statement
        # which checks if LLM was intended to be used (either by --use-llm or auto-detection)
        # but an API key is missing.
        if cfg.llm.enabled and not detect_llm_availability():
            logger.warning("--use-llm specified but OPENAI_API_KEY missing. Using local scoring.")
            cfg.llm.enabled = False
            
    run_pipeline(cfg)

if __name__ == "__main__":
    main()
