import os
import subprocess
import tempfile
import logging
from typing import List, Dict, Optional
from videocuts.config import Config
from videocuts.utils.system import run_ffmpeg_cmd
from videocuts.video.tracking import crop_x_expression_for_segments

logger = logging.getLogger(__name__)

def create_vertical_with_subs(
    input_video: str,
    sub_path: str,
    output_video: str,
    start: float,
    duration: float,
    cfg: Config,
    crop_x_expr: Optional[str] = None,
    hook_text: Optional[str] = None
) -> None:
    """Seek, cut, scale, crop, and burn subtitles into a vertical clip."""
    if duration <= 0:
        raise ValueError(f"Clip duration must be positive, got {duration} (start={start})")

    target_w = cfg.video.target_width
    target_h = cfg.video.target_height
    fps = cfg.video.output_fps

    cmd = ["ffmpeg", "-y"]
    cmd += ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", input_video]
    
    if crop_x_expr is None:
        x_expr = f"(in_w-{target_w})/2"
    else:
        x_expr = crop_x_expr.replace("'", "\\'")

    sub_ext = os.path.splitext(sub_path)[1].lower()
    escaped_sub_path = sub_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    if sub_ext == ".ass":
        sub_filter = f"ass='{escaped_sub_path}'"
    else:
        sub_filter = f"subtitles='{escaped_sub_path}'"

    base_vf_chain = (
        f"[0:v]scale=-2:{target_h},"
        f"crop={target_w}:{target_h}:'{x_expr}':0,"
        f"{sub_filter}"
    )
    
    from videocuts.caption.hook import create_hook_image, generate_hook_overlay_filter
    
    hook_img_path = None
    if hook_text and cfg.hook.enabled:
        temp_hook_img = f"temp_hook_{int(start)}.png"
        hook_img_path = create_hook_image(hook_text, temp_hook_img, cfg)
    
    if hook_img_path and os.path.exists(hook_img_path):
        cmd += ["-loop", "1", "-i", hook_img_path]
        
        fade_in_dur = cfg.hook.fade_in
        visible_dur = cfg.hook.display_duration
        fade_out_dur = cfg.hook.fade_out
        
        visible_end = fade_in_dur + visible_dur
        fade_out_end = visible_end + fade_out_dur
        
        fc = (
            f"{base_vf_chain}[main];"
            f"[1:v]format=rgba,fade=in:st=0:d={fade_in_dur}:alpha=1,"
            f"fade=out:st={visible_end}:d={fade_out_dur}:alpha=1[hook];"
            f"[main][hook]overlay=0:0:shortest=1[out]"
        )
        cmd += ["-filter_complex", fc, "-map", "[out]", "-map", "0:a"]
    else:
        vf = base_vf_chain
        if hook_text and cfg.hook.enabled:
             hook_filter = generate_hook_overlay_filter(hook_text, cfg)
             vf += f",{hook_filter}"
        cmd += ["-vf", vf]

    cmd += [
        "-r", str(fps),
        "-c:v", "libx264",
        "-preset", cfg.video.ffmpeg_preset,
        "-crf", str(cfg.video.ffmpeg_crf),
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_video
    ]

    run_ffmpeg_cmd(cmd)
    
    if hook_img_path and os.path.exists(hook_img_path):
        try:
            os.remove(hook_img_path)
        except:
            pass

def create_split_screen_clip(
    input_video: str,
    sub_path: str,
    output_video: str,
    start: float,
    duration: float,
    left_center: float,
    right_center: float,
    cfg: Config,
    hook_text: Optional[str] = None
) -> None:
    """Create a vertical split-screen clip showing two speakers stacked vertically."""
    target_w = cfg.video.target_width
    target_h = cfg.video.target_height
    fps = cfg.video.output_fps
    split_gap = cfg.layout.split_gap

    panel_h = (target_h - split_gap) // 2
    panel_w = target_w
    
    cmd = ["ffmpeg", "-y"]
    cmd += ["-ss", f"{start:.3f}", "-i", input_video, "-t", f"{duration:.3f}"]
    
    escaped_sub_path = sub_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    sub_ext = os.path.splitext(sub_path)[1].lower()
    if sub_ext == ".ass":
        sub_filter = f"ass='{escaped_sub_path}'"
    else:
        sub_filter = f"subtitles='{escaped_sub_path}'"
    
    if hook_text and cfg.hook.enabled:
        from videocuts.caption.hook import generate_hook_overlay_filter
        hook_filter = generate_hook_overlay_filter(hook_text, cfg)
        sub_filter = f"{sub_filter},{hook_filter}"
    
    scale_h = panel_h
    left_x = f"clip(({left_center:.4f})*iw-{panel_w}/2,0,iw-{panel_w})"
    right_x = f"clip(({right_center:.4f})*iw-{panel_w}/2,0,iw-{panel_w})"
    face_y_offset = 0.1
    
    filter_complex = (
        f"[0:v]scale=-2:{scale_h*2}[src];"
        f"[src]crop={panel_w}:{panel_h}:'{left_x}':{int(scale_h*2*face_y_offset)}[top];"
        f"[src]crop={panel_w}:{panel_h}:'{right_x}':{int(scale_h*2*face_y_offset)}[bottom];"
        f"color=black:{target_w}x{split_gap}:d={duration}[gap];"
        f"[top][gap][bottom]vstack=inputs=3[stacked];"
        f"[stacked]{sub_filter}[out]"
    )
    
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-map", "0:a?",
        "-r", str(fps),
        "-c:v", "libx264",
        "-preset", cfg.video.ffmpeg_preset,
        "-crf", str(cfg.video.ffmpeg_crf),
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_video
    ]
    
    run_ffmpeg_cmd(cmd)

def create_wide_shot_clip(
    input_video: str,
    sub_path: str,
    output_video: str,
    start: float,
    duration: float,
    center: float,
    cfg: Config,
    hook_text: Optional[str] = None
) -> None:
    """Create a vertical clip with wide shot showing both speakers (letterbox style)."""
    target_w = cfg.video.target_width
    target_h = cfg.video.target_height
    fps = cfg.video.output_fps

    cmd = ["ffmpeg", "-y"]
    cmd += ["-ss", f"{start:.3f}", "-i", input_video, "-t", f"{duration:.3f}"]
    
    escaped_sub_path = sub_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    sub_ext = os.path.splitext(sub_path)[1].lower()
    if sub_ext == ".ass":
        sub_filter = f"ass='{escaped_sub_path}'"
    else:
        sub_filter = f"subtitles='{escaped_sub_path}'"
    
    if hook_text and cfg.hook.enabled:
        from videocuts.caption.hook import generate_hook_overlay_filter
        hook_filter = generate_hook_overlay_filter(hook_text, cfg)
        sub_filter = f"{sub_filter},{hook_filter}"
    
    filter_complex = (
        f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
        f"crop={target_w}:{target_h},"
        f"boxblur=20:5[bg];"
        f"[0:v]scale={target_w}:-2:force_original_aspect_ratio=decrease[strip];"
        f"[bg][strip]overlay=(W-w)/2:(H-h)/2[comp];"
        f"[comp]{sub_filter}[out]"
    )
    
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-map", "0:a?",
        "-r", str(fps),
        "-c:v", "libx264",
        "-preset", cfg.video.ffmpeg_preset,
        "-crf", str(cfg.video.ffmpeg_crf),
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_video
    ]
    
    run_ffmpeg_cmd(cmd)

def create_multi_layout_clip(
    input_video: str,
    sub_path: str,
    output_video: str,
    start: float,
    duration: float,
    layout_segments: List[Dict],
    speaker_segments: List[Dict],
    clip_start: float,
    clip_end: float,
    cfg: Config,
    hook_text: Optional[str] = None
) -> None:
    """Create a clip with dynamic layout switching."""
    target_w = cfg.video.target_width
    target_h = cfg.video.target_height
    fps = cfg.video.output_fps

    if not layout_segments:
        crop_expr = crop_x_expression_for_segments(speaker_segments, clip_start, clip_end, target_w)
        create_vertical_with_subs(input_video, sub_path, output_video, start, duration, cfg, crop_expr, hook_text)
        return
    
    if len(layout_segments) == 1:
        seg = layout_segments[0]
        if seg["layout"] == "split":
            create_split_screen_clip(input_video, sub_path, output_video, start, duration, seg.get("left_center", 0.25), seg.get("right_center", 0.75), cfg, hook_text)
        elif seg["layout"] == "wide":
            create_wide_shot_clip(input_video, sub_path, output_video, start, duration, seg.get("center", 0.5), cfg, hook_text)
        else:
            crop_expr = crop_x_expression_for_segments(speaker_segments, clip_start, clip_end, target_w)
            create_vertical_with_subs(input_video, sub_path, output_video, start, duration, cfg, crop_expr, hook_text)
        return
    
    with tempfile.TemporaryDirectory() as temp_dir:
        segment_files = []
        for i, seg in enumerate(layout_segments):
            seg_start, seg_end = seg["start"], seg["end"]
            seg_layout = seg["layout"]
            seg_duration = seg_end - seg_start
            temp_output = os.path.join(temp_dir, f"segment_{i:03d}.mp4")
            
            if seg_layout == "split":
                _render_split_segment(input_video, temp_output, seg_start, seg_duration, seg.get("left_center", 0.25), seg.get("right_center", 0.75), cfg)
            elif seg_layout == "wide":
                _render_wide_segment(input_video, temp_output, seg_start, seg_duration, seg.get("center", 0.5), cfg)
            else:
                crop_expr = _get_crop_for_time_range(speaker_segments, seg_start, seg_end, clip_start, target_w)
                _render_single_segment(input_video, temp_output, seg_start, seg_duration, crop_expr, cfg)
            segment_files.append(temp_output)
        
        concat_list = os.path.join(temp_dir, "concat_list.txt")
        with open(concat_list, "w") as f:
            for seg_file in segment_files:
                f.write(f"file '{seg_file}'\n")
        
        temp_concat = os.path.join(temp_dir, "concat_nosubs.mp4")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", temp_concat], check=True, capture_output=True)
        
        escaped_sub_path = sub_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        sub_filter = f"ass='{escaped_sub_path}'" if sub_path.endswith(".ass") else f"subtitles='{escaped_sub_path}'"
        
        vf = sub_filter
        if hook_text and cfg.hook.enabled:
            from videocuts.caption.hook import generate_hook_overlay_filter
            hook_filter = generate_hook_overlay_filter(hook_text, cfg)
            vf += f",{hook_filter}"
        
        run_ffmpeg_cmd(["ffmpeg", "-y", "-i", temp_concat, "-vf", vf, "-c:v", "libx264", "-preset", cfg.video.ffmpeg_preset, "-crf", str(cfg.video.ffmpeg_crf), "-c:a", "copy", "-movflags", "+faststart", output_video])

def _render_single_segment(input_video: str, output_path: str, start: float, duration: float, crop_x_expr: Optional[str], cfg: Config) -> None:
    target_w, target_h, fps = cfg.video.target_width, cfg.video.target_height, cfg.video.output_fps
    x_expr = crop_x_expr.replace("'", "\\'") if crop_x_expr else f"(in_w-{target_w})/2"
    cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", input_video, "-t", f"{duration:.3f}", "-vf", f"scale=-2:{target_h},crop={target_w}:{target_h}:'{x_expr}':0", "-r", str(fps), "-c:v", "libx264", "-preset", cfg.video.ffmpeg_preset, "-crf", str(cfg.video.ffmpeg_crf), "-c:a", "aac", "-b:a", "128k", output_path]
    subprocess.run(cmd, check=True, capture_output=True)

def _render_split_segment(input_video: str, output_path: str, start: float, duration: float, left_center: float, right_center: float, cfg: Config) -> None:
    target_w, target_h, fps = cfg.video.target_width, cfg.video.target_height, cfg.video.output_fps
    panel_h = (target_h - cfg.layout.split_gap) // 2
    panel_w = target_w
    scale_h = panel_h
    left_x = f"clip(({left_center:.4f})*iw-{panel_w}/2,0,iw-{panel_w})"
    right_x = f"clip(({right_center:.4f})*iw-{panel_w}/2,0,iw-{panel_w})"
    face_y_offset = 0.1
    filter_complex = (f"[0:v]scale=-2:{scale_h*2}[src];"
                      f"[src]crop={panel_w}:{panel_h}:'{left_x}':{int(scale_h*2*face_y_offset)}[top];"
                      f"[src]crop={panel_w}:{panel_h}:'{right_x}':{int(scale_h*2*face_y_offset)}[bottom];"
                      f"color=black:{target_w}x{cfg.layout.split_gap}:d={duration}[gap];"
                      f"[top][gap][bottom]vstack=inputs=3[out]")
    cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", input_video, "-t", f"{duration:.3f}", "-filter_complex", filter_complex, "-map", "[out]", "-map", "0:a?", "-r", str(fps), "-c:v", "libx264", "-preset", cfg.video.ffmpeg_preset, "-crf", str(cfg.video.ffmpeg_crf), "-c:a", "aac", "-b:a", "128k", output_path]
    subprocess.run(cmd, check=True, capture_output=True)

def _render_wide_segment(input_video: str, output_path: str, start: float, duration: float, center: float, cfg: Config) -> None:
    target_w, target_h, fps = cfg.video.target_width, cfg.video.target_height, cfg.video.output_fps
    filter_complex = (f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},boxblur=20:5[bg];"
                      f"[0:v]scale={target_w}:-2:force_original_aspect_ratio=decrease[strip];"
                      f"[bg][strip]overlay=(W-w)/2:(H-h)/2[out]")
    cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", input_video, "-t", f"{duration:.3f}", "-filter_complex", filter_complex, "-map", "[out]", "-map", "0:a?", "-r", str(fps), "-c:v", "libx264", "-preset", cfg.video.ffmpeg_preset, "-crf", str(cfg.video.ffmpeg_crf), "-c:a", "aac", "-b:a", "128k", output_path]
    subprocess.run(cmd, check=True, capture_output=True)

def _get_crop_for_time_range(speaker_segments: List[Dict], seg_start: float, seg_end: float, clip_start: float, target_w: int) -> Optional[str]:
    if not speaker_segments: return None
    relevant_segments = []
    for spk_seg in speaker_segments:
        if spk_seg["end"] <= seg_start or spk_seg["start"] >= seg_end: continue
        relevant_segments.append({"start": max(spk_seg["start"], seg_start), "end": min(spk_seg["end"], seg_end), "center": spk_seg["center"]})
    if not relevant_segments: return None
    expr = f"(in_w-{target_w})/2"
    for seg in reversed(relevant_segments):
        rel_start, rel_end = max(seg["start"] - seg_start, 0.0), min(seg["end"] - seg_start, seg_end - seg_start)
        if rel_end <= rel_start: continue
        center_expr = f"clip(({seg['center']:.4f})*in_w-{target_w}/2,0,in_w-{target_w})"
        expr = f"if(between(t,{rel_start:.3f},{rel_end:.3f}),{center_expr},{expr})"
    return expr
