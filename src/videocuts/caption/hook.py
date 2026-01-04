from typing import List, Dict, Optional
from videocuts.config import Config
from videocuts.utils.font import get_font_path_with_fallback

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

def detect_hook_phrase(
    segments: List[Dict],
    clip_start: float,
    clip_end: float,
    cfg: Config,
    keywords: Optional[List[str]] = None
) -> Optional[Dict]:
    """Detect a catchy hook phrase from the first few seconds of a clip."""
    hook_keywords = keywords if keywords is not None else []
    
    scan_seconds = cfg.hook.scan_seconds
    min_words = cfg.hook.min_words
    max_words = cfg.hook.max_words
    display_dur = cfg.hook.display_duration
    fade_in = cfg.hook.fade_in
    
    scan_end = clip_start + scan_seconds
    scan_segments = []
    
    for seg in segments:
        seg_start, seg_end = seg.get("start", 0), seg.get("end", 0)
        if seg_end >= clip_start and seg_start <= scan_end:
            text = seg.get("text", "").strip()
            if text:
                scan_segments.append({
                    "text": text,
                    "start": max(seg_start - clip_start, 0),
                    "end": min(seg_end - clip_start, clip_end - clip_start)
                })
    
    if not scan_segments: return None
    
    for seg in scan_segments:
        text_lower = seg["text"].lower()
        for keyword in hook_keywords:
            if keyword in text_lower:
                hook_text = _extract_hook_around_keyword(seg["text"], keyword, max_words)
                if hook_text and len(hook_text.split()) >= min_words:
                    return {
                        "text": hook_text,
                        "start": seg["start"],
                        "end": min(seg["end"], display_dur + fade_in),
                        "has_keyword": True
                    }
    
    combined_text = " ".join(seg["text"] for seg in scan_segments)
    words = combined_text.split()
    if len(words) < min_words: return None
    
    hook_words = []
    for word in words[:max_words]:
        hook_words.append(word)
        if word.endswith(('.', '!', '?')) and len(hook_words) >= min_words: break
    
    if len(hook_words) < min_words: hook_words = words[:max_words]
    hook_text = " ".join(hook_words).rstrip('.,;:')
    
    if len(hook_text.split()) >= min_words:
        return {
            "text": hook_text,
            "start": scan_segments[0]["start"],
            "end": min(scan_segments[0]["end"], display_dur + fade_in),
            "has_keyword": False
        }
    return None

def _extract_hook_around_keyword(text: str, keyword: str, max_words: int) -> str:
    text_lower = text.lower()
    keyword_pos = text_lower.find(keyword)
    if keyword_pos == -1: return text[:max_words * 10]
    
    words = text.split()
    keyword_words = keyword.split()
    char_count, keyword_start_idx = 0, 0
    for i, word in enumerate(words):
        word_end = char_count + len(word)
        if char_count <= keyword_pos < word_end:
            keyword_start_idx = i
            break
        char_count = word_end + 1
    
    kw_len = len(keyword_words)
    before = max(0, (max_words - kw_len) // 2)
    after = max_words - kw_len - before
    return " ".join(words[max(0, keyword_start_idx - before):min(len(words), keyword_start_idx + kw_len + after)])

def create_hook_image(text: str, output_path: str, cfg: Config) -> str:
    if Image is None: return ""
    target_w, target_h = cfg.video.target_width, cfg.video.target_height
    font_size, padding, radius = cfg.hook.font_size, cfg.hook.box_padding, cfg.hook.corner_radius
    bg_color, text_color = cfg.hook.bg_color, cfg.hook.primary_color
    font_path = get_font_path_with_fallback(cfg.hook.font_name, cfg.hook.font_path, bold=True)
    
    try: font = ImageFont.truetype(font_path, font_size)
    except: font = ImageFont.load_default()

    max_text_w = target_w - 100 - (2 * padding)
    words = text.split()
    lines, curr_line = [], []
    for word in words:
        test = " ".join(curr_line + [word])
        if font.getbbox(test)[2] - font.getbbox(test)[0] <= max_text_w: curr_line.append(word)
        else:
            if curr_line: lines.append(" ".join(curr_line))
            curr_line = [word]
    if curr_line: lines.append(" ".join(curr_line))
    
    line_h = int(font_size * 1.3)
    total_h = len(lines) * line_h
    max_line_w = max((font.getbbox(l)[2] - font.getbbox(l)[0] for l in lines)) if lines else 0
    box_w, box_h = max_line_w + (2 * padding), total_h + (2 * padding)
    
    img = Image.new('RGBA', (target_w, target_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    bx, by = (target_w - box_w) // 2, cfg.hook.position_y
    
    alpha = int(255 * cfg.hook.bg_opacity)
    fill = (255, 255, 255, alpha) if bg_color.lower() == 'white' else (0, 0, 0, alpha) if bg_color.lower() == 'black' else (255, 255, 255, alpha)
    draw.rounded_rectangle([(bx, by), (bx + box_w, by + box_h)], radius=radius, fill=fill)
    
    txt_fill = (0, 0, 0, 255) if text_color.lower() == 'black' else (255, 255, 255, 255) if text_color.lower() == 'white' else (0, 0, 0, 255)
    curr_y = by + padding
    for line in lines:
        lw = font.getbbox(line)[2] - font.getbbox(line)[0]
        draw.text((bx + (box_w - lw) // 2, curr_y), line, font=font, fill=txt_fill)
        curr_y += line_h
    
    img.save(output_path)
    return output_path

def generate_hook_overlay_filter(hook_text: str, cfg: Config) -> str:
    target_w, target_h = cfg.video.target_width, cfg.video.target_height
    font_size, padding, box_border_w = cfg.hook.font_size, cfg.hook.box_padding, cfg.hook.box_border_w
    bg_color, bg_opacity, text_color = cfg.hook.bg_color, cfg.hook.bg_opacity, cfg.hook.primary_color
    pos_y, shadow_color = cfg.hook.position_y, cfg.hook.shadow_color
    fi, fd, fo = cfg.hook.fade_in, cfg.hook.display_duration, cfg.hook.fade_out
    
    hm = 50
    usable_w = target_w - (2 * hm) - (2 * padding)
    chars_per_line = max(10, int(usable_w / (font_size * 0.6)) - 2)
    
    words = hook_text.strip().split()
    lines, curr_line, curr_len = [], [], 0
    for word in words:
        if curr_len + len(word) + (1 if curr_line else 0) <= chars_per_line:
            curr_line.append(word)
            curr_len += len(word) + (1 if len(curr_line) > 1 else 0)
        else:
            if curr_line: lines.append(" ".join(curr_line))
            curr_line, curr_len = [word], len(word)
            if len(lines) >= 2:
                last_line = " ".join(words[words.index(word):])
                if len(last_line) > chars_per_line: last_line = last_line[:chars_per_line - 3].rstrip() + "..."
                lines.append(last_line)
                curr_line = []
                break
    if curr_line: lines.append(" ".join(curr_line))
    
    escaped_text = "\n".join(lines).replace("\\", "\\\\\\\\").replace(":", "\\:").replace("'", "\\'").replace(",", "\\,")
    alpha_expr = f"if(lt(t\\,{fi})\\,t/{fi}\\,if(lt(t\\,{fi+fd})\\,1\\,if(lt(t\\,{fi+fd+fo})\\,(({fi+fd+fo}-t)/{fo})\\,0)))"
    font_file = get_font_path_with_fallback(cfg.hook.hook_font_name if hasattr(cfg.hook, 'hook_font_name') else cfg.hook.font_name, cfg.hook.font_path, bold=True).replace(":", "\\:")
    
    return (f"drawtext=text='{escaped_text}'\\:fontfile={font_file}\\:fontsize={font_size}\\:fontcolor={text_color}\\:x=max({hm}\\,(w-text_w)/2)\\:y={pos_y}\\:alpha={alpha_expr}\\:box=1\\:boxcolor={bg_color}@{bg_opacity}\\:boxborderw={box_border_w}\\:shadowcolor={shadow_color}\\:shadowx=0\\:shadowy=0")
