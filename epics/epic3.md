# EPIC 3 — Clip Candidate Discovery (Start–Middle–End)

This epic generates **ranked short clip candidates** (15–60s) from long-form multi-speaker podcast content. Output clips should have a clear **hook (start)**, **substance (middle)**, and **resolution/punch (end)**. V1 uses a **hybrid approach**: deterministic heuristics + optional LLM reranking.

---

## US-3.0 — Build an analysis-ready timeline

**As a** system  
**I want to** build a unified timeline from transcript + diarization + audio features  
**So that** clip discovery can be data-driven and reproducible

### Default Technical Design
Inputs:
- `transcript.json` (segments + optional word timings + speaker labels from EPIC 2)
- `diarization.json`
- `audio_16k_mono.wav`

Compute and persist:
- `timeline.json` with:
  - `segments[]` (from transcript) enriched with:
    - `speaker`
    - `speaker_confidence`
    - `has_overlap`
    - `rms_energy` (audio loudness proxy per segment)
    - `speech_rate_wps` (words per second; if words unavailable, estimate via chars/sec)
    - `silence_before_s` and `silence_after_s` (using VAD or energy threshold)
- `audio_features.json` (optional, but useful for debugging):
  - global RMS
  - per-1s RMS array
  - peaks

Defaults:
- RMS window size: `0.5s`
- RMS hop size: `0.1s`
- Silence threshold: `-40 dBFS` equivalent (tune later)

### Acceptance Criteria
- [ ] `timeline.json` exists and references the same segment IDs as `transcript.json`.
- [ ] Every timeline segment has `rms_energy` and `speech_rate_wps`.
- [ ] Silence estimates do not exceed segment boundaries.
- [ ] Timeline duration matches `transcript.duration_s` within ±0.25s.

---

## US-3.1 — Generate raw candidate windows

**As a** system  
**I want to** generate many candidate clip windows  
**So that** we can score and rank them effectively

### Default Technical Design
Candidate generation is transcript-first with sentence/segment snapping.

Defaults:
- `min_len_s = 18`
- `max_len_s = 55`
- `target_len_s = 35`
- `step_s = 6` (new candidate every 6 seconds)
- Snap window boundaries to nearest transcript segment boundary within ±`1.2s`

Algorithm (V1):
1. Create sliding windows over time: `t = 0..duration` step `6s`.
2. For each `t`, propose window `[t, t+target_len]`.
3. Snap start and end to nearest segment boundary.
4. If duration < min or > max after snapping, discard.

Persist `candidates_raw.json`:
```json
{
  "candidates": [
    {
      "candidate_id": "c_0001",
      "start_s": 120.4,
      "end_s": 155.2,
      "duration_s": 34.8
    }
  ]
}
```

### Acceptance Criteria
- [ ] At least one candidate exists for videos longer than 2 minutes.
- [ ] All candidates satisfy duration constraints.
- [ ] Candidates are snapped to transcript segment boundaries.
- [ ] Candidate IDs are stable/deterministic given the same inputs.

---

## US-3.2 — Compute “hook / middle / end” structure features

**As a** system  
**I want to** compute structure features for each candidate  
**So that** we can prioritize clips with a complete narrative arc

### Default Technical Design
For each candidate window, compute:

**Hook region:** first `3.0s` of candidate  
**Body region:** middle  
**Close region:** last `4.0s` of candidate

Features (V1):
- `hook_question`: boolean (contains `?` or leading interrogatives)
- `hook_open_loop`: boolean (phrases like "here's the thing", "wait", "but", "the secret")
- `hook_contrast`: boolean (but/however/actually)
- `close_resolution`: boolean (phrases like "that's why", "the point is", "so yeah", "in summary")
- `ends_on_sentence_boundary`: boolean (last segment ends cleanly; no trailing cut mid-sentence)
- `speaker_dominance_ratio`: max fraction of time occupied by a single speaker in window
- `speaker_switch_count`
- `overlap_ratio` (fraction of window in overlap regions)
- `energy_peak_ratio`: peak RMS in window / median RMS
- `silence_at_start_s`, `silence_at_end_s`

Defaults:
- Start silence max: `<= 0.8s` allowed
- End silence max: `<= 1.0s` allowed
- Penalize if `overlap_ratio > 0.18`

### Acceptance Criteria
- [ ] Every raw candidate has a computed feature vector stored in `candidates_features.json`.
- [ ] Hook/close regions are computed even if candidate is near start/end of file.
- [ ] `speaker_dominance_ratio` is in `[0,1]`.
- [ ] `speaker_switch_count` is a non-negative integer.

---

## US-3.3 — Score candidates with deterministic heuristics (V1 primary ranker)

**As a** system  
**I want to** score candidates using a transparent heuristic model  
**So that** we get reasonable clip suggestions without requiring an LLM

### Default Technical Design

#### Hard filters (discard if any true)
- Candidate starts mid-word/sentence boundary (cannot be snapped): discard
- `silence_at_start_s > 0.8` OR `silence_at_end_s > 1.0`: discard
- `speaker_dominance_ratio < 0.70`: discard (too much back-and-forth for V1)
- `overlap_ratio > 0.25`: discard
- Contains too little speech: speech coverage < `0.75`: discard

#### Scoring (0–100)
Compute a score with weighted components:

- `hook_score` (0–30)
  - +15 if `hook_question`
  - +10 if `hook_open_loop`
  - +5 if `hook_contrast`
- `continuity_score` (0–25)
  - +25 * clamp((speaker_dominance_ratio - 0.70) / 0.30, 0, 1)
  - -5 * min(speaker_switch_count, 3)
- `energy_score` (0–20)
  - +20 * clamp((energy_peak_ratio - 1.0) / 1.0, 0, 1)
- `closure_score` (0–25)
  - +15 if `close_resolution`
  - +10 if `ends_on_sentence_boundary`
  - -10 if last 2s contains a question (often implies unresolved ending)

Store results in `candidates_scored.json`:
```json
{
  "candidates": [
    {
      "candidate_id": "c_0001",
      "start_s": 120.4,
      "end_s": 155.2,
      "score": 84.2,
      "reasons": [
        "hook_question",
        "high_speaker_dominance",
        "strong_energy_peak",
        "clean_resolution"
      ]
    }
  ]
}
```

### Acceptance Criteria
- [ ] Candidates failing hard filters are excluded from `candidates_scored.json` (or included with `eligible=false`).
- [ ] Scores are deterministic.
- [ ] Each candidate includes human-readable `reasons`.
- [ ] Scores are bounded between 0 and 100.

---

## US-3.4 — (Optional) LLM rerank for “clip worthiness” and titles

**As a** system  
**I want to** optionally rerank the top candidates with an LLM  
**So that** we improve semantic selection and generate usable titles

### When to use (V1)
- Only run on the **top 12** heuristic candidates to control cost/latency.
- This is optional; the system must work without it.

### Default Technical Design
Input to LLM:
- Candidate transcript text (from `transcript.json`)
- Start/end timestamps
- Basic features summary (speaker dominance, overlap ratio, energy peak)

Output:
- `candidates_llm.json` containing:
  - `llm_score` (0–10)
  - `title`
  - `hook_line` (first on-screen caption suggestion)
  - `notes` (why good/bad)

Recommended settings:
- `temperature = 0.2` for consistency

### LLM Prompt (copy/paste)

**System**
You are a short-form video editor assistant. Your job is to evaluate whether a transcript excerpt will perform well as a standalone TikTok/YouTube Shorts clip. Be strict: prefer clips with a strong hook, clear payoff, and clean ending. Avoid clips that require missing context.

**User**
Evaluate the following podcast clip candidate and return JSON only.

Candidate metadata:
- start_s: {{start_s}}
- end_s: {{end_s}}
- duration_s: {{duration_s}}
- speaker_dominance_ratio: {{speaker_dominance_ratio}}
- speaker_switch_count: {{speaker_switch_count}}
- overlap_ratio: {{overlap_ratio}}
- energy_peak_ratio: {{energy_peak_ratio}}

Transcript excerpt:
"""
{{transcript_text}}
"""

Requirements:
1) Determine if the clip has a complete start–middle–end arc and is understandable without external context.
2) Assign an llm_score from 0 to 10 (10 = very likely to perform well).
3) Write a short title (max 8 words) that describes the idea without misleading clickbait.
4) Write a hook_line (max 8 words) suitable as the first caption on screen.
5) Provide brief notes (max 3 bullets) why it is good/bad.
6) If the clip ends mid-thought, heavily penalize the score.

Return JSON with this exact schema:
{
  "llm_score": number,
  "title": string,
  "hook_line": string,
  "self_contained": boolean,
  "has_clear_payoff": boolean,
  "notes": [string, string, string]
}

### Acceptance Criteria
- [ ] LLM rerank runs only on the top N candidates (default N=12).
- [ ] Output is valid JSON and matches the schema exactly.
- [ ] If the LLM fails, the pipeline continues using heuristic ranking.
- [ ] LLM output is stored in `candidates_llm.json`.

---

## US-3.5 — Select final candidates for rendering

**As a** system  
**I want to** select the best candidates for rendering  
**So that** we produce a manageable set of downloadable clips

### Default Technical Design
Selection logic:
- Rank by:
  1) `llm_score` (if enabled), then heuristic `score`
  2) else heuristic `score`
- Enforce diversity:
  - Do not select two clips with > `35%` time overlap
- Output count:
  - `top_k_render = 5` default

Persist `clips_selected.json`:
```json
{
  "selected": [
    {
      "candidate_id": "c_0007",
      "start_s": 842.1,
      "end_s": 890.0,
      "final_rank": 1,
      "final_score": 92.3,
      "title": "Why most advice fails"
    }
  ]
}
```

### Acceptance Criteria
- [ ] Exactly `top_k_render` clips are selected if at least that many eligible candidates exist.
- [ ] Selected clips do not overlap each other by more than 35%.
- [ ] Each selected clip has a title:
  - from LLM if enabled
  - else fallback title derived from first sentence (max 8 words)
- [ ] Selection is deterministic given the same inputs and the same LLM outputs.

---

## US-3.6 — Clip boundary refinement (clean starts/ends)

**As a** system  
**I want to** refine clip boundaries to avoid awkward cuts  
**So that** clips feel edited, not chopped

### Default Technical Design
Refinement steps for each selected clip:
- Extend start backward by up to `1.5s` if it captures a setup phrase and does not add silence > `0.8s`.
- Extend end forward by up to `2.0s` if it captures a resolution phrase and does not exceed `max_len_s`.
- Ensure:
  - no leading silence > `0.8s`
  - no trailing silence > `1.0s`
  - end on sentence boundary when possible

Persist `clips_refined.json` with before/after:
```json
{
  "clips": [
    {
      "candidate_id": "c_0007",
      "start_s_before": 842.1,
      "end_s_before": 890.0,
      "start_s_after": 840.9,
      "end_s_after": 891.6,
      "changes": ["extended_start_setup", "extended_end_resolution"]
    }
  ]
}
```

### Acceptance Criteria
- [ ] Refined clips still satisfy min/max duration constraints.
- [ ] Clip start/end adjustments are logged.
- [ ] No refined clip has leading/trailing silence beyond thresholds.
- [ ] If refinement would violate constraints, keep original boundaries.

---

## Outputs of EPIC 3 (Artifacts)

Minimum required artifacts:
- `timeline.json`
- `candidates_raw.json`
- `candidates_features.json`
- `candidates_scored.json`
- `clips_selected.json`
- `clips_refined.json`

Optional artifacts (if LLM enabled):
- `candidates_llm.json`
