from typing import Literal, Union

import numpy as np

from video_gaze_shifts.utils import round_dict

CENTER_AREA = Literal["center"]
LEFT_TOP_AREA = Literal["left-top"]
LEFT_CENTER_AREA = Literal["left-center"]
LEFT_BOTTOM_AREA = Literal["left-bottom"]
RIGHT_TOP_AREA = Literal["right-top"]
RIGHT_CENTER_AREA = Literal["right-center"]
RIGHT_BOTTOM_AREA = Literal["right-bottom"]
AREA_TAG = Union[CENTER_AREA, LEFT_TOP_AREA, LEFT_CENTER_AREA, LEFT_BOTTOM_AREA, RIGHT_TOP_AREA, RIGHT_CENTER_AREA, RIGHT_BOTTOM_AREA]  # noqa: BLK100
AREA_TAGS = ("center", "left-top", "left-center", "left-bottom", "right-top", "right-center", "right-bottom")
SHIFTS = ("center-->left-top", "center-->left-center", "center-->left-bottom", "center-->right-top", "center-->right-center", "center-->right-bottom", "left-top-->center", "left-top-->left-center", "left-top-->left-bottom", "left-top-->right-top", "left-top-->right-center", "left-top-->right-bottom", "left-center-->center", "left-center-->left-top", "left-center-->left-bottom", "left-center-->right-top", "left-center-->right-center", "left-center-->right-bottom", "left-bottom-->center", "left-bottom-->left-top", "left-bottom-->left-center", "left-bottom-->right-top", "left-bottom-->right-center", "left-bottom-->right-bottom", "right-top-->center", "right-top-->left-top", "right-top-->left-center", "right-top-->left-bottom", "right-top-->right-center", "right-top-->right-bottom", "right-center-->center", "right-center-->left-top", "right-center-->left-center", "right-center-->left-bottom", "right-center-->right-top", "right-center-->right-bottom", "right-bottom-->center", "right-bottom-->left-top", "right-bottom-->left-center", "right-bottom-->left-bottom", "right-bottom-->right-top", "right-bottom-->right-center")


def estimate_shift(pitch: float, yaw: float, mode: str = "circle", **kwargs) -> AREA_TAG:
    if mode == "circle":
        return estimate_circle_shift(pitch=pitch, yaw=yaw, **kwargs)
    elif mode == "octagon":
        return estimate_ortagon_shift(pitch=pitch, yaw=yaw, **kwargs)
    elif mode == "hexagon":
        return estimate_hexagon_shift(pitch=pitch, yaw=yaw, **kwargs)

    raise NotImplementedError


def tags_to_stats(tags: list[AREA_TAGS], n_digits: int = 4, fps: int = 30) -> dict:
    per_tag_frames, per_tag_norm_frames, per_tag_time_sec, per_tag_mean_time_sec, smoothed_tags = tags_to_per_tag_times(tags=tags, fps=fps)
    _, per_shift_times, per_shift_norm_times, mean_frames_wo_shift_per_tag, mean_secs_wo_shift_per_tag = tags_to_shifts(tags=smoothed_tags)
    num_shifts = sum(per_shift_times.values())
    unused_tags = [tag for tag in per_tag_frames if per_tag_frames[tag] == 0.0]

    return {
        "per_tag_frames": round_dict(dictionary=per_tag_frames, n_digits=n_digits),
        "per_tag_norm_frames": round_dict(dictionary=per_tag_norm_frames),
        "per_tag_time_sec": round_dict(dictionary=per_tag_time_sec),
        "per_tag_mean_time_sec": round_dict(dictionary=per_tag_mean_time_sec),
        "per_shift_times": round_dict(dictionary=per_shift_times),
        "per_shift_norm_times": round_dict(dictionary=per_shift_norm_times),
        "total_num_shifts": num_shifts,
        "unused_tags": unused_tags,
        "freq_frames_shift_from_center": 1.0 / mean_frames_wo_shift_per_tag["center"],
        "mean_frames_wo_shift_per_tag": round_dict(mean_frames_wo_shift_per_tag),
        "freq_secs_shift_from_center": 1.0 / mean_secs_wo_shift_per_tag["center"],
        "mean_secs_wo_shift_per_tag": round_dict(mean_secs_wo_shift_per_tag),
    }


def tags_to_per_tag_times(tags: list[AREA_TAG], fps: int = 30, smooth_window: float = 10):
    per_tag_frames = {tag: 0.0 for tag in AREA_TAGS}
    for tag in tags:
        per_tag_frames[tag] += 1.0
    per_tag_norm_frames = {tag: per_tag_frames[tag] / len(tags) for tag in AREA_TAGS}

    smoothed_tags = []
    for beg in range(0, len(tags), smooth_window):
        step_tag_freq = {tag: 0 for tag in AREA_TAGS}
        for tag in tags[beg : beg + smooth_window]:
            step_tag_freq[tag] += 1
        smoothed_tag = list(step_tag_freq.keys())[np.argmax(list(step_tag_freq.values()))]
        smoothed_tags.append(smoothed_tag)
    prev_tag = None
    per_tag_time_sec = {tag: [] for tag in AREA_TAGS}
    time_counter = 0.0
    for tag in smoothed_tags:
        if prev_tag is None:
            prev_tag = tag

        if prev_tag is not None and tag != prev_tag:
            per_tag_time_sec[tag].append(time_counter)
            time_counter = 0.0
            prev_tag = tag
        else:
            time_counter += 1.0

    per_tag_mean_time_sec = {
        tag: (np.mean(per_tag_time_sec[tag]) * (smooth_window / fps)) if per_tag_time_sec[tag] else 0.0
        for tag in AREA_TAGS
    }
    per_tag_time_sec = {tag: sum(per_tag_time_sec[tag]) * (smooth_window / fps) for tag in AREA_TAGS}

    return per_tag_frames, per_tag_norm_frames, per_tag_time_sec, per_tag_mean_time_sec, smoothed_tags


def tags_to_shifts(tags: list[AREA_TAG], fps: int = 30, smooth_window: float = 10):
    shifts = []
    mean_frames_wo_shift_per_tag = {tag: [] for tag in AREA_TAGS}
    frames_wo_shift_per_tag_counter = {tag: 0.0 for tag in AREA_TAGS}
    for prev_tag, tag in zip(tags[:-1], tags[1:]):
        frames_wo_shift_per_tag_counter[prev_tag] += 1.0

        if prev_tag != tag:
            shifts.append(f"{prev_tag}-->{tag}")

            mean_frames_wo_shift_per_tag[prev_tag].append(frames_wo_shift_per_tag_counter[prev_tag])
            frames_wo_shift_per_tag_counter[prev_tag] = 0.0
    mean_frames_wo_shift_per_tag = {
        tag: np.mean(mean_frames_wo_shift_per_tag[tag]) * smooth_window if mean_frames_wo_shift_per_tag[tag] else 0.0
        for tag in mean_frames_wo_shift_per_tag
    }
    mean_secs_wo_shift_per_tag = {
        tag: np.mean(mean_frames_wo_shift_per_tag[tag]) * smooth_window / fps if mean_frames_wo_shift_per_tag[tag] else 0.0
        for tag in mean_frames_wo_shift_per_tag
    }

    per_shift_times = {shift: 0.0 for shift in SHIFTS}
    for shift in shifts:
        per_shift_times[shift] += 1.0

    if shifts:
        per_shift_norm_times = {shift: per_shift_times[shift] / len(shifts) for shift in per_shift_times}
    else:
        per_shift_norm_times = per_shift_times

    return shifts, per_shift_times, per_shift_norm_times, mean_frames_wo_shift_per_tag, mean_secs_wo_shift_per_tag


def estimate_circle_shift(pitch: float, yaw: float, r: float, **kwargs) -> AREA_TAG:
    if abs(pitch) <= r and abs(yaw) <= r or yaw == 0.0:
        return "center"

    half_r = r / 2.0
    left_right_tag = "left"
    top_bottom_tag = "center"
    if yaw > 0.0:
        left_right_tag = "right"
    if pitch > half_r:
        top_bottom_tag = "top"
    elif pitch < -half_r:
        top_bottom_tag = "bottom"

    return f"{left_right_tag}-{top_bottom_tag}"


def estimate_ortagon_shift(pitch: float, yaw: float, r: float, **kwargs) -> bool:
    raise NotImplementedError


def estimate_hexagon_shift(pitch: float, yaw: float, r: float, **kwargs) -> bool:
    raise NotImplementedError
