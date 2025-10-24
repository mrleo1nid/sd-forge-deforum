"""Pure functions for camera movement analysis and description generation.

This module contains pure functions extracted from
scripts/deforum_helpers/wan/utils/movement_analyzer.py, following functional
programming principles with no side effects.
"""

from typing import List, Dict, Tuple


# Movement intensity thresholds (can be scaled by sensitivity)
DEFAULT_TRANSLATION_THRESHOLD = 0.001
DEFAULT_ROTATION_THRESHOLD = 0.001
DEFAULT_ZOOM_THRESHOLD = 0.0001


def analyze_frame_ranges(
    values: List[float],
    movement_type: str,
    sensitivity: float = 1.0,
) -> List[Dict]:
    """Analyze movement values frame by frame and identify movement ranges.

    Detects segments of continuous movement in a single direction by analyzing
    frame-to-frame changes. Uses sensitivity-adjusted thresholds to identify
    significant movement.

    Args:
        values: List of movement values, one per frame
        movement_type: Type of movement ("translation_x", "rotation_y", "zoom", etc.)
        sensitivity: Multiplier for detection thresholds (default: 1.0)

    Returns:
        List of movement segment dicts with keys:
        - start_frame: First frame of movement
        - end_frame: Last frame of movement
        - direction: "increasing" or "decreasing"
        - movement_type: Type of movement
        - max_change: Maximum frame-to-frame change
        - total_range: Total range of movement in segment

    Examples:
        >>> values = [0.0, 0.1, 0.2, 0.3, 0.2, 0.1]
        >>> segments = analyze_frame_ranges(values, "translation_x")
        >>> len(segments)
        2
        >>> segments[0]['direction']
        'increasing'
    """
    if not values or len(values) < 2:
        return []

    # Determine threshold based on movement type
    if movement_type == "zoom":
        threshold = DEFAULT_ZOOM_THRESHOLD * sensitivity
    elif movement_type in ["rotation_x", "rotation_y", "rotation_z"]:
        threshold = DEFAULT_ROTATION_THRESHOLD * sensitivity
    else:  # translation
        threshold = DEFAULT_TRANSLATION_THRESHOLD * sensitivity

    # Calculate frame-by-frame changes
    changes = []
    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        changes.append(change)

    # Find movement segments
    segments = []
    current_segment = None

    for frame, change in enumerate(changes):
        abs_change = abs(change)

        # Check if this frame has significant movement
        if abs_change > threshold:
            direction = "increasing" if change > 0 else "decreasing"

            # Start new segment or continue existing one
            if current_segment is None or current_segment["direction"] != direction:
                # End previous segment
                if current_segment is not None:
                    current_segment["end_frame"] = frame
                    segments.append(current_segment)

                # Start new segment
                current_segment = {
                    "start_frame": frame,
                    "end_frame": frame,
                    "direction": direction,
                    "movement_type": movement_type,
                    "max_change": abs_change,
                    "total_range": abs(values[frame + 1] - values[frame]),
                }
            else:
                # Continue current segment
                current_segment["end_frame"] = frame
                current_segment["max_change"] = max(
                    current_segment["max_change"], abs_change
                )
                current_segment["total_range"] = abs(
                    values[frame + 1] - values[current_segment["start_frame"]]
                )

        elif current_segment is not None:
            # End current segment if movement stopped
            current_segment["end_frame"] = frame
            segments.append(current_segment)
            current_segment = None

    # Close final segment if needed
    if current_segment is not None:
        current_segment["end_frame"] = len(changes)
        segments.append(current_segment)

    return segments


def generate_segment_description(segment: Dict, total_frames: int) -> str:
    """Generate specific description for a movement segment.

    Creates a human-readable description of a camera movement segment,
    including intensity, direction, and frame range information.

    Args:
        segment: Movement segment dict with keys: start_frame, end_frame,
                movement_type, direction, total_range
        total_frames: Total number of frames in animation

    Returns:
        Descriptive string like "subtle panning left (frames 10-25)"

    Examples:
        >>> segment = {
        ...     'start_frame': 10,
        ...     'end_frame': 25,
        ...     'movement_type': 'translation_x',
        ...     'direction': 'decreasing',
        ...     'total_range': 0.5
        ... }
        >>> desc = generate_segment_description(segment, 100)
        >>> 'panning left' in desc
        True
    """
    start = segment["start_frame"]
    end = segment["end_frame"]
    movement_type = segment["movement_type"]
    direction = segment["direction"]
    total_range = segment["total_range"]

    # Frame range description
    if end - start < 10:
        frame_desc = f"frames {start}-{end}"
    elif end - start < total_frames * 0.3:
        frame_desc = f"frames {start}-{end} (brief)"
    elif end - start < total_frames * 0.7:
        frame_desc = f"frames {start}-{end} (moderate)"
    else:
        frame_desc = f"frames {start}-{end} (extended)"

    # Movement intensity
    if total_range < 1.0:
        intensity = "subtle"
    elif total_range < 5.0:
        intensity = "gentle"
    elif total_range < 20.0:
        intensity = "moderate"
    else:
        intensity = "strong"

    # Movement type specific descriptions
    if movement_type == "translation_x":
        motion = "panning right" if direction == "increasing" else "panning left"
    elif movement_type == "translation_y":
        motion = "moving up" if direction == "increasing" else "moving down"
    elif movement_type == "translation_z":
        motion = "dolly forward" if direction == "increasing" else "dolly backward"
    elif movement_type == "rotation_x":
        motion = "tilting up" if direction == "increasing" else "tilting down"
    elif movement_type == "rotation_y":
        motion = "rotating right" if direction == "increasing" else "rotating left"
    elif movement_type == "rotation_z":
        motion = (
            "rolling clockwise"
            if direction == "increasing"
            else "rolling counter-clockwise"
        )
    elif movement_type == "zoom":
        motion = "zooming in" if direction == "increasing" else "zooming out"
    else:
        motion = f"{movement_type} {direction}"

    return f"{intensity} {motion} ({frame_desc})"


def group_similar_segments(
    segments: List[Dict], max_frames: int, proximity_threshold: float = 0.1
) -> List[List[Dict]]:
    """Group similar movement segments that happen close together.

    Combines movement segments of the same type and direction that occur
    within a proximity threshold of each other.

    Args:
        segments: List of movement segment dicts
        max_frames: Total number of frames
        proximity_threshold: Max gap between segments as fraction of total frames
                           (default: 0.1 = 10% of total frames)

    Returns:
        List of groups, where each group is a list of similar segments

    Examples:
        >>> seg1 = {'start_frame': 0, 'end_frame': 10, 'movement_type': 'translation_x', 'direction': 'increasing'}
        >>> seg2 = {'start_frame': 15, 'end_frame': 25, 'movement_type': 'translation_x', 'direction': 'increasing'}
        >>> seg3 = {'start_frame': 50, 'end_frame': 60, 'movement_type': 'translation_y', 'direction': 'increasing'}
        >>> groups = group_similar_segments([seg1, seg2, seg3], 100)
        >>> len(groups)
        2
    """
    if not segments:
        return []

    groups = []
    current_group = [segments[0]]

    for i in range(1, len(segments)):
        current_seg = segments[i]
        prev_seg = segments[i - 1]

        # Check if segments should be grouped together
        same_movement_type = current_seg["movement_type"] == prev_seg["movement_type"]
        same_direction = current_seg["direction"] == prev_seg["direction"]
        close_frames = (
            current_seg["start_frame"] - prev_seg["end_frame"]
            <= max_frames * proximity_threshold
        )

        if same_movement_type and same_direction and close_frames:
            # Add to current group
            current_group.append(current_seg)
        else:
            # Start new group
            groups.append(current_group)
            current_group = [current_seg]

    # Add the last group
    groups.append(current_group)

    return groups


def generate_group_description(
    group: List[Dict], max_frames: int
) -> Tuple[str, float]:
    """Generate description for a group of similar movement segments.

    Combines information from multiple similar segments into a single
    description with intensity and duration information.

    Args:
        group: List of similar movement segment dicts
        max_frames: Total number of frames in animation

    Returns:
        Tuple of (description_string, strength_value)
        where strength is a measure of movement intensity (0.0 - unlimited)

    Examples:
        >>> segments = [
        ...     {'start_frame': 0, 'end_frame': 10, 'movement_type': 'translation_x',
        ...      'direction': 'increasing', 'total_range': 2.0},
        ...     {'start_frame': 15, 'end_frame': 25, 'movement_type': 'translation_x',
        ...      'direction': 'increasing', 'total_range': 3.0}
        ... ]
        >>> desc, strength = generate_group_description(segments, 100)
        >>> 'panning right' in desc
        True
        >>> strength > 0.0
        True
    """
    if not group:
        return "", 0.0

    # Use the first segment to determine movement type and direction
    main_segment = group[0]
    movement_type = main_segment["movement_type"]
    direction = main_segment["direction"]

    # Calculate total range and duration for the entire group
    start_frame = min(seg["start_frame"] for seg in group)
    end_frame = max(seg["end_frame"] for seg in group)
    total_duration = end_frame - start_frame + 1
    total_range = sum(seg["total_range"] for seg in group)

    # Determine movement intensity
    if total_range < 1.0:
        intensity = "subtle"
    elif total_range < 10.0:
        intensity = "gentle"
    elif total_range < 50.0:
        intensity = "moderate"
    else:
        intensity = "strong"

    # Generate specific directional description
    if movement_type == "translation_x":
        motion = "panning right" if direction == "increasing" else "panning left"
    elif movement_type == "translation_y":
        motion = "moving up" if direction == "increasing" else "moving down"
    elif movement_type == "translation_z":
        motion = "dolly forward" if direction == "increasing" else "dolly backward"
    else:
        motion = f"{movement_type} {direction}"

    # Frame range description
    if total_duration < max_frames * 0.2:
        duration_desc = "brief"
    elif total_duration < max_frames * 0.5:
        duration_desc = "extended"
    else:
        duration_desc = "sustained"

    # Combine into final description
    description = f"{intensity} {motion} ({duration_desc})"

    # Calculate strength
    strength = (total_range * total_duration) / (max_frames * 10.0)

    return description, strength


def generate_rotation_group_description(
    group: List[Dict], max_frames: int
) -> Tuple[str, float]:
    """Generate description for a group of similar rotation segments.

    Similar to generate_group_description but specialized for rotation movements
    (tilt, rotate, roll).

    Args:
        group: List of similar rotation segment dicts
        max_frames: Total number of frames in animation

    Returns:
        Tuple of (description_string, strength_value)

    Examples:
        >>> segments = [
        ...     {'start_frame': 0, 'end_frame': 20, 'movement_type': 'rotation_x',
        ...      'direction': 'increasing', 'total_range': 5.0}
        ... ]
        >>> desc, strength = generate_rotation_group_description(segments, 100)
        >>> 'tilting up' in desc
        True
    """
    if not group:
        return "", 0.0

    # Use the first segment to determine movement type and direction
    main_segment = group[0]
    movement_type = main_segment["movement_type"]
    direction = main_segment["direction"]

    # Calculate total range and duration for the entire group
    start_frame = min(seg["start_frame"] for seg in group)
    end_frame = max(seg["end_frame"] for seg in group)
    total_duration = end_frame - start_frame + 1
    total_range = sum(seg["total_range"] for seg in group)

    # Determine movement intensity
    if total_range < 1.0:
        intensity = "subtle"
    elif total_range < 10.0:
        intensity = "gentle"
    elif total_range < 50.0:
        intensity = "moderate"
    else:
        intensity = "strong"

    # Generate specific directional description
    if movement_type == "rotation_x":
        motion = "tilting up" if direction == "increasing" else "tilting down"
    elif movement_type == "rotation_y":
        motion = "rotating right" if direction == "increasing" else "rotating left"
    elif movement_type == "rotation_z":
        motion = (
            "rolling clockwise"
            if direction == "increasing"
            else "rolling counter-clockwise"
        )
    else:
        motion = f"{movement_type} {direction}"

    # Frame range description
    if total_duration < max_frames * 0.2:
        duration_desc = "brief"
    elif total_duration < max_frames * 0.5:
        duration_desc = "extended"
    else:
        duration_desc = "sustained"

    # Combine into final description
    description = f"{intensity} {motion} ({duration_desc})"

    # Calculate strength
    strength = (total_range * total_duration) / (max_frames * 10.0)

    return description, strength
