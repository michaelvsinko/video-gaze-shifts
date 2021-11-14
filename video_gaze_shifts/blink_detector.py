from typing import Any, Optional

import numpy as np

from video_gaze_shifts.blink_extraction_tf.blink_extraction import load_vgg16_model


def get_blink_detector(name: str = "tf__vgg16", weights_path: Optional[str] = None):
    if name == "tf__vgg16":
        model = load_vgg16_model(weights_path=weights_path)
    else:
        raise NotImplementedError

    return model


def is_blink(
    eyes: tuple[np.ndarray, np.ndarray],
    model: Any,
    transform,
    mode: str = "mean",
    thr: float = 0.15,
) -> bool:
    # TODO: eye_detect_vgg16.h5 uses not normalized image

    blink_probs = []
    for eye in eyes:
        blink_prob = model.predict(transform(eye))
        blink_probs.append(float(blink_prob))

    if mode == "mean":
        result_prob = np.mean(blink_probs)
    else:
        print(mode)
        raise NotImplementedError

    if result_prob > thr:
        return False

    return True
