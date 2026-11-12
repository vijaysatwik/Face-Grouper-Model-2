"""Conservative fallback detection for user-supplied query photographs."""

from __future__ import annotations

from pathlib import Path


def detect_one_query_face(gray, frontal_detector, profile_detector):
    """Return one face box, or fail closed if no single candidate is clear.

    First use the project's normal detector. If it finds no candidate, retry a
    slightly more sensitive frontal cascade on the original grayscale image,
    then histogram-equalized grayscale. Any retry returning multiple regions is
    treated as ambiguous; the function never chooses a box from a multi-face
    result.
    """
    import cv2

    from prepare_dataset import detect_one_face

    face, method, frontal_n, profile_n, flipped_n = detect_one_face(
        gray, frontal_detector, profile_detector
    )
    if face is not None:
        return face, method, "single face found by the standard detector"
    if frontal_n + profile_n + flipped_n:
        return None, "ambiguous", "standard detector found multiple candidate regions"

    candidates = [gray, cv2.equalizeHist(gray)]
    for index, image in enumerate(candidates):
        regions = frontal_detector.detectMultiScale(
            image,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(30, 30),
        )
        count = len(regions)
        if count == 1:
            method = "sensitive_frontal" if index == 0 else "contrast_adjusted_frontal"
            return regions[0], method, "one face found by the more sensitive fallback"
        if count > 1:
            return None, "ambiguous", f"fallback detector found {count} candidate regions"

    # OpenCV ships this second frontal cascade with cv2; it is a local fallback,
    # not a downloaded model. Use it only if the first cascade found nothing.
    default_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    default_detector = cv2.CascadeClassifier(str(default_path))
    if not default_detector.empty():
        for index, image in enumerate(candidates):
            regions = default_detector.detectMultiScale(
                image,
                scaleFactor=1.05,
                minNeighbors=5,
                minSize=(30, 30),
            )
            count = len(regions)
            if count == 1:
                method = "default_frontal" if index == 0 else "contrast_adjusted_default_frontal"
                return regions[0], method, "one face found by OpenCV's bundled default cascade"
            if count > 1:
                return None, "ambiguous", f"bundled default cascade found {count} candidate regions"
    return None, "no_face", "no face found by standard or fallback detectors"
