"""
enhancement.py
--------------
Look-and-feel part of the pipeline (applied after perspective correction):

    Shadow / illumination correction -> Noise removal -> Contrast (CLAHE)
    -> Sharpening -> Thresholding (Black & White mode)
"""

import cv2
import numpy as np

MODE_COLOR = "Enhanced Color"
MODE_BW = "Black & White"


# ---------------------------------------------------------------- Shadow / background removal
def remove_shadows(channel):
    """Background normalisation on a single channel.

    1. Dilation removes the thin dark text, leaving paper + shadow.
    2. Median blur smooths that into an estimate of the background illumination.
    3. Dividing the image by the background flattens uneven lighting and shadows.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(channel, kernel)
    background = cv2.medianBlur(dilated, 21)
    return cv2.divide(channel, background, scale=255)


# ---------------------------------------------------------------- Sharpening
def sharpen(image):
    """Unsharp masking: image + (image - blurred)."""
    blurred = cv2.GaussianBlur(image, (0, 0), 2)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


# ---------------------------------------------------------------- Mode 1: Enhanced Color
def enhance_color(bgr):
    """Natural-looking page: even lighting, better contrast, slight sharpening."""
    # shadow removal on each colour channel
    flat = cv2.merge([remove_shadows(c) for c in cv2.split(bgr)])

    # contrast enhancement: CLAHE on the lightness channel only (keeps colours natural)
    l, a, b = cv2.split(cv2.cvtColor(flat, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    out = cv2.bilateralFilter(out, 5, 40, 40)          # noise removal, keeps edges
    out = sharpen(out)                                  # text visibility
    out = cv2.convertScaleAbs(out, alpha=1.1, beta=-5)  # brightness / contrast tweak
    return out


# ---------------------------------------------------------------- Mode 2: Black & White
def enhance_bw(bgr, method="Adaptive"):
    """Scanner-like black & white page using adaptive or Otsu thresholding."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = remove_shadows(gray)                 # illumination correction
    gray = cv2.medianBlur(gray, 3)              # salt-and-pepper noise removal

    if method == "Otsu":
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 15)

    # morphological closing removes tiny black specks on the white background
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------- Entry point used by app.py
def enhance_page(bgr, mode=MODE_COLOR, bw_method="Adaptive"):
    """Return the enhanced page (BGR) for the chosen mode."""
    if mode == MODE_BW:
        return enhance_bw(bgr, bw_method)
    return enhance_color(bgr)
