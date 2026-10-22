"""
scanner.py
----------
Geometry part of the pipeline:

    Input -> Preprocessing -> Canny edges -> Contours -> Document detection
          -> Four corners -> Perspective correction

Every function works on ONE page at a time, so a failure on one page never
affects the others.
"""

import cv2
import numpy as np

MAX_WORK_DIM = 1000  # detection is done on a smaller copy for speed


# ---------------------------------------------------------------- 1. Preprocessing
def preprocess(image):
    """Resize (for speed), convert to grayscale and remove noise with Gaussian blur.

    Returns: small_bgr, blurred_gray, ratio   (small = original * ratio)
    """
    h, w = image.shape[:2]
    ratio = min(1.0, MAX_WORK_DIM / max(h, w))
    small = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA) if ratio < 1 else image.copy()

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)   # grayscale conversion
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)      # noise reduction
    return small, blurred, ratio


# ---------------------------------------------------------------- 2. Edge detection
def detect_edges(blurred, relaxed=False):
    """Canny edge detection followed by dilation/closing so the page outline has no gaps.

    relaxed=True derives the Canny thresholds from the image median (used on Retry).
    """
    if relaxed:
        med = float(np.median(blurred))
        low, high = int(max(0, 0.5 * med)), int(min(255, 1.2 * med))
    else:
        low, high = 75, 200

    edges = cv2.Canny(blurred, low, high)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2 if relaxed else 1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return edges


# ---------------------------------------------------------------- 3. Document detection
def find_document_contour(edges, relaxed=False):
    """Find the largest contour that approximates to a 4-sided shape (the page)."""
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # sort by area

    image_area = edges.shape[0] * edges.shape[1]
    min_area = image_area * (0.05 if relaxed else 0.15)   # ignore small objects
    epsilons = [0.02, 0.03, 0.05] if relaxed else [0.02]

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            break
        shapes = [contour, cv2.convexHull(contour)] if relaxed else [contour]
        for shape in shapes:
            peri = cv2.arcLength(shape, True)
            for eps in epsilons:
                approx = cv2.approxPolyDP(shape, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    return approx.reshape(4, 2).astype("float32")
    return None


# ---------------------------------------------------------------- 4. Corner ordering
def order_points(pts):
    """Order corners as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()      # y - x
    rect[0] = pts[np.argmin(s)]           # top-left     (smallest x+y)
    rect[2] = pts[np.argmax(s)]           # bottom-right (largest x+y)
    rect[1] = pts[np.argmin(d)]           # top-right    (smallest y-x)
    rect[3] = pts[np.argmax(d)]           # bottom-left  (largest y-x)
    return rect


# ---------------------------------------------------------------- 5. Perspective correction
def four_point_transform(image, pts):
    """Warp the quadrilateral described by `pts` into a flat rectangle."""
    tl, tr, br, bl = order_points(pts)

    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 50 or height < 50:
        return None

    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), dst)
    return cv2.warpPerspective(image, matrix, (width, height))


# ---------------------------------------------------------------- Full pipeline for one page
def scan_page(image, relaxed=False):
    """Run the geometry pipeline on one BGR image.

    Returns a dict:
        original   - input image (BGR)
        edges      - Canny edge image
        annotated  - image with detected outline and corners drawn
        corners    - ordered corners in ORIGINAL image coordinates (or None)
        warped     - perspective-corrected page (BGR) or None if detection failed
        error      - None or an error string
    """
    result = {"original": image, "edges": None, "annotated": None,
              "corners": None, "warped": None, "error": None}
    try:
        small, blurred, ratio = preprocess(image)
        edges = detect_edges(blurred, relaxed)
        result["edges"] = edges
        result["annotated"] = small.copy()

        pts = find_document_contour(edges, relaxed)
        if pts is None:
            result["error"] = "Document not detected"
            return result

        ordered_small = order_points(pts)
        # draw outline and numbered corners on the preview image
        annotated = small.copy()
        cv2.polylines(annotated, [ordered_small.astype(int)], True, (0, 255, 0), 3)
        for i, (x, y) in enumerate(ordered_small.astype(int)):
            cv2.circle(annotated, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(annotated, ["TL", "TR", "BR", "BL"][i], (x + 12, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        result["annotated"] = annotated

        corners = ordered_small / ratio           # back to full-resolution coordinates
        result["corners"] = corners
        warped = four_point_transform(image, corners)
        if warped is None:
            result["error"] = "Document not detected"
            return result
        result["warped"] = warped
    except Exception as exc:                      # never crash the app because of one page
        result["error"] = f"Processing error: {exc}"
    return result


# ---------------------------------------------------------------- Already-flat pages (PDF pages, flat scans)
def scan_flat(image):
    """For pages that are already flat (PDF pages, flatbed scans, screenshots).

    There is no background to remove, so the whole image is used as the page and
    only the enhancement steps are applied afterwards.
    """
    result = {"original": image, "edges": None, "annotated": None,
              "corners": None, "warped": image, "error": None, "flat": True}
    try:
        _, blurred, _ = preprocess(image)
        result["edges"] = detect_edges(blurred)
    except Exception:
        result["edges"] = np.zeros(image.shape[:2], np.uint8)
    result["annotated"] = image.copy()
    return result
