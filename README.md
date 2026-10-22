# Multi-Page Document Scanner and Image Enhancement

Digital Image Processing project using Python, OpenCV, NumPy, Pillow and Streamlit.

## Folder structure

```
document-scanner/
├── app.py            # Streamlit UI, session state, page management, downloads
├── scanner.py        # preprocessing, Canny, contours, corners, perspective transform
├── enhancement.py    # shadow removal, contrast, sharpening, thresholding, modes
├── pdf_generator.py  # multi-page PDF and ZIP export
├── requirements.txt
└── README.md
```

## Setup and run (VS Code terminal)

```
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

## How to use

1. Upload one or more JPG/PNG photos of document pages (more can be added later).
2. Check the original, detected corners and enhanced result for each page.
3. Click **Add to Document** (or **Add all processed pages**).
4. Reorder with the up/down buttons, delete unwanted pages.
5. Download `scanned_document.pdf` or a ZIP of the individual images.
6. If a page fails: **Retry**, **Use original page**, or **Remove page**.

## Pipeline

Input → Resize → Grayscale → Gaussian blur → Canny edges → Contours (sorted by area)
→ 4-point approximation → Corner ordering (TL, TR, BR, BL) → Perspective transform
→ Shadow removal → Contrast (CLAHE) → Noise removal → Sharpening
→ Thresholding (B&W mode) → Add to document → PDF

## Viva notes

- **Why blur before Canny?** Removes noise so fewer false edges appear.
- **Why dilate the edges?** Closes small gaps so the page outline forms one contour.
- **Why approxPolyDP?** A page is a quadrilateral; a contour with 4 vertices is the page.
- **Corner ordering:** TL has the smallest x+y, BR the largest x+y; TR has the smallest y-x, BL the largest.
- **Perspective transform:** a 3x3 homography maps the 4 photo corners to a flat rectangle.
- **Shadow removal:** dilation removes text, median blur estimates the lighting, dividing by it flattens the page.
- **CLAHE:** contrast enhancement in small tiles, so local contrast improves without over-brightening.
- **Adaptive vs Otsu:** adaptive uses a local threshold (good for uneven light); Otsu uses one global threshold.

## Future enhancements

OCR / searchable PDF, AI-based detection, automatic orientation, mobile app, cloud storage.
Each can be added as a new function or module without changing the existing ones.
