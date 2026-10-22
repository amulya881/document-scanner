"""
pdf_generator.py
----------------
Turns the ordered list of processed pages into a multi-page PDF (using Pillow)
or a ZIP of individual images.  Pages are RGB NumPy arrays.
"""

import io
import zipfile

import numpy as np
from PIL import Image


def generate_pdf(pages, resolution=150):
    """Return the bytes of a PDF whose page order matches the `pages` list."""
    if not pages:
        raise ValueError("No pages to export.")
    images = [Image.fromarray(p).convert("RGB") for p in pages]
    buffer = io.BytesIO()
    images[0].save(buffer, format="PDF", save_all=True,
                   append_images=images[1:], resolution=resolution)
    return buffer.getvalue()


def images_to_zip(pages, fmt="PNG"):
    """Return the bytes of a ZIP containing page_1.png, page_2.png, ... in order."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, page in enumerate(pages, start=1):
            img_bytes = io.BytesIO()
            Image.fromarray(page).save(img_bytes, format=fmt)
            zf.writestr(f"page_{i}.{fmt.lower()}", img_bytes.getvalue())
    return buffer.getvalue()


def pdf_to_images(pdf_bytes, dpi=200):
    """Render every page of an uploaded PDF to an RGB NumPy array (uses PyMuPDF).

    PyMuPDF needs no extra programs (no Poppler), so it works on Windows out of the box.
    """
    import fitz  # PyMuPDF

    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            images.append(arr.copy())
    return images
