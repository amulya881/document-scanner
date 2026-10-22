"""
app.py  -  Streamlit UI for the Multi-Page Document Scanner.

Run with:   streamlit run app.py
"""

import io
import uuid

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from enhancement import MODE_BW, MODE_COLOR, enhance_page
from pdf_generator import generate_pdf, images_to_zip, pdf_to_images
from scanner import scan_flat, scan_page

MAX_INPUT_DIM = 2500  # limit very large phone photos so processing stays fast

st.set_page_config(page_title="Document Scanner", layout="wide")


# ================================================================ session state
def init_state():
    defaults = {
        "pages": [],            # final document: list of {"id", "name", "image"(RGB)}
        "expanded": {},         # uploaded file key -> [(page key, label)] (a PDF gives many pages)
        "scans": {},            # page key -> scan result
        "previews": {},         # (file key, mode, method, attempt) -> enhanced RGB
        "added": set(),         # file keys already added to the document
        "dismissed": set(),     # file keys removed by the user
        "uploader_n": 0,        # bump to reset the file uploader
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ================================================================ helpers
def limit_size(bgr):
    scale = min(1.0, MAX_INPUT_DIM / max(bgr.shape[:2]))
    if scale < 1:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return bgr


def load_bgr(data):
    """Bytes -> BGR image (fixes phone rotation from EXIF, limits size)."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    return limit_size(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def error_result(message):
    return {"original": None, "edges": None, "annotated": None, "corners": None,
            "warped": None, "error": message, "attempt": 0}


def expand_file(key, f):
    """Scan an uploaded file once.

    - Image (JPG/PNG): one page, document detection + perspective correction
      (or the whole image if 'already flat' is ticked).
    - PDF: every PDF page becomes its own page; PDF pages are already flat,
      so only the enhancement steps are applied.
    Returns [(page_key, label), ...]
    """
    if key in st.session_state.expanded:
        return st.session_state.expanded[key]

    data, items = f.getvalue(), []
    if f.name.lower().endswith(".pdf"):
        try:
            for p, rgb in enumerate(pdf_to_images(data), start=1):
                pk = f"{key}_p{p}"
                res = scan_flat(limit_size(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)))
                res["attempt"] = 0
                st.session_state.scans[pk] = res
                items.append((pk, f"{f.name} - page {p}"))
        except Exception:
            st.session_state.scans[key] = error_result("The PDF could not be read")
            items = [(key, f.name)]
    else:
        try:
            bgr = load_bgr(data)
            res = scan_flat(bgr) if st.session_state.flat_images else scan_page(bgr)
            res["attempt"] = 0
        except Exception:
            res = error_result("The image could not be read")
        st.session_state.scans[key] = res
        items = [(key, f.name)]

    st.session_state.expanded[key] = items
    return items


def get_enhanced(key, res):
    """Enhanced RGB page for the current mode (cached so reruns stay fast)."""
    ck = (key, st.session_state.mode, st.session_state.bw_method, res["attempt"])
    if ck not in st.session_state.previews:
        out = enhance_page(res["warped"], st.session_state.mode, st.session_state.bw_method)
        st.session_state.previews[ck] = to_rgb(out)
    return st.session_state.previews[ck]


# ================================================================ callbacks
def add_to_document(key, name, use_original=False):
    res = st.session_state.scans[key]
    image = to_rgb(res["original"]) if use_original else get_enhanced(key, res)
    st.session_state.pages.append({"id": uuid.uuid4().hex, "name": name, "image": image})
    st.session_state.added.add(key)


def add_all(keys):
    for key, name in keys:
        add_to_document(key, name)


def retry_page(key):
    old = st.session_state.scans[key]
    if old["original"] is None:
        return
    res = scan_page(old["original"], relaxed=True)   # looser edge / contour settings
    res["attempt"] = old["attempt"] + 1
    st.session_state.scans[key] = res


def use_full_image(key):
    """Treat the whole image as the page (for images that are already flat scans)."""
    old = st.session_state.scans[key]
    if old["original"] is None:
        return
    res = scan_flat(old["original"])
    res["attempt"] = old["attempt"] + 1
    st.session_state.scans[key] = res


def remove_upload(key):
    st.session_state.dismissed.add(key)


def move_page(index, step):
    pages = st.session_state.pages
    j = index + step
    if 0 <= j < len(pages):
        pages[index], pages[j] = pages[j], pages[index]


def delete_page(index):
    st.session_state.pages.pop(index)


def clear_all():
    st.session_state.pages = []
    st.session_state.expanded = {}
    st.session_state.scans = {}
    st.session_state.previews = {}
    st.session_state.added = set()
    st.session_state.dismissed = set()
    st.session_state.uploader_n += 1   # gives a fresh, empty uploader


# ================================================================ sidebar
with st.sidebar:
    st.header("Settings")
    st.radio("Enhancement mode", [MODE_COLOR, MODE_BW], key="mode")
    st.selectbox("Thresholding (B&W mode)", ["Adaptive", "Otsu"], key="bw_method")
    st.checkbox("Images are already flat scans (skip document detection)", key="flat_images")
    st.caption("The mode is applied when a page is added to the document. PDF pages are always treated as flat.")
    st.button("Clear all pages & start again", on_click=clear_all)

# ================================================================ header + upload
st.title("Document Scanner & Enhancement")
st.caption("Multi-Page Document Scanner and Image Enhancement Using Digital Image Processing")

st.subheader("1. Upload Documents or Pages")
uploads = st.file_uploader(
    "Choose PDF documents and/or JPG / JPEG / PNG photos. Every PDF page becomes a page. You can upload more at any time.",
    type=["pdf", "jpg", "jpeg", "png"], accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_n}",
)
pages_in = []   # (page key, label) for every image / PDF page
for f in uploads or []:
    fkey = f"{f.name}_{f.size}"
    with st.spinner(f"Reading {f.name}..."):
        for pk, label in expand_file(fkey, f):
            if pk not in st.session_state.dismissed:
                pages_in.append((pk, label))
st.write(f"**Selected Pages: {len(pages_in)}**")

# ================================================================ page processing
if pages_in:
    st.subheader("2. Page Processing")

    # process every page independently
    results = [(n, key, label, st.session_state.scans[key])
               for n, (key, label) in enumerate(pages_in, start=1)]

    # status summary
    for n, key, name, res in results:
        if res["error"]:
            st.write(f"Page {n} ({name}) → ⚠ {res['error']}")
        elif key in st.session_state.added:
            st.write(f"Page {n} ({name}) → ✓ Added to document")
        else:
            st.write(f"Page {n} ({name}) → ✓ Successfully processed")

    ready = [(key, name) for n, key, name, res in results
             if not res["error"] and key not in st.session_state.added]
    if ready:
        st.button(f"Add all processed pages ({len(ready)})", on_click=add_all, args=(ready,))

    # one card per page
    for n, key, name, res in results:
        with st.container(border=True):
            st.markdown(f"**Page {n}** — {name}")

            if res["error"]:
                st.warning(f"Document boundary could not be detected for Page {n}. "
                           "Please try another image.")
                c1, c2 = st.columns(2)
                if res["original"] is not None:
                    c1.image(to_rgb(res["original"]), caption="Original Page", use_container_width=True)
                    c2.image(res["edges"], caption="Edges found", use_container_width=True)
                b1, b2, b3, b4 = st.columns(4)
                b1.button("Retry (relaxed detection)", key=f"retry_{key}",
                          on_click=retry_page, args=(key,), disabled=res["original"] is None)
                b2.button("Use full image & enhance", key=f"full_{key}",
                          on_click=use_full_image, args=(key,), disabled=res["original"] is None)
                if key in st.session_state.added:
                    b3.write("Original added ✓")
                else:
                    b3.button("Use original page", key=f"orig_{key}",
                              on_click=add_to_document, args=(key, name, True),
                              disabled=res["original"] is None)
                b4.button("Remove page", key=f"rm_{key}", on_click=remove_upload, args=(key,))
                continue

            c1, c2, c3 = st.columns(3)
            c1.image(to_rgb(res["original"]), caption="Original Page", use_container_width=True)
            c2.image(to_rgb(res["annotated"]),
                     caption="Full page (no cropping needed)" if res.get("flat") else "Detected Document + Corners",
                     use_container_width=True)
            c3.image(get_enhanced(key, res), caption=f"Enhanced Page ({st.session_state.mode})",
                     use_container_width=True)

            with st.expander("Show intermediate steps"):
                s1, s2 = st.columns(2)
                s1.image(res["edges"], caption="Canny edges", use_container_width=True)
                s2.image(to_rgb(res["warped"]), caption="Page used for enhancement",
                         use_container_width=True)

            b1, b2 = st.columns(2)
            if key in st.session_state.added:
                b1.success("Added to document ✓")
            else:
                b1.button("Add to Document", key=f"add_{key}",
                          on_click=add_to_document, args=(key, name))
            b2.button("Remove page", key=f"rm_{key}", on_click=remove_upload, args=(key,))

# ================================================================ scanned document
st.divider()
st.subheader("3. Scanned Document")
pages = st.session_state.pages

if not pages:
    st.info("No pages yet. Upload photos above and click **Add to Document**.")
else:
    per_row = 4
    for start in range(0, len(pages), per_row):
        cols = st.columns(per_row)
        for offset, page in enumerate(pages[start:start + per_row]):
            i = start + offset
            with cols[offset]:
                st.image(page["image"], caption=f"Page {i + 1}", use_container_width=True)
                u, d, x = st.columns(3)
                u.button("⬆", key=f"up_{page['id']}", help="Move up (earlier)",
                         on_click=move_page, args=(i, -1), disabled=i == 0)
                d.button("⬇", key=f"down_{page['id']}", help="Move down (later)",
                         on_click=move_page, args=(i, 1), disabled=i == len(pages) - 1)
                x.button("🗑", key=f"del_{page['id']}", help="Delete page",
                         on_click=delete_page, args=(i,))

    st.markdown("**Page order:** " + " → ".join(f"Page {i + 1}" for i in range(len(pages))))
    st.markdown(f"### Total Pages: {len(pages)}")

    images = [p["image"] for p in pages]
    d1, d2 = st.columns(2)
    d1.download_button("Download Multi-Page PDF", data=generate_pdf(images),
                       file_name="scanned_document.pdf", mime="application/pdf")
    d2.download_button("Download Individual Images (ZIP)", data=images_to_zip(images),
                       file_name="scanned_pages.zip", mime="application/zip")
    st.caption("Need more pages? Upload more files in section 1 and add them.")
