import fitz
import numpy as np
from pathlib import Path
from langchain_core.documents import Document

_ocr_engine = None

def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _ocr_engine = RapidOCR()
        except Exception:
            _ocr_engine = False
    return _ocr_engine if _ocr_engine is not False else None

def _ocr_pixmap(pix: fitz.Pixmap) -> str:
    engine = get_ocr_engine()
    if not engine:
        return ""
    try:
        if pix.alpha or pix.n != 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.h, pix.w, 3))
        results, _ = engine(img_np)
        if results:
            return "\n".join([res[1] for res in results if res and len(res) > 1 and res[1]])
    except Exception:
        pass
    return ""

def _ocr_bytes(img_bytes: bytes) -> str:
    engine = get_ocr_engine()
    if not engine:
        return ""
    try:
        results, _ = engine(img_bytes)
        if results:
            return "\n".join([res[1] for res in results if res and len(res) > 1 and res[1]])
    except Exception:
        pass
    return ""

def load_pdf(file_path: str | Path, session_id: str = "default") -> list[Document]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    
    documents = []
    with fitz.open(str(path)) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            should_ocr = len(text.strip()) < 500 or len(text.split()) < 70 or bool(page.get_images())
            if should_ocr:
                try:
                    pix = page.get_pixmap(dpi=150)
                    ocr_page_text = _ocr_pixmap(pix)
                    if ocr_page_text:
                        native_lower = text.lower()
                        new_lines = [
                            line.strip() for line in ocr_page_text.split("\n")
                            if line.strip() and line.strip().lower() not in native_lower
                        ]
                        if new_lines:
                            text = (text + "\n" + "\n".join(new_lines)).strip()
                except Exception:
                    pass

            page_images = []
            try:
                image_list = page.get_images()
                if image_list:
                    media_dir = Path("app/static/media") / str(session_id)
                    media_dir.mkdir(parents=True, exist_ok=True)
                    for img_index, img_info in enumerate(image_list):
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        if base_image:
                            img_bytes = base_image["image"]
                            img_ext = base_image["ext"]
                            img_name = f"page_{page_num}_img_{img_index}.{img_ext}"
                            img_path = media_dir / img_name
                            with open(img_path, "wb") as img_file:
                                img_file.write(img_bytes)
                            page_images.append(f"/media/{session_id}/{img_name}")
            except Exception:
                pass
                
            metadata = {
                "source": str(path),
                "page": page_num,
                "images": page_images,
            }
            documents.append(Document(page_content=text, metadata=metadata))
    return documents
