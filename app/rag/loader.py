import fitz
import numpy as np
from pathlib import Path
from langchain_core.documents import Document

_ocr_engine = None

import gc

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
        del img_np
        gc.collect()
        if results:
            return "\n".join([res[1] for res in results if res and len(res) > 1 and res[1]])
    except Exception:
        pass
    return ""

def _extract_page_layout_text(page: fitz.Page) -> str:
    blocks = page.get_text("blocks")
    if not blocks:
        return ""
    
    text_blocks = [b for b in blocks if len(b) >= 5 and isinstance(b[4], str) and b[4].strip()]
    if not text_blocks:
        return ""

    page_rect = page.rect
    page_width = page_rect.width
    
    left_column = []
    right_column = []
    full_width = []
    
    mid_point = page_width / 2.0
    for b in text_blocks:
        x0, y0, x1, y1, content = b[0], b[1], b[2], b[3], b[4]
        if (x1 - x0) > 0.7 * page_width:
            full_width.append((y0, content))
        elif x1 <= mid_point + 20:
            left_column.append((y0, content))
        elif x0 >= mid_point - 20:
            right_column.append((y0, content))
        else:
            full_width.append((y0, content))
            
    left_column.sort(key=lambda item: item[0])
    right_column.sort(key=lambda item: item[0])
    full_width.sort(key=lambda item: item[0])
    
    combined = []
    if left_column or right_column:
        for y, content in left_column:
            combined.append(content.strip())
        for y, content in right_column:
            combined.append(content.strip())
        for y, content in full_width:
            combined.append(content.strip())
        return "\n\n".join(combined)
    
    text_blocks.sort(key=lambda b: (b[1], b[0]))
    return "\n\n".join([b[4].strip() for b in text_blocks])

def _extract_tables_markdown(page: fitz.Page) -> str:
    markdown_tables = []
    try:
        tabs = page.find_tables()
        if tabs and tabs.tables:
            for t_idx, table in enumerate(tabs.tables, start=1):
                df_data = table.extract()
                if not df_data or len(df_data) < 2:
                    continue
                header = [str(c or "").strip().replace("\n", " ") for c in df_data[0]]
                separator = ["---"] * len(header)
                table_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(separator) + " |"]
                for row in df_data[1:]:
                    row_cells = [str(c or "").strip().replace("\n", " ") for c in row]
                    table_lines.append("| " + " | ".join(row_cells) + " |")
                markdown_tables.append(f"\n\n### Table {t_idx}\n" + "\n".join(table_lines))
    except Exception:
        pass
    return "\n".join(markdown_tables)

def load_pdf(file_path: str | Path, session_id: str = "default") -> list[Document]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    
    documents = []
    with fitz.open(str(path)) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = _extract_page_layout_text(page)
            if not text:
                text = page.get_text()
                
            table_md = _extract_tables_markdown(page)
            if table_md:
                text = (text + table_md).strip()

            should_ocr = len(text.strip()) < 100 or len(text.split()) < 15
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
                "has_tables": bool(table_md)
            }
            documents.append(Document(page_content=text, metadata=metadata))
    return documents

def load_document(file_path: str | Path, session_id: str = "default") -> list[Document]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
        
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path, session_id)
        
    if ext in [".png", ".jpg", ".jpeg"]:
        engine = get_ocr_engine()
        extracted_text = ""
        if engine:
            try:
                with open(path, "rb") as f:
                    img_bytes = f.read()
                results, _ = engine(img_bytes)
                if results:
                    extracted_text = "\n".join([res[1] for res in results if res and len(res) > 1 and res[1]])
            except Exception:
                pass
        if not extracted_text:
            extracted_text = f"[Image File: {path.name}]"
        return [Document(page_content=extracted_text, metadata={"source": str(path), "page": 0, "images": []})]

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return [Document(page_content=content, metadata={"source": str(path), "page": 0, "images": []})]
    except Exception as e:
        return [Document(page_content=f"[Error reading file {path.name}: {e}]", metadata={"source": str(path), "page": 0, "images": []})]
