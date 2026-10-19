import os
import pypdfium2 as pdfium

base_dir = r"D:\Data Train"
folders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))])

for folder in folders:
    fpath = os.path.join(base_dir, folder)
    print(f"\n==========================================")
    print(f"FOLDER: {folder}")
    print(f"==========================================")
    pdf_files = sorted([f for f in os.listdir(fpath) if f.lower().endswith(".pdf")])
    for fn in pdf_files:
        full_pdf = os.path.join(fpath, fn)
        try:
            doc = pdfium.PdfDocument(full_pdf)
            page_count = len(doc)
            p1_text = doc[0].get_textpage().get_text_range()
            lines = [l.strip() for l in p1_text.splitlines() if l.strip()][:3]
            sample = " | ".join(lines)
            print(f"  [{fn}] ({page_count} trang): {sample[:110]}")
        except Exception as e:
            print(f"  [{fn}] ERROR: {e}")
