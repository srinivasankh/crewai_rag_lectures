import os
from PyPDF2 import PdfReader
from typing import List, Dict

def load_pdfs_from_folder(folder: str) -> List[Dict]:
    docs = []
    for filename in os.listdir(folder):
        if filename.endswith('.pdf'):
            path = os.path.join(folder, filename)
            reader = PdfReader(path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                docs.append({
                    "source": filename,
                    "page": i + 1,
                    "content": text
                })
    return docs

def chunk_document(content: str, chunk_size=800, overlap=50) -> List[str]:
    chunks = []
    start = 0
    while start < len(content):
        end = min(start + chunk_size, len(content))
        chunks.append(content[start:end])
        start = end - overlap
    return chunks
