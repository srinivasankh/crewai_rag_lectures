from crewai_tools import tool
from .embed_store import EmbedStore

@tool("pdf_retriever")
def retrieve_from_pdf(query: str) -> str:
    """
    Retrieve the most relevant chunks from lecture PDFs for the user's query.
    """
    store = EmbedStore()
    store.load_index()
    results = store.query(query)
    formatted = ""
    for res in results:
        formatted += f"\n[Source: {res['source']} - Page {res['page']}]\n"
        formatted += res['content'][:1000]
        formatted += "\n---\n"
    return formatted
