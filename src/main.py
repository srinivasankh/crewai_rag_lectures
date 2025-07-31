import os
from rag.pdf_loader import load_pdfs_from_folder, chunk_document
from rag.embed_store import EmbedStore
from crewai import Agent, Task, Crew, Process
from rag.retrieval_tool import retrieve_from_pdf

def build_vector_store(pdf_folder, force_rebuild=False):
    store = EmbedStore()
    if not os.path.exists(store.persist_path) or force_rebuild:
        print("Building embeddings and vector store...")
        docs = load_pdfs_from_folder(pdf_folder)
        texts, metas = [], []
        for doc in docs:
            for chunk in chunk_document(doc["content"]):
                texts.append(chunk)
                metas.append({
                    "source": doc["source"],
                    "page": doc["page"],
                    "content": chunk
                })
        store.build_index(texts, metas)
        print("Vector store built and persisted.")
    else:
        print("Vector store exists, loading...")
        store.load_index()
    return store

def main():
    from dotenv import load_dotenv
    load_dotenv()  # Looks for .env in root

    pdf_folder = "data/lectures"
    build_vector_store(pdf_folder)

    # === Agents (defined in Python) ===
    retriever_agent = Agent(
        role="Lecture Retriever",
        goal="Extract precise, factual answers and code snippets from lecture PDFs.",
        backstory=(
            "You're an expert at finding information from large documents and returning accurate, well-cited content."
        ),
        tools=[retrieve_from_pdf],
        verbose=True,
    )

    verifier_agent = Agent(
        role="Answer Verifier",
        goal="Validate that answers from the retriever are correct, well-cited, and accurately reflect the lecture PDFs.",
        backstory=(
            "You are a meticulous academic with an eye for detail, focused on ensuring all answers are true to the source material and all code snippets and citations are accurate. You never hesitate to flag errors or misrepresentations."
        ),
        tools=[retrieve_from_pdf],
        verbose=True,
    )

    # === Tasks (defined in Python) ===
    answer_question_task = Task(
        description=(
            "Given a user question, consult the lecture PDFs and return the most accurate answer. "
            "If the answer includes code, include the full code block. Always cite the PDF and page number. "
            "Your final answer MUST be grounded in the retrieved content and clearly show the relevant excerpts or code."
        ),
        expected_output=(
            "A concise, factual answer that includes code snippets if found, with references to source PDF and page number."
        ),
        agent=retriever_agent,
    )

    verify_answer_task = Task(
        description=(
            "Review the answer provided by the retriever for the user question. "
            "Cross-check the answer against the retrieved lecture PDF content to ensure it is factually correct, fully cited, and matches the source material. "
            "Clearly state if the answer is correct, and if not, point out inaccuracies and provide corrections. "
            "Your final output MUST include a short justification and reference the relevant sources."
        ),
        expected_output=(
            "A verification report stating if the answer is accurate, partially accurate, or inaccurate, with reasons and corrections if necessary."
        ),
        agent=verifier_agent,
        # Pass the answer from the first task as context (CrewAI will handle this in sequential process)
    )

    # === Crew and Workflow ===
    crew = Crew(
        agents=[retriever_agent, verifier_agent],
        tasks=[answer_question_task, verify_answer_task],
        process=Process.sequential,  # Runs tasks in sequence
    )

    print("Ask your question about the lecture PDFs (type 'exit' to quit):")
    while True:
        user_question = input("Q: ")
        if user_question.lower() in ("exit", "quit"):
            break

        result = crew.kickoff(inputs={"question": user_question})

        # CrewAI may return a dict with task results if multiple tasks
        if isinstance(result, dict):
            print("\nAnswer:")
            print(result.get("answer_question"))
            print("\nVerification Report:")
            print(result.get("verify_answer"))
        else:
            print(result)

if __name__ == "__main__":
    main()
