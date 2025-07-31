import os
from rag.pdf_loader import load_pdfs_from_folder, chunk_document
from rag.embed_store import EmbedStore
from crewai import Crew, Process
from crewai.config import AgentsConfig, TasksConfig
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

    agents_config = AgentsConfig.from_yaml('src/crew/agents.yaml')
    tasks_config = TasksConfig.from_yaml('src/crew/tasks.yaml')
    agents = agents_config.create_agents()
    tasks = tasks_config.create_tasks(agents)

    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential  # The answer will be generated, then verified
    )

    print("Ask your question about the lecture PDFs (type 'exit' to quit):")
    while True:
        user_question = input("Q: ")
        if user_question.lower() in ("exit", "quit"):
            break

        # Step 1: Generate the answer
        result = crew.kickoff(inputs={"question": user_question})

        # Depending on CrewAI's output, `result` may be a dict with all task results.
        # You may need to access both outputs:
        # e.g., result['answer_question'] and result['verify_answer']
        answer = result.get("answer_question") if isinstance(result, dict) else result
        verification = result.get("verify_answer") if isinstance(result, dict) else None

        print("\nAnswer:")
        print(answer)
        if verification:
            print("\nVerification Report:")
            print(verification)
        else:
            print("\n[No verification output found - check task chaining or CrewAI version.]")

if __name__ == "__main__":
    main()
