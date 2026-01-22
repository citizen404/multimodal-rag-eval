import os
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset
from database import RAGSystem
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("API_KEY не найден")

def run_evaluation():
    rag = RAGSystem()
    rag.build_or_load_index() # Загружаем уже созданную базу

    # 1. Подготовим тестовые вопросы и ответы через GPT

    test_questions = [
        "What is the main feature that distinguishes a smart grid from a traditional grid?",
        "What four operations does a traditional power grid support?",
        "Why is a communication network critical for smart grid operation?",
        "What is a microgrid?",
        "Which wireless standard is commonly used between smart meters and home appliances?",
        "Why are wireless mesh networks suitable for microgrids?",
        "What does Vehicle-to-Grid (V2G) mean?"
    ]
    
    # Сюда запишем эталонные ответы (Ground Truth) 
    ground_truths = [
        "A smart grid enables two-way flows of electricity and information, unlike traditional one-way grids.",
        "It supports electricity generation, transmission, distribution, and control.",
        "It enables real-time pricing, monitoring, automation, and self-healing of the grid.",
        "A microgrid is a group of distributed generators and loads that can operate autonomously.",
        "ZigBee Smart Energy Profile is commonly used for smart meter to appliance communication.",
        "They are self-organizing, reliable, and support multiple communication paths.",
        "V2G allows electric vehicles to supply power back to the grid."
    ]

    answers = []
    contexts = []

    print("Собираем ответы от нашего RAG для оценки...")
    for query in test_questions:
        response, docs = rag.query(query)

        if not docs: # Если нет найденных документов (пустой поиск)
            answers.append("")
            contexts.append([])
        else:
            answers.append(response) # Извлекаем текст из найденных документов для оценки качества поиска
            contexts.append([doc.page_content for doc in docs]) 

    # 2. Формируем датасет для Ragas
    data = {
        "question": test_questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(data)

    # 3. Запускаем оценку
    
    ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    ragas_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))


    print("Запуск Ragas...")
    result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall],
    llm=ragas_llm,
    embeddings=ragas_emb
)

    result.to_pandas().to_csv("evaluation_report.csv", index=False)
    print("\n--- РЕЗУЛЬТАТЫ ОЦЕНКИ ---")
    print(result)

if __name__ == "__main__":
    run_evaluation()
