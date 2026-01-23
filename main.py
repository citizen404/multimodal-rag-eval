# main.py
import os
import sys
from dotenv import load_dotenv
from ingestion import PDFProcessor
from database import RAGSystem

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set")

def main():
    rag = RAGSystem()
    processor = PDFProcessor()
    data_dir = "data"

    # 1. Пробуем загрузить базу, если она есть
    if os.path.exists("chroma_db"):
        rag.build_or_load_index()
    
    # 2. Проверяем папку data
    # Метод get_indexed_files теперь вернет пустой набор, если базы нет
    indexed_files = rag.get_indexed_files() 
    new_chunks = []

    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(".pdf") and file not in indexed_files:
                print(f"Новый файл: {file}")
                new_chunks.extend(processor.process(os.path.join(data_dir, file)))

    # 3. Если нашли новое — добавляем. Если базы не было — она создастся тут.
    if new_chunks:
        rag.add_to_index(new_chunks)
    elif not os.path.exists("chroma_db"):
        print("База пуста и новых файлов нет. Положите PDF в папку 'data'.")
        return

    print("\n--- RAG Система Готова (OpenAI) ---")
    
    while True:
        query = input("\nВаш вопрос (или 'exit' для выхода): ")
        if query.lower() in ['exit', 'quit', 'выход']: break
        
        answer, sources = rag.query(query)
        print(f"\nОТВЕТ: {answer}")
        print("\nИСТОЧНИКИ:")
        # Чтобы не дублировать один и тот же файл в списке источников
        seen_sources = set()
        for doc in sources:
            src_name = doc.metadata.get('source', 'test.pdf')
            if src_name not in seen_sources:
                print(f"- {src_name}")
                seen_sources.add(src_name)

if __name__ == "__main__":
    main()