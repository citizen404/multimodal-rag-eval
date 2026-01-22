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
    
    # Изменено: если базы нет, обрабатываем всю папку data/ вместо одного файла
    if not os.path.exists("chroma_db"):
        print("База не найдена. Начинаем парсинг всей папки data...")
        processor = PDFProcessor()
        all_chunks = []
        
        # Собираем чанки из всех PDF в папке data/
        data_dir = "data"
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith(".pdf"):
                    chunks = processor.process(os.path.join(data_dir, file))
                    all_chunks.extend(chunks)
        
        if not all_chunks:
            print("Ошибка: В папке 'data' не найдено PDF или чанки не созданы.")
            return

        rag.build_or_load_index(all_chunks)
    else:
        rag.build_or_load_index()

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