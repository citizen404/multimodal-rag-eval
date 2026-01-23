import os
import re
import pymupdf4llm
from dotenv import load_dotenv
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY не найден")

client = OpenAI(api_key=api_key)

class PDFProcessor:
    def __init__(self, img_dir="extracted_images"):
        self.img_dir = img_dir
        os.makedirs(self.img_dir, exist_ok=True)

    def _clean_text(self, text):
        """Очистка текста."""
        # Удаляем ссылки [1], [1, 2]
        text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
        text = re.sub(r'\[\d+[–-]\d+\]', '', text)
        # Удаляем артефакты форматирования
        text = text.replace('**', '').replace('_', '')
        # Оставляем только одиночные переносы строк для сохранения списков
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _get_image_caption(self, image_path):
        import base64
        try:
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode('utf-8')
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image for a RAG system. Focus on technical details and data. Be very concise."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ],
                }],
                max_tokens=200 # cost control
            )
            return response.choices[0].message.content
        except Exception:
            return ""

    def process(self, pdf_path):
        # 1. Экстракция
        md_text = pymupdf4llm.to_markdown(pdf_path, write_images=True, image_path=self.img_dir)

        # 2. Обогащение описаниями картинок
        img_pattern = r"\!\[.*?\]\((.*?)\)"
        def replacer(match):
            img_path = match.group(1)
            if os.path.exists(img_path):
                caption = self._get_image_caption(img_path)
                return f"\n\n[IMAGE_DESCRIPTION: {caption}]\n\n" if caption else ""
            return ""

        enriched_text = re.sub(img_pattern, replacer, md_text)
        cleaned_text = self._clean_text(enriched_text)

        # 3. Гранулярное разбиение
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,         # для захвата мелких деталей
            chunk_overlap=200,      # нахлест, чтобы не терять на стыках
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        
        # Создаем документы напрямую
        final_chunks = text_splitter.create_documents([cleaned_text])
        
        # 4. Метаданные и фильтрация
        valid_chunks = []
        for chunk in final_chunks:
            content = chunk.page_content.strip()
            # Фильтруем слишком короткие куски и малоинформативные описания картинок
            if len(content) < 20 or ("[IMAGE_DESCRIPTION]" in content and len(content) < 100):
                continue
            # Пропускаем пустые куски
            if len(content) < 20:
                continue
                
            chunk.metadata["source"] = os.path.basename(pdf_path)
            chunk.metadata["content_type"] = "image_caption" if "IMAGE_DESCRIPTION" in content else "text"
            valid_chunks.append(chunk)
            
        return valid_chunks

if __name__ == "__main__":
    from database import RAGSystem
    import shutil

    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")

    input_folder = "data"
    processor = PDFProcessor()
    all_chunks = []
    
    pdf_files = [f for f in os.listdir(input_folder) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"В папке {input_folder} не найдено PDF.")
    else:
        for pdf_file in pdf_files:
            pdf_path = os.path.join(input_folder, pdf_file)
            print(f"--- Обработка: {pdf_file} ---")
            processor.img_dir = f"extracted_images/{pdf_file.replace('.', '_')}"
            
            chunks = processor.process(pdf_path)
            print(f"Создано чанков: {len(chunks)}")
            all_chunks.extend(chunks)

        if all_chunks:
            print(f"\nИтого чанков: {len(all_chunks)}")
            rag = RAGSystem()
            rag.build_or_load_index(chunks=all_chunks) 
            print("База готова.")
