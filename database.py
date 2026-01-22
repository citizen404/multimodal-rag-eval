import os
import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class RAGSystem:
    def __init__(self, index_path="chroma_db"):
        self.index_path = index_path

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

        self.vector_store = None

    def build_or_load_index(self, chunks=None):
        """Создает или подгружает базу данных."""
        if chunks:
            print(f"Отправка {len(chunks)} чанков в OpenAI и создание базы...")
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.index_path
            )
            print("База успешно создана.")
        else:
            if os.path.exists(self.index_path):
                print("Загрузка существующей базы Chroma...")
                self.vector_store = Chroma(
                    persist_directory=self.index_path,
                    embedding_function=self.embeddings
                )
            else:
                raise ValueError("База данных не найдена по указанному пути.")

    def query(self, user_question):
        if not self.vector_store:
            return "База данных не инициализирована.", []

        # Ретривер (поиск топ-5 релевантных кусков)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

        template = """Answer ONLY in English. 
        Be maximally concise and straight to the point. Do not repeat the question. 
        Use ONLY the provided facts. If specific numbers or names are not in the context, do not invent them.
        If the answer is not in the context, respond "Information not found".

        Context: {context}
        Question: {question}
        Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        # Выполняем поиск документов для источников
        source_docs = retriever.invoke(user_question)

        # В методе query перед объединением в строку контекста:
        valid_docs = [doc for doc in source_docs if len(doc.page_content.strip()) > 50]

        if not valid_docs:
            return "There is no information in the document to answer this question.", []
        # Генерируем ответ
        answer = (
            prompt
            | self.llm
            | StrOutputParser()
        ).invoke({
            "context": "\n\n".join(d.page_content for d in valid_docs),
            "question": user_question
        })
        
        return answer, valid_docs
