import os
import base64
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
import operator

load_dotenv()
client = OpenAI()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Tools

@tool
def process_pdf(file_path: str) -> str:
    # обработка PDF
    import pymupdf4llm
    import re

    try:
        md_text = pymupdf4llm.to_markdown(file_path)
        # Базовая очистка
        md_text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', md_text)
        md_text = re.sub(r'\n{3,}', '\n\n', md_text)
        return md_text.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

@tool
def process_image(file_path: str) -> str:
    # обработка изображения
    try:
        with open(file_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')

        ext = file_path.split('.')[-1].lower()
        mime = "image/png" if ext == "png" else "image/jpeg"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail for a RAG knowledge base. Include all visible text, data, charts, and technical details."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64_image}"}}
                ]
            }],
            max_tokens=500
        )
        description = response.choices[0].message.content
        return f"[IMAGE: {os.path.basename(file_path)}]\n{description}"
    except Exception as e:
        return f"ERROR: {str(e)}"

@tool
def process_text(file_path: str) -> str:
    # обработка текстового файла
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"ERROR: {str(e)}"

@tool
def ingest_to_vectorstore(content: str, source_name: str) -> str:
    # чанкинг
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from database import RAGSystem

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.create_documents([content])
        for chunk in chunks:
            chunk.metadata["source"] = source_name
            chunk.metadata["content_type"] = "agent_ingested"

        valid_chunks = [c for c in chunks if len(c.page_content.strip()) >= 20]

        rag = RAGSystem()
        rag.build_or_load_index(chunks=valid_chunks)

        return f"OK: {len(valid_chunks)} chunks ingested from {source_name}"
    except Exception as e:
        return f"ERROR: {str(e)}"

tools = [process_pdf, process_image, process_text, ingest_to_vectorstore]
llm_with_tools = llm.bind_tools(tools)

# STATE
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    files_processed: int
    files_failed: int

# AGENT NODE
def agent(state: AgentState) -> AgentState:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response], "files_processed": state["files_processed"], "files_failed": state["files_failed"]}

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

# GRAPH
tool_node = ToolNode(tools)

graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile()

# RUN
def run_ingestion_agent(data_folder: str = "data"):
    supported = {'.pdf', '.png', '.jpg', '.jpeg', '.txt', '.md'}

    files = [
        f for f in os.listdir(data_folder)
        if os.path.splitext(f)[1].lower() in supported
    ]

    if not files:
        print(f"No supported files found in {data_folder}/")
        return

    print(f"Found {len(files)} file(s): {files}\n")

    file_list = "\n".join([f"- {f}" for f in files])
    prompt = f"""You are a document ingestion agent. Your job is to process files and store them in a vector database for RAG.

Data folder: {data_folder}
Files to process:
{file_list}

For each file:
1. Choose the right tool based on file extension:
   - .pdf → process_pdf
   - .png, .jpg, .jpeg → process_image  
   - .txt, .md → process_text
2. Call the tool with the full file path: {data_folder}/<filename>
3. If successful, call ingest_to_vectorstore with the content and filename
4. Process ALL files before finishing

Start processing now."""

    from langchain_core.messages import HumanMessage

    result = app.invoke({
        "messages": [HumanMessage(content=prompt)],
        "files_processed": 0,
        "files_failed": 0
    })

    print("\n--- Ingestion Agent Complete ---")
    last_message = result["messages"][-1]
    print(last_message.content)

if __name__ == "__main__":
    run_ingestion_agent("data")