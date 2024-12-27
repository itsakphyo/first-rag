from first_rag.config import settings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


llm = ChatOpenAI(model="gpt-4o", api_key=settings.OPENAI_API_KEY)

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question} and chat history: {history}
"""
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


def generate_response(text: str, history ,db_path: str | None = None) -> str:
    if not db_path:
        messages = [{"role": "user", "content": text}]
        return llm.invoke(messages).content
    else:
        embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)

        db = Chroma(persist_directory=db_path, embedding_function=embeddings)

        results = db.similarity_search_with_relevance_scores(text, k=3)
        
        context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])

        prompt = prompt_template.format(context=context_text, question=text, history=history)

        response_text = llm.invoke(prompt)

        return response_text.content

