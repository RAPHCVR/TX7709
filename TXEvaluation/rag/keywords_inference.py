from typing import List, Set
import json, pydantic, os, dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

with open("rag/data/db_1.json", "r", encoding="utf-8") as f:
    db = json.load(f)
    
with open("rag/data/keyword_prompt_1.json", "r", encoding="utf-8") as f:
    keyword_prompt = f.read()
    
class KW(pydantic.BaseModel):
    keywords: List[str]
    
def respond(user_input: str) -> str:
    """
    Run the RAG system once assuming the user input is the first message a user sends to the system.
    """
    response_1: KW = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),  # type: ignore
        model="gpt-4.1-mini",
    ).with_structured_output(KW).invoke(
        [
            SystemMessage(
                content="Sélectionne tous les mots clés qui correspondent à peu près à la situation de l'utilisateur d'après la conversation."
                        + "\nSélectionne uniquement les mots clés parmi la liste suivante:"
                        + str(list(db["keywords"].keys()))
                        + "\n\nVoici quelques informations additionnelles sur les mots clés et les informations disponibles:\n"
            ),
            HumanMessage(content=user_input),
        ]
    )
    
    chunks: Set[str] = set()
    for k in response_1.keywords:
        chunks.update([doc for doc, doc_kws in db["documents"].items() if k in doc_kws])

    response_2: AIMessage = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),  # type: ignore
        model="o4-mini",
    ).invoke(
        [
            SystemMessage(
                content="Répond à la requête de l'utilisateur. "
                        + "Utilise tes connaissances si elles sont pertinentes."
                        + "\nConnaissances:\n"
                        + "\n".join(chunks)
                        
            ),
            HumanMessage(content=user_input),
        ]
    )
    
    if isinstance(response_2.content, str):
        return response_2.content
    else:
        raise ValueError("Expected response_2.content to be a string, got: " + str(type(response_2.content)))
