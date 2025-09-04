"""
title: TX7709
author: Ryustiel
date: 2025-03-14
version: 1.0
license: MIT
description: A pipeline for generating text using Raphlib
requirements: tiktoken
"""

from typing import (
    List, Dict, Union, 
    Generator, Iterator, Literal,
    TypedDict, Coroutine, Dict, Optional
)
from langchain_core.messages import AIMessage, SystemMessage, AIMessageChunk

import re, os, pydantic, tiktoken, asyncio, json
from collections import defaultdict
from raphlib import tool
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage

# =================================================================== UTILITIES

def extract_source_contexts(content: str) -> Dict[int, str]:
    """
    Process the content and return a dictionary:
      { source_id (int): source_context }
    """
    source_dict = defaultdict(list)

    for i, match in enumerate(re.findall(r"<source id=[^>]*>.*?</source>", content, re.DOTALL)):
        context = match.strip()   # The captured content
        # find what's in between the "" in the first sequence name="" that appear in string context
        match = re.search(r'name="([^"]+)"', context)
        if match:
            name = match.group(1)
        else:
            name = f"source_{i+1}"
        source_dict[name].append(context)
        
    result = {}
    for source_id, chunks in source_dict.items():
        result[source_id] = "".join(chunks)
    
    return result

def count_tokens(text: str) -> int:
    return len(tiktoken.get_encoding("cl100k_base").encode(text))
                
# =================================================================== PROMPTS

def make_prompt(
    prompt_type: Literal["w84_docs", "w84_questions", "ask_for_confirmation", "check_confirmation", "process_output"],
    messages: List[Dict[str, str]],
    n_uploaded_documents: Optional[int] = None,
    table_prompt: Optional[str] = None,
) -> List[HumanMessage|AIMessage|SystemMessage]:
    """Produit le prompt demandé."""
    
    system_message = "Tu es une interface d'analyse de documents."
    
    match prompt_type:
        case "w84_docs":
            system_message += (
                "Tu n'as encore reçu aucun document. "
                "Demande à l'utilisateur de te fournir des documents pour les analyser et y trouver de l'information. "
                "Pour ce faire, l'utilisateur peut les glisser-déposer dans la fenêtre de chat. "
            )
        case "w84_questions":
            system_message += (
                f"L'utilisateur t'as fourni {n_uploaded_documents or 0} documents à analyser. "
                "Tu ne peux pas les lire avant d'avoir compris son besoin en information puis lancé l'analyse selon le protocole défini ci-dessous. "
                "\n#1. Demande à l'utilisateur ce qu'il veut savoir sur ces documents. "
                "\n#2. Tant que son besoin en information n'est pas clair, discute brièvement et naturellement avec lui pour clarifier sa demande ou lui donner des idées. "
                "Garde tes suggestions limitées et évocatives (\"on pourrait chercher le nombre de fois que ..., qu'est ce que vous en dites?\") "
                "\n#3. Une fois sa demande claire, synthétise sa demande d'information sous la forme d'une liste d'informations à rechercher "
                "pour lui demander si ça correspond bien à ce qu'il veut savoir. "
                "\n#4. Dès que l'utilisateur est en connaissance de cause que cette liste sera utilisée pour l'analyse "
                "et qu'il a confirmé que la liste que tu lui as proposée lui convient, "
                "Lance ton tool analyze_documents avec cette liste pour commencer "
                "à lire les documents et à trouver ces informations. "
            )
        case "ask_for_confirmation":
            system_message += (
                "L'utilisateur a fourni des documents et vous vous êtes mis d'accord sur l'information à chercher. "
                "L'utilisateur doit exprimer son accord avec les sujets proposés pour démarrer l'analyse. "
                "L'analyse ne pourra pas commencer tant l'utilisateur n'aura pas confirmé que les sujets proposés lui conviennent. "
                "L'utilisateur peut toujours apporter des révisions à cette expression de besoin en information."
            )
        case "check_confirmation":
            system_message += (
                "Dans ta réponse structurée, "
                "Indique si l'utilisateur a demandé de lancer l'analyser de documents "
                "ou si il a simplement répondu positivement à la demande de confirmation du LLM, "
                "suite à une reformulation des questions de l'utilisateur ou autres."
                "Ou si il n'a pas répondu positivement (fait une remarque, dit non, abandonne, etc...)"
            )
        case "process_output":
            system_message += (
                "\n\nPrésente les résultats du brouillon dans un unique tableau markdown (**SANS** la balise ```markdown) et correctement formatté d'après les contraintes suivantes: "
                
                "\n\n#1. Chaque cellule du tableau ne peut contenir qu'un chiffre et unité (21.3 dollars, 4 degres C, 3eme, ...), "
                "un mot clé ou un paragraphe simple. Si tu dois lister des éléments fais le sous la forme d'un paragraphe à virgule (a, b et c). "
                "Pour des raisons de sécurité, il est interdit de citer du code ou des caractères qui peuvent se trouver dans du code. (donc tu n'as le droit qu'aux lettres, chiffres et ponctuation simple). "
                
                "\n\n#2. N'utilise que des lettres, des chiffres et de la ponctuation simple (,;:?!) sans autre caractère spécial dans les cellules du tableau. "
                "Une cellule ne peut contenir au plus qu'un paragraphe simple, pas de retour à la ligne."
                
                "\n\n#3. Attention, toutes les informations doivent être présentées dans un unique tableau final, "
                "où chaque ligne correspond à un document et chaque colonne à une information demandée par l'utilisateur. "
                "Il **ne doit pas** être encapsulé dans ```markdown```. Tu dois le produire directement tel quel dans ta réponse."
                
                "\nLa première colonne doit impérativement contenir le nom du document."
                "\n\nBrouillon à reformatter correctement: " + (table_prompt or "")
            )

    return [
            SystemMessage(content=system_message),
        ] + [
            AIMessage(content=message["content"]) 
            if message["role"] == "assistant"
            else HumanMessage(content=message["content"]) 
            for message in messages if message["role"] in ("assistant", "user")
        ]
                
# =================================================================== PIPELINE

class Pipeline:

    class Valves(pydantic.BaseModel):
        UTC_API_KEY: str = ""
        UTC_ENDPOINT: str = ""
        TOKEN_LIMIT_ANALYZE: str = ""
        TOKEN_LIMIT_CHAT: str = ""
        MODEL_NAME_ANALYZE: str = ""
        MODEL_NAME_CHAT: str = ""

    def __init__(self):
        self.name = "Analyse de Documents"
        self.valves = self.Valves(**{key: os.getenv(key, "") for key in self.Valves.model_fields.keys()})

    def pipe(
        self, user_message: str, model_id: str, messages: List[Dict[str, str]], body: dict
    ) -> Union[str, Generator, Iterator]:
        
        if any(value == "UNDEFINED" for value in self.valves.model_dump().values()):
            return "Please set ALL the valves before using this pipeline."

        def stream():
            
            try:
            
                if count_tokens(" ".join([msg["content"] for msg in messages if msg["role"] != "system"])) > int(self.valves.TOKEN_LIMIT_CHAT):
                    yield (
                        "Cette conversation dépasse la limite de tokens autorisée. "
                        + "Veuillez réduire le nombre de messages ou la taille des messages."
                    )
                    return

                MODE = "w84_docs"  # Initial mode is to ask for documents

                # I ============== Extraction des documents à partir du premier message de la conversation

                if messages[0]["role"] == "system":

                    # Get the system message containing all the docs, for example:
                    system_msg = next(msg for msg in messages if msg["role"] == "system")
                    content = system_msg["content"]

                    uploaded_documents = extract_source_contexts(content)
                    if len(uploaded_documents) > 0:
                        MODE = "w84_questions"  # If documents are found, switch to questions mode
                    
                else:
                    uploaded_documents = {}
                    
                # II ============= Vérification de la taille des documents

                qt = False
                for name, doc in uploaded_documents.items():
                    token_count = count_tokens(doc)
                    print(f"Document size: {token_count} tokens, vs TOKEN_LIMIT: {self.valves.TOKEN_LIMIT_ANALYZE}")
                    if token_count > int(self.valves.TOKEN_LIMIT_ANALYZE):
                        yield f"\nLe document \"{name}\" dépasse la limite de tokens autorisée ({token_count} > {self.valves.TOKEN_LIMIT_ANALYZE}). Veuillez réduire la taille du document."
                        qt = True
                if qt:
                    return
                    
                # III ============================ Initialisation de la conversation
            
                chat_llm = ChatOpenAI(
                        api_key = self.valves.UTC_API_KEY, 
                        base_url = self.valves.UTC_ENDPOINT,
                        model = self.valves.MODEL_NAME_CHAT,
                        temperature = 0.0,
                    )
            
                if MODE == "w84_docs":
                    
                    print("Mode documents")
                
                    PROMPT = make_prompt("w84_docs", messages)
                    
                    response_generator = chat_llm.stream(PROMPT)  # Pas besoin de tools ici, juste une conversation initiale
                
                elif MODE == "w84_questions":  # <=> else
                    
                    print("Mode questions")
                
                    class AnalyzeDocumentsInput(pydantic.BaseModel):
                        information_request: str = pydantic.Field(..., description="Un prompt qui explique les informations demandées en précisant leur format / unité / métrique pour chaque type d'information dès que possible.")
                    @tool
                    def analyze_documents(inp: AnalyzeDocumentsInput) -> str:
                        """
                        Recherche les informations demandées par l'utilisateur dans les documents. 
                        Le paramètre information_request doit demander avec le plus de détails possible 
                        toutes les informations requises, 
                        par exemple \"Trouve le nom-prénom de toutes les personnes mentionnées, 
                        invente un titre alternatif pour le document jusqu'à deux caractères, ...\"
                        Précise les unités / format / métrique dès que possible, par exemple "nombre de pages", "date jour mois année", "liste de mots uniques", etc...
                        """
                        pass
                        
                    response_generator = chat_llm.bind_tools([analyze_documents]).stream(
                        make_prompt("w84_questions", messages, len(uploaded_documents))
                    )
                    
                    # IV ============================ Vérification des tool calls
                    
                    # get next iteration of the response generator
                    msg: AIMessageChunk = response_generator.__next__()
                    if msg.tool_calls:  # Called analyze_documents
                        for m in response_generator:
                            msg += m  # Collect all chunks of the response
                
                        information_request = msg.tool_calls[0]["args"]["information_request"]
                
                        # V ============================= Si tool call, vérification que l'utilisateur a exprimé son accord
                        # (changement de prompt vers "w84_confirm"), sinon repartir en mode questions ("w84_questions")

                        yield {"event":{"type":"status","data":{"description":"Vérification des paramètres...","done": False}}}
                        
                        class UserConfirmedResponse(pydantic.BaseModel):
                            they_said_do_analyze: bool = pydantic.Field(..., description="L'utilisateur a-t-il ordonné ou demandé de lancer l'analyse ?")
                            they_said_yes: bool = pydantic.Field(..., description="L'utilisateur a-t-il formulé une réponse positive (ok, oui, yes, affirmatif, ouais, ...) ?")

                        resultat: UserConfirmedResponse = chat_llm.with_structured_output(UserConfirmedResponse).invoke(
                            make_prompt("check_confirmation", messages)
                        )
                        
                        if resultat.they_said_do_analyze or resultat.they_said_yes:
                            
                            print("L'utilisateur a confirmé")
                            
                            yield {"event":{"type":"status","data":{"description":"Analyse des Documents","done": False}}}
                        
                            analyzer_llm = ChatOpenAI(
                                api_key = self.valves.UTC_API_KEY, 
                                base_url = self.valves.UTC_ENDPOINT,
                                model = self.valves.MODEL_NAME_ANALYZE,
                                temperature = 0.0,
                            )
                        
                            # queries: List[Coroutine[AIMessage]] = [
                            #     analyzer_llm.ainvoke([SystemMessage(
                            #         content=(
                            #             "Répond en détails aux questions posées par l'utilisateur sur le document."
                            #             + "\n\n## Questions: " + information_request
                            #             + "\n\n## Document: " + doc
                            #         )
                            #     )])
                            #     for doc in uploaded_documents.values()
                            # ]
                            # # run with asyncio.gather to run all queries concurrently
                            # responses: List[AIMessage] = asyncio.run(asyncio.gather(*queries))
                            # print("\n\n", [r.content for r in responses], "\n\n")
                            
                            responses: List[AIMessage] = []
                            for doc_name, doc in uploaded_documents.items():
                                yield {"event":{"type":"status","data":{"description":f"Analyse du document {doc_name}","done": False}}}
                                response: AIMessage = analyzer_llm.invoke([
                                    SystemMessage(
                                        content=(
                                            "Répond en détails aux questions posées par l'utilisateur sur le document."
                                            + "\n\n## Questions: " + information_request
                                            + "\n\n## Document: " + doc
                                        )
                                    )
                                ])
                                responses.append(response)
                            
                            yield {"event":{"type":"status","data":{"description":"Synthèse des Résultats","done": False}}}
                            
                            # Répondre avec la prompt réponse markdown
                            table_response: AIMessage = analyzer_llm.invoke([
                                SystemMessage(
                                    content=(
                                        "Tu as reçu des réponses à des questions posées par l'utilisateur pour chaque document qu'il t'a fourni. "
                                        + "Présente ces résultats dans un unique tableau markdown synthétique et uniforme "
                                        + "où chaque ligne correspond à un document et "
                                        + "chaque colonne correspond à une question posée par l'utilisateur. "
                
                                        + "\n\n#1. Chaque cellule du tableau ne peut contenir qu'un chiffre et unité (21.3 dollars, 4 degres C, 3eme, ...), "
                                        + "un mot clé ou un paragraphe simple. Si tu dois lister des éléments fais le sous la forme d'un paragraphe à virgule (a, b et c). "
                                        + "Pour des raisons de sécurité, il est interdit de citer du code ou des caractères qui peuvent se trouver dans du code. (donc tu n'as le droit qu'aux lettres, chiffres et ponctuation simple). "
                
                                        + "\n\n#2. N'utilise que des lettres, des chiffres et de la ponctuation simple (,;:?!) sans autre caractère spécial dans les cellules du tableau. "
                                        + "Une cellule ne peut contenir au plus qu'un paragraphe simple, pas de retour à la ligne."
                
                                        + "\n\n#3. Attention, toutes les informations doivent être présentées dans un unique tableau final, "
                                        + "La première colonne doit impérativement contenir le nom du document, "
                                        + "\nIl est impératif de respecter le formattage ligne=document, colonne=question, "
                                        + "et de produire un unique tableau. "
                                        
                                        + "\n\nRéponses brutes à réorganiser en un tableau mieux compartimenté:" 
                                        + (
                                            "\n".join(
                                                f"| {document_name} | {response.content} |" for document_name, response in zip(uploaded_documents.keys(), responses)
                                            ) 
                                            if responses 
                                            else "Aucun résultat trouvé."
                                        )
                                    )
                                )
                            ])
                            print("\n\n", table_response.content, "\n\n")
                            
                            yield {"event":{"type":"status","data":{"description":"","done": True}}}
                            
                            response_generator = chat_llm.stream(
                                make_prompt("process_output", messages, table_prompt=table_response.content)
                            )
                            
                        else:
                            
                            print("L'utilisateur n'a pas confirmé")
                            
                            yield {"event":{"type":"status","data":{"description":"","done": True}}}
                            
                            # Repartir en mode confirmation, puis en mode question si toujours pas de confirmation
            
                            response_generator = chat_llm.stream(
                                make_prompt("ask_for_confirmation", messages, len(uploaded_documents))
                            )
                            
                    else:
                        yield msg.content
                        
                for msg in response_generator:  # Stream the rest of the response
                    yield msg.content
                
            except Exception as e:
                yield f"\n\n{type(e)} {e} : {__name__}"
        
        return stream()
