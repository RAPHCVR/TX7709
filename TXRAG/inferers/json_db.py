
from typing import List, Dict
import pydantic

from .base_db import BaseKeywordDB

class JSONKeywordDB(BaseKeywordDB, pydantic.BaseModel):
    keywords: Dict[str, str] = {}  # keyword : description
    documents: Dict[str, List[str]] = {}  # Doc : List[keywords]

    def get(self, keywords: List[str]) -> List[str]:
        result = []
        for document, doc_keywords in self.documents.items():
            if any(keyword in doc_keywords for keyword in keywords):
                result.append(document)
        return list(set(result))
    
    def insert_keyword(self, keyword: str, description: str) -> None:
        if keyword not in self.keywords:
            self.keywords[keyword] = description
            
    def insert_document(self, document: str, keywords: List[str]) -> None:
        if document not in self.documents:
            self.documents[document] = keywords