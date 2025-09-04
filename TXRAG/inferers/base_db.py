
from typing import List
import abc

class BaseKeywordDB(abc.ABC):
    
    @abc.abstractmethod
    def get(self, keywords: List[str]) -> List[str]:
        """
        Récupère les documents associés aux mots clés.
        """
        pass
    
    @abc.abstractmethod
    def insert_keyword(self, keyword: str, description: str) -> None:
        """
        Ajoute un mot clé à la base de données.
        """
        pass
    
    @abc.abstractmethod
    def insert_document(self, document: str, keywords: List[str]) -> None:
        """
        Insère un document dans la base de données.
        """
        pass
