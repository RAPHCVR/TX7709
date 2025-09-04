"""
title: TX7709
author: Ryustiel, RAPHCVR
date: 2025-03-14
version: 2.0
license: MIT
description: A pipeline for generating text using Raphlib
requirements: tiktoken
"""

from typing import (
    List,
    Union,
    Generator,
    Iterator,
    Literal,
    Optional,
)

from typing_extensions import TypedDict

from langchain_core.messages import (
    SystemMessage,
    AIMessage,
    HumanMessage,
    AIMessageChunk,
)
from raphlib import tool

import os, pydantic, tiktoken
from langchain_ollama import ChatOllama

import logging

logging.basicConfig(level=logging.INFO)

# =================================================================== TYPING


class PipeMessageInput(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


class PipeUserInput(pydantic.BaseModel):
    name: str = pydantic.Field(
        ..., description="The name of the user currently writing in this chat."
    )
    id: str = pydantic.Field(
        ..., description="Some alphanumerical id : 529a23c6-b326-..."
    )
    email: str
    role: Literal["user", "admin"]


class PipeBodyInput(pydantic.BaseModel):
    stream: bool
    model: str
    messages: List[PipeMessageInput] = pydantic.Field(
        ..., description="This is a duplicate of PipeInput.messages"
    )
    user: PipeUserInput


class PipeInput(pydantic.BaseModel):
    user_message: str = pydantic.Field(
        ...,
        description="The latest user message. This is a duplicate of PipeInput.messages[0].content",
    )
    messages: List[PipeMessageInput] = pydantic.Field(
        ..., description="Also contains the latest user message."
    )
    model_id: str = pydantic.Field(
        ...,
        description="Typically this file's name, unless multiple models are supported.",
    )
    body: PipeBodyInput


# =================================================================== BDD

BDD = {
    "keywords": {
        "connexion-eduroam": "",
        "renvoi-dappel": "",
        "conference-a-3": "",
        "pas-d-internet": "",
        "deconnecte": "",
        "envoi-mail-echoue": "",
        "alerte-virus": "",
        "mot-de-passe-oublie": "",
        "smtp-error": "",
        "forward-etudiant": "",
        "aucun-reseau": "",
        "imap-timeout": "",
        "lecteur-reseau-inaccessible": "",
        "reception-bloquee": "",
        "parking-dappel": "",
        "montage-smb-echoue": "",
        "globalprotect": "",
        "vpn-ne-demarre-pas": "",
        "redirection-mail": "",
        "posture-vpn": "",
        "messagerie-vocale": "",
        "malware-detecte": "",
        "sftp-filezilla": "",
        "erreur-openvpn": "",
    },
    "documents": {
        "Changer mot de passe UTC : https://comptes.utc.fr/accounts-web/tools/changePassword.xhtml\nRecommandations : 8–16 caractères (majuscules, minuscules, chiffres, symboles), interdits : < > ; ` : & % * ? / \\ { } | ` ; pas de mot du dictionnaire, ni lien à votre identité, ne jamais communiquer ni stocker sur poste. Conseils : 8 caractères suffisent, mix maj/min(chiffres) dans l’ordre souhaité, mots générés aléatoirement ou mnémoniques (phrase→syllabes ou initiales). Ne pas envoyer par mail et consulter ANSSI : http://www.ssi.gouv.fr/IMG/pdf/NP_MDP_NoteTech.pdf\n\nChanger mot de passe session Windows : appuyer simultanément Ctrl+Alt+Suppr (voir https://5000.utc.fr/front/document.send.php?docid=845, 846, 847) puis suivre l’option «Changer de mot de passe». Changer mot de passe Windows pour imprimantes et partages réseau : Panneau de configuration → Gestionnaire d'identification → Windows Credentials → sélectionner chaque serveur (ex «lokorn») → Modifier → nouveau mot de passe → Enregistrer. Répéter pour chaque ressource.": [
            "mot-de-passe-oublie"
        ],
        "Messagerie standard (étudiants et personnels) : IMAP – Serveur imaps.utc.fr, SSL port 993 ; SMTP – smtps.utc.fr, SSL port 465 avec authentification. Clients : Outlook → doc https://5000.utc.fr/front/knowbaseitem.form.php?id=110 ; Thunderbird → https://5000.utc.fr/front/document.send.php?docid=826. Webmail via ENT : https://ent.utc.fr/ → roue centrale → Webmail.\n\nMessagerie Exchange (réservée personnels) : client spécialisé – contacter le 5000 ; Webmail Outlook via ENT : lien https://5000.utc.fr/front/knowbaseitem.form.php?id=60.": [
            "smtp-error",
            "imap-timeout",
            "envoi-mail-echoue",
            "reception-bloquee",
        ],
        "SFTP (FileZilla) : installer depuis https://filezilla-project.org/, config – Host : stargate.utc.fr, Username : votre login, Password : votre mot de passe, Port : 22. Dans FileZilla : Fichier → Gestionnaire de sites → Nouveau site → Hôte : stargate.utc.fr, Protocole : SFTP, Port : 22. Activer service ssh-users dans https://comptes.utc.fr/ (onglet Services) et ouvrir port 22.\n\nMontage lecteur réseau : Windows – Explorateur → Connecter un lecteur réseau → lettre + \\\\serveur\\partage, cocher «Se connecter avec des identifiants différents» et «Se reconnecter», Terminer → entrer login/mdp. MacOS – Finder → Aller → Se connecter au serveur (cmd+K) → smb://serveur/partage → identifiants (+ option trousseau). Session Etu : \\\\nasetu.utc\\~login, user login@AD, mot de passe CAS. Session UV : \\\\nasetu.utc\\~loginUV, user loginUV@AD, mot de passe TP. En cas de problème, contacter le 5000.": [
            "sftp-filezilla",
            "montage-smb-echoue",
            "lecteur-reseau-inaccessible",
        ],
        "UTC migre sa téléphonie classique vers IP (IP Touch 4028…) jusqu’en 2009, avec deux autocommutateurs interconnectés. Le téléphone IP se connecte en RJ45 sur le réseau IP UTC. «Tél sur IP» n’implique pas baisse immédiate des coûts externes mais facilite futures offres. Unification voix/mail et contrôle PC à venir.\n\nNuméros directs : 43xx, 44xx, 45xx, 46xx, 49xx, 52xx, 73xx, 79xx, 88xx (Escom) – accessibles ext. par 0+num. Non directs : 40xx–55xx (55xx Escom) – via standard 03.44.23.44.23 ou robot 03.44.23.46.99.": [],
        "Sécurité : code confidentiel par défaut 0000 (modif 72), verrouillage/déverrouillage (86, 86+code). Renvois : messagerie 76, immédiat 82, après 3 sonneries 87, occupé 88, inconditionnel 82, immédiat à distance 84, annulation 65. Parcage : mise en attente 64 + raccrocher, reprise 64.\n\nAppels/messagerie : consultation VM 77, opératrices 9, rappel occupé 5, rappel dernier interne 83, dernier appelant 60, fr. vocale *4. Organisation : modifier poste associé 25, rappel RDV 28, annulation 29, interception individuelle 67, groupe 68, débordement 80, annulation 81. Conférence 3 pers. : composer 1er → attente → composer 2e → touche Conf. Conf >3 : poste 2 : 69+code (ex 4567), basculer appel poste 1 : 69+4567. Postes IP : IP Touch 4018, 4028, 4038, 4068, 8028S, 8068S, 8088.": [
            "messagerie-vocale",
            "renvoi-dappel",
            "parking-dappel",
            "conference-a-3",
        ],
        "Télécharger configuration : étudiants client-etu-2045.ovpn (https://5000.utc.fr/front/document.send.php?docid=3777), personnel client-pers-2045.ovpn (https://5000.utc.fr/front/document.send.php?docid=3778). Windows : installer OpenVPN Connect v3 (<https://openvpn.net/downloads/openvpn-connect-v3-windows.msi>), lancer, onglet FILE → importer profil, entrer login UTC, Connect, accepter certificat, saisir mot de passe. Linux (Ubuntu) : Paramètres → Réseau → + VPN → Importer depuis fichier → sélectionner ovpn, login UTC, mot de passe «Demander à chaque fois», activer. MacOS : installer TunnelBlick (https://tunnelblick.net/downloads.html), glisser fichier ovpn sur icône TunnelBlick, Connect client-etu/pers via menu, saisir identifiants UTC.": [
            "erreur-openvpn",
            "vpn-ne-demarre-pas",
        ],
        "Agent GlobalProtect (PaloAlto) pour postes pro : accès VPN chiffré interne/extérieur selon identité, posture et ressources. Par défaut connecté en interne ; déconnecter pour usage perso.\n\nWindows : ouvrir https://portail.vpn.utc.fr, s’identifier ENT, télécharger et installer agent, config barre des tâches – Nom serveur : portail.vpn.utc.fr + identifiants. MacOS : voir guide https://secu.dsi.utc.fr/images/GlobalProtect_Mac.pdf. Linux : décompacter PanGPLinux-6.2.1-c15.tgz, ./gp_install, si erreur dpkg -i GlobalProtect_*.deb, se connecter à la passerelle puis déconnecter via gpctl/GUI.": [
            "globalprotect",
            "posture-vpn",
        ],
        "Réseau sécurisé (postes DSI ou admin local) impose 3 agents : OCS Inventory NG – inventaire, correctifs (DSI préinstallé ou télécharger https://secu.dsi.utc.fr/telechargements.php) ; Cortex XDR – antivirus/EDR PaloAlto, alertes centralisées, microSOC Orange Cyber (DSI ou téléchargement secu.dsi) ; GlobalProtect – VPN identité (déployé DSI via OCS, cf Doc6).\n\nRéseau standard (postes personnels) : maintenir OS à jour (Windows Update, iOS, Android, Linux) et antivirus actif (ex Windows Defender), installer uniquement logiciels d’éditeurs reconnus.": [
            "alerte-virus",
            "malware-detecte",
        ],
        "Wi-Fi UTC (BF,CR,PG,SI,Paris,Escom) : utcetu (login UTC, cycle ingé/master), utcpers (login UTC, personnel et 3e cycle), eduroam (login@utc.fr, mdp UTC; https://www.eduroam.fr/), portailutc (invités). Résidences : crousroberval (login UTC), crousextetu (login_crous@etu.crous-amiens.fr). Filaire : RJ45 après déclaration sur https://neptune.utc.fr/devices (UTC/ESCOM) ou formulaire CROUS (login@etu.crous-amiens.fr).\n\nDéclaration : UTC/ESCOM → https://neptune.utc.fr/devices/new ; CROUS externes – formulaire Neptune (prenom, nom, mail, tél, résid., bât., chambre, MAC). Dépannage : vérifier identifiants ENT, déclaration, désactiver MAC aléatoire (Win : Paramètres → Wi-Fi → Non; Android : type MAC réel; iOS : adresse privée Off), DNS auto, rebrancher câble pour filaire, contacter 5000@utc.fr en donnant MAC, date/heure, lieu, réseau.": [
            "connexion-eduroam",
            "pas-d-internet",
            "aucun-reseau",
            "deconnecte",
        ],
        "Le DNS résout noms de domaine en IP. Changer DNS Windows : Panneau de configuration → Connexions réseau (XP/Vista/7/8/10 via Centre Réseau) → Clic droit → Propriétés → Protocole Internet (TCP/IP) → Propriétés → cocher «Utiliser l’adresse de serveur DNS suivante» → renseigner serveurs DNS → OK. Référence Win10 : http://forums.cnetfrance.fr/forum/99-windows-10/, CNET : http://forums.cnetfrance.fr/topic/158796-changer-de-dns-manuellement/, https://5000.utc.fr/front/knowbaseitem.form.php?id=58.\n\nMacOS : Menu Pomme → Préférences Système → Réseau → sélectionner interface Ethernet ou Wi-Fi → Avancé → onglet DNS → + ajouter adresses (ex 8.8.8.8,8.8.4.4) → OK.": [
            "pas-d-internet",
            "aucun-reseau",
        ],
        "Rédirection mails Etu (étudiants uniquement) : se connecter sur https://comptes.utc.fr/ → onglet «Mail Redirection» → saisir adresse perso dans «Address to add» → Add → Submit → message «Changes applied successfully». Les mails arrivent sur votre boîte perso.": [
            "redirection-mail",
            "forward-etudiant",
        ],
        "Windows : PowerShell – Get-NetAdapter -Physical | Format-Table Name,MacAddress,InterfaceDescription ; relever MacAddress (Wi-Fi/Ethernet).\nMacOS : Préférences Système → Réseau → interface → Avancé → onglet Matériel → adresse MAC.\niOS : Réglages → Général → Informations → Adresse MAC Wi-Fi.\nAndroid : Paramètres → À propos → État → Adresse MAC Wi-Fi.\nLinux : terminal → ip -c l → relever link/ether de l’interface concernée.": [],
    },
}

# =================================================================== TICKET

# --- Literals pour les champs avec des choix limités ---

TypeDemande = Literal["incident", "demande"]
Site = Literal[
    "BF",
    "CR",
    "PG1",
    "PG2",
    "Site innovation",
    "Cima",
    "UTC-Paris",
    "Escom",
    "Crous/Résidences",
]
TypeReseau = Literal["Filaire", "Wifi"]

# --- Modèle Pydantic pour le Ticket Réseau ---


class TicketReseau(pydantic.BaseModel):
    """
    Modèle Pydantic représentant les informations nécessaires
    pour la création d'un ticket d'assistance réseau GLPI.
    """

    personne_concernee: str = pydantic.Field(
        ...,
        description="Nom/Prénom de la personne concernée (auto-complétion GLPI normalement), champ nécessaire, ne peut pas être inconnu.",
        examples=["Jean Dupont"],
    )
    objet: str = pydantic.Field(
        ...,
        description="Résumé de la demande en une ligne, champ nécessaire, ne peut pas être inconnu.",
        examples=["Problème connexion Wifi bureau B134"],
    )
    type_demande: TypeDemande = pydantic.Field(
        ...,
        description="Précise s'il s'agit d'un incident ou d'une demande, champ nécessaire, ne peut pas être inconnu.",
    )
    site: Site = pydantic.Field(
        ...,
        description="Site de l'UTC concerné par la demande, champ nécessaire, ne peut pas être inconnu.",
    )
    numero_bureau_salle: str = pydantic.Field(
        ...,
        description="Numéro du bureau ou de la salle concerné(e), champ nécessaire, ne peut pas être inconnu.",
        examples=["CR B134", "Résidence G - 101"],
    )
    departement: str = pydantic.Field(
        ...,
        description="Direction (et service) ou département concerné, champ nécessaire, ne peut pas être inconnu.",
        examples=["DSI/SAGP", "Génie Informatique"],
    )
    telephone: str = pydantic.Field(
        ...,
        description="Numéro de poste/téléphone de contact, champ nécessaire, ne peut pas être inconnu.",
        examples=["034423XXXX", "4567"],
    )
    materiel_declare: bool = pydantic.Field(
        ...,
        description="Le matériel est-il déclaré sur le réseau UTC (Neptune) ? Champ nécessaire, ne peut pas être inconnu",
    )
    # Ce champ est conditionnel
    type_reseau: Optional[TypeReseau] = pydantic.Field(
        default=None,
        description="Type de réseau (Filaire/Wifi) si le matériel est déclaré.",
    )
    description: str = pydantic.Field(
        ...,
        description="Description détaillée de la demande, champ nécessaire, ne peut pas être inconnu.",
        examples=[
            "Mon ordinateur portable déclaré sur Neptune ne parvient pas à se connecter au réseau Wifi eduroam dans mon bureau B134."
        ],
    )
    pieces_jointes: Optional[List[str]] = pydantic.Field(
        default=None,
        description="Liste des noms de fichiers joints (optionnel).",
        examples=[["screenshot_erreur.png", "config.txt"]],
    )


# =================================================================== COUNT TOKENS


def count_tokens(text: str) -> int:
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


# =================================================================== PIPELINE


class Pipeline:

    class Valves(pydantic.BaseModel):
        UTC_API_KEY: str = ""
        UTC_ENDPOINT: str = ""
        MODEL_NAME_CHAT: str = ""
        TOKEN_LIMIT_CHAT: str = ""
        MODEL_NAME_ANALYZE: str = ""

    def __init__(self):
        self.name = "Assistant Technique Expérimental"
        self.valves = self.Valves(
            **{key: os.getenv(key, "") for key in self.Valves.model_fields.keys()}
        )

    async def on_startup(self):
        await self.on_valves_updated()

    async def on_valves_updated(self):
        """
        Redefine the graph and tools using the updated values.
        """
        pass

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[PipeMessageInput],
        body: PipeBodyInput,
    ) -> Union[str, Generator, Iterator]:

        if any(value == "UNDEFINED" for value in self.valves.model_dump().values()):
            return "Please set ALL the valves before using this pipeline."

        try:
            # 0. Collecter les informations de la requête

            if len(messages) > 1:
                ai = messages[-2]  # last ai message
                user = messages[-1]  # last user message
                if ai["role"] == "assistant" and ai["content"].strip().startswith(
                    "[TICKET]"
                ):
                    if user["content"].lower() == "envoyer":
                        return "Le ticket a été envoyé!"
                    else:
                        messages.append(
                            {
                                "role": "system",
                                "content": "Explique à l'utilisateur que l'envoi du ticket a été annulé car il n'a pas dit le mot clé. Si il a toujours l'air de vouloir envoyer le ticket, tu peux lui proposer de réutiliser ton tool pour redemander confirmation.",
                            }
                        )

            # 1. Create a prompt object from the current messages

            CONVERSATION = []
            for msg in body["messages"]:

                match msg["role"]:
                    case "user":
                        CONVERSATION.append(HumanMessage(content=msg["content"]))
                    case "assistant":
                        CONVERSATION.append(AIMessage(content=msg["content"]))

            # 1.5. Verification des limites de tokens

            if count_tokens(" ".join([msg.content for msg in CONVERSATION])) > int(
                self.valves.TOKEN_LIMIT_CHAT
            ):
                return "Cette conversation dépasse la limite de tokens autorisée. Veuillez réduire le nombre de messages ou la taille des messages."

            # 2. Trouver des chunks pertinents

            class KW(pydantic.BaseModel):
                keywords: List[str]

            kw = (
                ChatOllama(
                    client_kwargs={  # Sera utilisé pour httpx.Client et httpx.AsyncClient
                        "headers": {
                            "Authorization": f"Bearer {self.valves.UTC_API_KEY}"
                        }
                    },
                    base_url=self.valves.UTC_ENDPOINT,
                    model=self.valves.MODEL_NAME_ANALYZE,
                    temperature=0.0,
                )
                .with_structured_output(KW)
                .invoke(
                    [
                        SystemMessage(
                            content="Sélectionne tous les mots clés qui correspondent à peu près à la situation de l'utilisateur d'après la conversation."
                            + "\nSi l'utilisateur n'est pas en train de parler d'un problème (par exemple il créé un ticket), ne choisit aucun mot clé."
                            + "\nSélectionne uniquement les mots clés parmi la liste suivante:"
                            + str(list(BDD["keywords"].keys()))
                        )
                    ]
                    + CONVERSATION
                )
            )
            chunks = set()
            for k in kw.keywords:
                chunks.update(
                    [doc for doc, keywords in BDD["documents"].items() if k in keywords]
                )

            # 3. Répondre en utilisant les chunks

            @tool
            def create_ticket(
                inp: TicketReseau,
            ):  # L'annotation de type 'inp: TicketReseau' utilisera la nouvelle définition
                """Uniquement si l'utilisateur a suivi effectué des actions de l'agent pour tenter de résoudre son problème et que cela a échoué, Crée un ticket de problème réseau et demande à l'utilisateur de confirmer son envoi."""
                return "Waiting for confirmation"

            PROMPT = CONVERSATION + [
                SystemMessage(
                    content=f"""
                    Vous êtes une interface pour la DSI (Service Informatique de l'Université de Technologie de Compiègne (UTC)).
                    
                    Discute avec l'utilisateur en respectant les règles suivantes :
                    #1. Tente toujours de repérer le problème de l'utilisateur en coopérant avec lui. Si le problème sort du cadre du Knowledge et du contexte de la DSI, recadre la conversation.
                    #2. Si les conditions d'apparition du problème ou le besoin de l'utilisateur a été clairement identifié dans les messages précédents, utilise uniquement les informations de la section "Knowledge" pour l'assister, si elles sont utiles. Si ces connaissances n'ont pas de rapport avec le problème de l'utilisateur et que le problème a été identifié avec précision, alors le problème sort du cadre de l'assistance par IA et tu ne peux pas l'aider.
                    #3. Lorsque tu es en train d'expliquer quelque chose à l'utilisateur, tu peux inclure des liens utiles http(s) dans ta réponse si et seulement si ils sont présents dans la section "Knwoledge".
                    #4. Ne suggère jamais de créer un ticket sauf si l'agent et l'utilisateurs ont échangé sur leur problème et que l'utilisateur a clairement exprimé avoir suivi les instructions de l'agent ou exprimé de la frustration quant à l'inefficacité de l'agent pour repérer son problème. Les tickets sont réservés aux conversations qui montrent un effort manifeste de l'utilisateur de coopérer et de clairement avoir échoué à résoudre son problème malgré avoir appliqué les étapes claires que l'agent lui a proposé avec confiance.
                    
                    Voici quelques sujets reçus usuellement par la DSI.\n\n1. Authentification et accès  \n   Mots-clés/phrases :  \n     • “mot de passe oublié” / “login refusé” / “identifiants invalides”  \n     • “accès refusé” / “permission denied”  \n     • “blocage compte” / “verrouillage session”  \n   Thématiques RAG suggérées :  \n     – Gestion des mots de passe (changement, recommandations ANSSI)  \n     – Déclaration et gestion des comptes UTC  \n     – Récupération et réinitialisation d’identifiants  \n\n2. Réseau et connectivité  \n   Mots-clés/phrases :  \n     • “pas d’internet” / “aucun réseau” / “déconnecté”  \n     • “Wi-Fi ne s’affiche pas” / “connexion eduroam”  \n     • “VPN ne démarre pas” / “erreur OpenVPN / GlobalProtect”  \n   Thématiques RAG suggérées :  \n     – VPN, Wi-Fi et filaire (profils, ports, SSID)  \n     – Dépannage réseau de base (DNS, MAC, DHCP)  \n     – Configuration manuelle de DNS  \n\n3. Partages et stockage  \n   Mots-clés/phrases :  \n     • “lecteur réseau inaccessible” / “montage SMB échoue”  \n     • “SFTP / FileZilla” / “téléversement impossible”  \n     • “droits écriture/lecture”  \n   Thématiques RAG suggérées :  \n     – Accès aux fichiers et lecteurs réseau (SMB, SFTP)  \n     – Activation du service SSH/SFTP  \n     – Gestionnaire d’identification Windows  \n\n4. Messagerie  \n   Mots-clés/phrases :  \n     • “envoi mail échoue” / “SMTP error”  \n     • “réception bloquée” / “IMAP timeout”  \n     • “redirection mail” / “forward étudiant”  \n   Thématiques RAG suggérées :  \n     – Configuration de la messagerie (IMAP/SMTP, Exchange)  \n     – Webmail via ENT  \n     – Redirection des mails étudiants  \n\n5. Imprimantes et périphériques  \n   Mots-clés/phrases :  \n     • “imprimante non trouvée” / “erreur spooler”  \n     • “connexion USB/ réseau”  \n     • “driver manquant”  \n   Thématiques RAG suggérées :  \n     – Mise à jour des mots de passe d’imprimante (Gestionnaire d’identification)  \n     – Installation et partage d’imprimantes sur Windows  \n     – Dépannage spooler  \n\n6. Performance et lenteur  \n   Mots-clés/phrases :  \n     • “ordinateur lent” / “démarrage trop long”  \n     • “applications réagissent mal”  \n     • “goulot d’étranglement réseau”  \n   Thématiques RAG suggérées :  \n     – Agents de sécurité et inventaire (OCS, Cortex XDR)  \n     – Analyse de charge réseau / débogage DNS  \n     – Vérification des services et mises à jour  \n\n7. Sécurité et antivirus  \n   Mots-clés/phrases :  \n     • “alerte virus” / “malware detecté”  \n     • “pare-feu bloque”  \n     • “posture VPN”  \n   Thématiques RAG suggérées :  \n     – Installation et configuration de Cortex XDR  \n     – GlobalProtect : posture et remontées  \n     – Bonnes pratiques de sécurité  \n\n8. Téléphonie et messagerie vocale  \n   Mots-clés/phrases :  \n     • “pas de tonalité” / “pas d’appel”  \n     • “renvoi d’appel” / “messagerie vocale”  \n     • “conférence à 3” / “parking d’appel”  \n   Thématiques RAG suggérées :  \n     – Guide Téléphonie IP (codes fonctions, conf call)  \n     – Numérotation internes/externe  \n     – Paramètres code de sécurité et messagerie  \n\nChaque fois qu’une plainte ou un mot-clé est détecté, le système RAG peut renvoyer :  \n • Le document ou la section précise à consulter  \n • Un diagnostic automatisé (checklist de vérifications)  \n • Des FAQ ou didacticiels associés  \n • Des liens vers les guides de l’ENT ou le portail 5000.
                    
                    Knowledge: \n{chunks}"""
                )
            ]

            LLM = ChatOllama(
                client_kwargs={  # Sera utilisé pour httpx.Client et httpx.AsyncClient
                    "headers": {"Authorization": f"Bearer {self.valves.UTC_API_KEY}"}
                },
                base_url=self.valves.UTC_ENDPOINT,
                model=self.valves.MODEL_NAME_CHAT,
                temperature=0.0,
            ).bind_tools([create_ticket])

            def stream():
                # TODO : Intégrer tiktoken ici
                it = LLM.stream(PROMPT)

                while True:

                    try:
                        chunk = next(it)
                    except StopIteration:
                        break
                    except Exception as e:
                        yield f"Erreur : {type(e)} {e}"
                        break

                    if chunk.tool_calls:

                        try:
                            # Tente de créer l'objet TicketReseau avec les arguments fournis par l'LLM
                            # La validation Pydantic s'exécute ici.
                            ticket_args = chunk.tool_calls[0]["args"]
                            ticket = TicketReseau(**ticket_args)
                        except pydantic.ValidationError as e:
                            yield f"Erreur lors de la création du ticket : {type(e)} {e}\n\n"
                            PROMPT.append(
                                SystemMessage(
                                    content=f"""
                                    Les informations fournies pour le ticket ne sont pas valides ou sont incomplètes.
                                    Erreur: {e.errors()}.
                                    Veuillez vérifier les informations et les redemander à l'utilisateur si nécessaire, en expliquant clairement ce qui manque ou ce qui est incorrect,
                                    puis relancez l'outil create_ticket avec les informations corrigées.
                                    Assurez-vous que tous les champs requis sont présents et que les valeurs respectent les formats attendus (par exemple, les choix pour 'type_demande', 'site', 'type_reseau').
                                    N'oubliez pas la logique conditionnelle : si 'materiel_declare' est 'oui', 'type_reseau' est obligatoire. Si 'materiel_declare' est 'non', 'type_reseau' doit être omis ou null.
                                    """
                                )
                            )
                            it = LLM.stream(
                                PROMPT
                            )  # Relance la génération avec la demande de correction
                            continue

                        yield f'[TICKET] {ticket.model_dump_json(indent=2, by_alias=True)}\n\nRépondez "envoyer" pour envoyer le ticket.'

                    if isinstance(chunk, AIMessageChunk):
                        yield chunk.content

            return stream()

        except Exception as e:
            return f"{type(e)} {e} : {__name__} User Message - {user_message}"
