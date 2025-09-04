# Projet TX7709 : Ingénierie d'Agents IA et Déploiement MLOps

Ce dépôt contient l'ensemble du travail réalisé dans le cadre de l'unité d'enseignement TX7709 à l'Université de Technologie de Compiègne (UTC). Le projet couvre le cycle de vie complet de la création de services basés sur l'IA : de la conception d'agents conversationnels à leur déploiement sur une infrastructure cloud-native, en passant par leur évaluation rigoureuse.

## 🎯 Objectif du Projet

L'objectif principal était de développer des solutions d'IA pratiques pour répondre à des besoins spécifiques au sein de l'université, tout en maîtrisant les outils et les bonnes pratiques de l'ingénierie logicielle et du MLOps. Le projet s'articule autour de plusieurs composants clés, démontrant une approche complète et professionnelle.

## ✨ Caractéristiques Principales

*   **🤖 Agents Conversationnels Spécialisés** :
    *   **Assistant Technique pour la DSI** : Un agent RAG (Retrieval-Augmented Generation) conçu pour répondre aux questions fréquentes des étudiants et du personnel de l'UTC, capable de diagnostiquer des problèmes et de générer des tickets d'assistance au format GLPI.
    *   **Analyseur de Documents** : Un agent interactif capable d'ingérer plusieurs documents et de répondre à des questions complexes en extrayant et synthétisant l'information pertinente dans un format structuré.

*   **🚀 Plateforme de Déploiement Cloud-Native** :
    *   **Conteneurisation avec Docker** : Création d'images Docker optimisées pour les applications Python et pour une base de données PostgreSQL enrichie avec les extensions `pgvector` (recherche vectorielle) et `age` (graphes).
    *   **Orchestration avec Kubernetes & Helm** : Développement d'un chart Helm complet (`Lightrag`) pour un déploiement automatisé, scalable et reproductible de l'ensemble de la solution sur un cluster Kubernetes.
    *   **Déploiement d'un Écosystème IA** : Configuration et déploiement d'outils open-source de premier plan comme **OpenWebUI**, **n8n** et **Flowise** pour créer une plateforme d'expérimentation et d'interaction complète.

*   **📊 Framework d'Évaluation** :
    *   Mise en place d'un processus d'évaluation (`TXEvaluation`) pour mesurer la performance des agents RAG selon des métriques clés comme le **rappel** (pertinence des informations), le **délai** de réponse et le **volume** des données générées.

## 🛠️ Architecture & Technologies

Le projet s'appuie sur un écosystème d'outils modernes et performants :

*   **Langages & Frameworks** : Python 3.12+, Pydantic, LangChain.
*   **Modèles de Langage (LLM)** : Intégration flexible avec des modèles via des API (OpenAI, UTC) et des serveurs locaux (Ollama).
*   **Base de Données** : PostgreSQL 16 avec `pgvector` pour la recherche sémantique et `Apache AGE` pour les structures de graphes.
*   **Conteneurisation & Orchestration** : Docker, Kubernetes (K8s), Helm.
*   **Évaluation & Data Science** : Jupyter, Pandas, Matplotlib, Seaborn.
*   **Dépendances Python Clés** : `langchain-openai`, `langchain-ollama`, `tiktoken`, `psycopg[binary]`.

## 📂 Structure du Dépôt

-   **/AgentsTX/** : Contient les pipelines et la logique principale des agents conversationnels (`document_analyzer5.py`, `rag_test4.py`).
-   **/Lightrag/** : Le cœur de la plateforme de déploiement, incluant le `Dockerfile` de l'application, le `Dockerfile` de l'image PostgreSQL custom, et le chart Helm complet dans `/chart`.
-   **/TXRAG/** : Scripts et notebooks pour la création de la base de connaissances (index RAG) à partir de documents bruts.
-   **/TXEvaluation/** : Le framework d'évaluation, avec le notebook d'analyse (`eval.ipynb`) et les données de référence (`truth.json`).
-   **/k8s (helm)/** : Fichiers `values.yaml` pour le déploiement d'outils tiers (Flowise, n8n, OpenWebUI) sur Kubernetes.
-   **/pipelines/** : `Dockerfile` pour étendre l'image `open-webui/pipelines` avec des bibliothèques Python personnalisées.

## 🚀 Démarrage Rapide (Déploiement)

Le déploiement de la solution `Lightrag` se fait via Helm.

**Prérequis :**
*   Un cluster Kubernetes (ex: minikube, k3s, ou un service cloud).
*   `kubectl` et `helm` installés et configurés.
*   Un registre d'images Docker (ex: Docker Hub, GHCR, DigitalOcean Container Registry).

**Étapes :**
1.  **Construire et Pousser les Images Docker :**
    ```bash
    # Image de l'application LightRAG
    docker build -t VOTRE_REGISTRY/lightrag:latest ./Lightrag
    docker push VOTRE_REGISTRY/lightrag:latest

    # Image PostgreSQL custom
    docker build -t VOTRE_REGISTRY/postgres-age-vector:latest ./Lightrag/Docker\ Postgre\ Pgvector+AGE
    docker push VOTRE_REGISTRY/postgres-age-vector:latest
    ```

2.  **Configurer le Chart Helm :**
    Modifiez le fichier `Lightrag/chart/values.yaml` pour pointer vers vos images Docker et ajuster la configuration (domaine, secrets, etc.).

3.  **Déployer avec Helm :**
    ```bash
    helm install release-name ./Lightrag/chart/
    ```

## Auteur

*   **RAPHCVR**
*   **Ryustel**
