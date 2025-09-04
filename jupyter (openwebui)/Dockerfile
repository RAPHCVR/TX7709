# Remplacez par image de base choisie et son tag
FROM quay.io/jupyter/datascience-notebook:latest

# Il est souvent nécessaire de passer root pour installer des packages
USER root

# Installe jupyter-collaboration en utilisant mamba (ou pip si mamba/conda n'est pas dispo dans votre image)
# L'option -c conda-forge est souvent nécessaire pour ce package
# --yes évite les questions interactives
RUN mamba install -c conda-forge jupyter-collaboration --yes && \
    # Optionnel mais recommandé : nettoyer les caches pour réduire la taille
    mamba clean --all -f -y && \
    # Important : Rétablir les permissions pour l'utilisateur final (souvent jovyan)
    # Le nom $NB_USER et le chemin /home/$NB_USER sont typiques des images jupyter/*-notebook
    # Adaptez si votre image est différente
    fix-permissions /opt/conda && \
    fix-permissions /home/$NB_USER

# Revenir à l'utilisateur non-root par défaut de l'image
USER $NB_UID