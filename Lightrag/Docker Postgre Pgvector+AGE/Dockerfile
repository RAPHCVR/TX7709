# ----------- Stage 1: Builder -----------
# Utiliser une image contenant les outils de build et PG16 dev.
# On pourrait utiliser debian + installer tout, mais partir de l'image age
# elle-même simplifie l'accès à postgresql-server-dev-16 correspondant.
ARG AGE_BASE_IMAGE=apache/age:latest
# Note: 'latest' peut changer. Si un tag spécifique pour PG16 existe (ex: apache/age:pg16-vX.Y.Z), utilisez-le.
FROM ${AGE_BASE_IMAGE} AS builder

ARG PGVECTOR_VERSION=v0.8.0

# Passer root pour installer les dépendances
USER root

# Installer les dépendances de build strictement nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    # postgresql-server-dev-16 est déjà présent ou compatible dans l'image apache/age
    git \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Cloner et compiler pgvector
WORKDIR /tmp
RUN git clone --branch ${PGVECTOR_VERSION} --depth 1 https://github.com/pgvector/pgvector.git \
 && cd pgvector \
 && make \
 # Important : 'make install' place les fichiers aux bons endroits DANS CE STAGE
 && make install

# Les fichiers compilés (.so, .sql, .control) sont maintenant dans /usr/lib/postgresql/16/lib/
# et /usr/share/postgresql/16/extension/ de ce stage 'builder'.

# ----------- Stage 2: Final Image -----------
# Repartir de la même image de base pour avoir AGE pré-configuré
FROM ${AGE_BASE_IMAGE}

# Métadonnées
LABEL maintainer="Votre Nom <votre.email@example.com>"
LABEL description="PostgreSQL 16 with Apache AGE and pgvector v0.8.0 extensions (auto-enabled, multi-stage build)"

# Copier UNIQUEMENT les artefacts pgvector compilés depuis le stage 'builder'
# On a besoin des droits root pour écrire dans /usr/lib et /usr/share
USER root
COPY --from=builder /usr/lib/postgresql/16/lib/vector.so /usr/lib/postgresql/16/lib/vector.so
COPY --from=builder /usr/share/postgresql/16/extension/vector* /usr/share/postgresql/16/extension/

# Copier les scripts d'initialisation qui s'exécuteront au premier démarrage
# Ces scripts s'exécutent en tant qu'utilisateur 'postgres' par le point d'entrée Docker
COPY modify-config.sh /docker-entrypoint-initdb.d/01-add-pgvector-config.sh
COPY enable-extensions.sql /docker-entrypoint-initdb.d/02-enable-extensions.sql

# Assurer les permissions d'exécution pour le script shell
RUN chmod +x /docker-entrypoint-initdb.d/01-add-pgvector-config.sh \
 # Optionnel mais propre : s'assurer que les permissions sont correctes pour postgres
 && chown postgres:postgres /docker-entrypoint-initdb.d/01-add-pgvector-config.sh \
 && chown postgres:postgres /docker-entrypoint-initdb.d/02-enable-extensions.sql

# Revenir à l'utilisateur postgres standard pour l'exécution
USER postgres

# Exposer le port (probablement hérité, mais explicite)
EXPOSE 5432

# Le CMD/ENTRYPOINT est hérité de l'image apache/age
# Il lancera postgres et exécutera les scripts dans /docker-entrypoint-initdb.d/