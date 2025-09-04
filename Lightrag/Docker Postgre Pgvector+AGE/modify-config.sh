#!/bin/sh
# Nom original: modify-config.sh (sera copié en 01-add-pgvector-config.sh)
# S'assure que 'age' et 'vector' sont dans shared_preload_libraries.
set -e

CONFIG_FILE="${PGDATA}/postgresql.conf"
TARGET_LIBS="age,vector" # L'ordre importe peu, mais soyons cohérents

echo "Vérification/Mise à jour de shared_preload_libraries dans ${CONFIG_FILE}..."

# Vérifie si la ligne existe déjà
if grep -q "^shared_preload_libraries" "${CONFIG_FILE}"; then
    # Vérifie si 'vector' n'est PAS déjà dedans
    if ! grep "^shared_preload_libraries" "${CONFIG_FILE}" | grep -q "'vector'"; then
        echo "Ajout de 'vector' à shared_preload_libraries existant."
        # Ajoute ,vector avant la quote fermante. Gère 'age' -> 'age,vector', 'age, other' -> 'age, other,vector'
        sed -i "s/^\(shared_preload_libraries\s*=\s*'\)\([^']*\)\('.*\)/\1\2,vector\3/" "${CONFIG_FILE}"
        echo "shared_preload_libraries mis à jour."
    else
         # Vérifie si 'age' n'est PAS déjà dedans (au cas où l'image de base changerait)
         if ! grep "^shared_preload_libraries" "${CONFIG_FILE}" | grep -q "'age'"; then
            echo "Ajout de 'age' à shared_preload_libraries existant (cas inhabituel)."
            sed -i "s/^\(shared_preload_libraries\s*=\s*'\)\([^']*\)\('.*\)/\1age,\2\3/" "${CONFIG_FILE}"
            echo "shared_preload_libraries mis à jour."
         else
            echo "'age' et 'vector' semblent déjà présents dans shared_preload_libraries."
         fi
    fi
else
    # La ligne n'existe pas, on la crée avec les deux
    echo "Création de shared_preload_libraries = '${TARGET_LIBS}' dans ${CONFIG_FILE}."
    # Ajoute la ligne à la fin du fichier
    echo "shared_preload_libraries = '${TARGET_LIBS}'" >> "${CONFIG_FILE}"
fi

echo "Configuration de shared_preload_libraries terminée."