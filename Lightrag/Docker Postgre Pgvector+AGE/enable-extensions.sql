-- Nom du fichier: enable-extensions.sql (sera copié en 02-enable-extensions.sql)
-- Ce script est exécuté automatiquement lors de l'initialisation de la base $POSTGRES_DB.

\set ON_ERROR_STOP on

DO $$
BEGIN
    RAISE NOTICE 'Activation des extensions AGE et pgvector si nécessaire...';

    -- Active l'extension Apache AGE
    CREATE EXTENSION IF NOT EXISTS age;

    -- Active l'extension pgvector
    CREATE EXTENSION IF NOT EXISTS vector;

    RAISE NOTICE 'Extensions AGE et pgvector activées (ou déjà existantes).';

    -- Optionnel: Définir le search_path par défaut pour cette base si souhaité
    -- ALTER DATABASE :"POSTGRES_DB" SET search_path = ag_catalog, "$user", public;
    -- RAISE NOTICE 'Search_path par défaut mis à jour pour la base de données.';

END $$;