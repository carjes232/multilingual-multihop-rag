-- HNSW index for fast cosine similarity search on pgvector
-- Usage:
--   psql -h localhost -p 5432 -U rag -d ragdb -f scripts/05_create_vector_index.sql

-- Ensure extension exists (idempotent)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create HNSW index on the chunks.embedding column using cosine distance
-- Notes:
-- - 'vector_cosine_ops' tells pgvector to use cosine distance (<=>) for this index
-- - Adjust m and ef_construction for recall/speed tradeoffs
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
ON chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- Optional: set default ef_search for this index (higher = better recall, slower)
-- ALTER INDEX idx_chunks_embedding_hnsw SET (ef_search = 64);

-- Update planner stats
ANALYZE chunks;
