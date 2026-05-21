-- pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Raw scraped documents (one row per source URL)
CREATE TABLE IF NOT EXISTS documents (
  id            SERIAL PRIMARY KEY,
  source_url    TEXT UNIQUE NOT NULL,
  title         TEXT,
  doc_type      TEXT NOT NULL,            -- 'n8n' | 'mcp'
  raw_content   TEXT NOT NULL,
  scraped_at    TIMESTAMP DEFAULT NOW()
);

-- Chunks with embeddings (one row per chunk, can have multiple strategies per doc)
CREATE TABLE IF NOT EXISTS chunks (
  id                 SERIAL PRIMARY KEY,
  document_id        INTEGER REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index        INTEGER NOT NULL,
  content            TEXT NOT NULL,
  embedding          vector(1536),
  chunking_strategy  TEXT NOT NULL,        -- 'fixed_size' | 'semantic' | 'markdown'
  metadata           JSONB DEFAULT '{}'::jsonb,
  created_at         TIMESTAMP DEFAULT NOW()
);

-- IVFFlat index for cosine similarity. Tune `lists` ~ sqrt(N) for production.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
  ON chunks USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- Supporting indexes
CREATE INDEX IF NOT EXISTS chunks_doc_idx       ON chunks(document_id);
CREATE INDEX IF NOT EXISTS chunks_strategy_idx  ON chunks(chunking_strategy);
CREATE INDEX IF NOT EXISTS documents_type_idx   ON documents(doc_type);
