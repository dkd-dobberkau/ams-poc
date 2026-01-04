-- Agent Memory System - Database Schema
-- PostgreSQL 16 with pgvector extension

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Main memories table
CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    memory_type VARCHAR(50) NOT NULL CHECK (memory_type IN ('fact', 'opinion', 'experience', 'observation')),
    content TEXT NOT NULL,
    content_embedding VECTOR(1536),
    confidence DECIMAL(3,2) DEFAULT 0.70,
    evidence_count INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT valid_confidence CHECK (confidence >= 0 AND confidence <= 1),
    CONSTRAINT valid_evidence_count CHECK (evidence_count >= 1)
);

-- Index for user lookups with type filter
CREATE INDEX IF NOT EXISTS idx_memories_user_type
ON memories(user_id, memory_type);

-- Index for user lookups ordered by recency
CREATE INDEX IF NOT EXISTS idx_memories_user_updated
ON memories(user_id, updated_at DESC);

-- Vector similarity search index (IVFFlat for approximate search)
-- Note: For production with >1M vectors, consider HNSW index instead
CREATE INDEX IF NOT EXISTS idx_memories_embedding
ON memories USING ivfflat (content_embedding vector_cosine_ops)
WITH (lists = 100);

-- Full-text search index for German language
CREATE INDEX IF NOT EXISTS idx_memories_fulltext
ON memories USING gin(to_tsvector('german', content));

-- Function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to auto-update updated_at
DROP TRIGGER IF EXISTS update_memories_updated_at ON memories;
CREATE TRIGGER update_memories_updated_at
    BEFORE UPDATE ON memories
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Optional: Create a view for opinions with formatted confidence
CREATE OR REPLACE VIEW opinions_with_confidence AS
SELECT
    id,
    user_id,
    content,
    confidence,
    evidence_count,
    created_at,
    updated_at,
    CASE
        WHEN confidence >= 0.9 THEN 'sehr sicher'
        WHEN confidence >= 0.7 THEN 'sicher'
        WHEN confidence >= 0.5 THEN 'unsicher'
        ELSE 'sehr unsicher'
    END AS confidence_label
FROM memories
WHERE memory_type = 'opinion'
ORDER BY updated_at DESC;

-- Conversations table for grouping chat messages
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    title VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fetching user's conversations
CREATE INDEX IF NOT EXISTS idx_conversations_user
ON conversations(user_id, updated_at DESC);

-- Trigger to auto-update conversations.updated_at
DROP TRIGGER IF EXISTS update_conversations_updated_at ON conversations;
CREATE TRIGGER update_conversations_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Chat messages table for conversation history
CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    memories_used JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fetching conversation messages
CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation
ON chat_messages(conversation_id, created_at ASC);

-- Index for fetching user's chat history
CREATE INDEX IF NOT EXISTS idx_chat_messages_user
ON chat_messages(user_id, created_at DESC);

-- Grant permissions (adjust role name as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ams_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ams_user;
