-- Priority queue of AALCs
CREATE TABLE IF NOT EXISTS aalc_queue (
  id UUID PRIMARY KEY,
  task_spec JSONB NOT NULL,
  ifv_spec JSONB,
  priority INT DEFAULT 100,
  status TEXT CHECK (status IN ('queued','leased','done','error')) DEFAULT 'queued',
  leased_until TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Repositories & archive
CREATE TABLE IF NOT EXISTS repo_rml (
  id UUID PRIMARY KEY,
  name TEXT,
  arch JSONB,
  weights BYTEA,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS repo_rfv (
  id UUID PRIMARY KEY,
  rml_id UUID REFERENCES repo_rml(id),
  rdata_uri TEXT,
  labels JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS aalc_snapshot (
  id UUID PRIMARY KEY,
  aalc_id UUID,
  meta JSONB,
  weights_uri TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Simple DS request/response log for RAG/SQL
CREATE TABLE IF NOT EXISTS ds_query_log (
  id BIGSERIAL PRIMARY KEY,
  aalc_id UUID,
  query TEXT,
  sql_generated TEXT,
  result_uri TEXT,
  entropy DOUBLE PRECISION,
  created_at TIMESTAMPTZ DEFAULT now()
);