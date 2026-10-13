// Index loading and search. Isomorphic: used by the browser app, the eval and the Node query script.
// Callers do the I/O (fetch or fs) and hand the raw contents in.
import { EMBED_DIM, EMBED_DTYPE, EMBED_MODEL, EMBED_POOLING } from './config.ts';

export type Chunk = {
  id: string; // "pr-123" | "issue-45" | "commit-ab12cd3"
  type: 'pr' | 'issue' | 'commit';
  title: string;
  date: string; // ISO, UTC
  author: string;
  url: string;
  state?: 'merged' | 'closed' | 'open';
  files: string[];
  text: string;
};

export type Manifest = {
  model: string;
  dtype: string;
  pooling: string;
  dim: number;
  count: number;
  repo: string;
  head: string;
  built_at: string;
};

export type Index = { manifest: Manifest; chunks: Chunk[]; vectors: Float32Array };

export type Hit = { chunk: Chunk; score: number };

export type Filters = {
  types?: Chunk['type'][]; // empty or missing = all types
  from?: string; // YYYY-MM-DD, inclusive
  to?: string; // YYYY-MM-DD, inclusive
  file?: string; // case-insensitive substring of any file path
};

export function parseIndex(manifest: Manifest, chunks: Chunk[], vectorBytes: ArrayBuffer): Index {
  const vectors = new Float32Array(vectorBytes);
  if (chunks.length !== manifest.count || vectors.length !== manifest.count * manifest.dim) {
    throw new Error(`index mismatch: manifest says ${manifest.count} x ${manifest.dim}, got ${chunks.length} chunks and ${vectors.length} floats`);
  }
  return { manifest, chunks, vectors };
}

// Query vectors are only comparable with the index if both came from the same embedding setup.
export function manifestMismatches(manifest: Manifest): string[] {
  const expected = { model: EMBED_MODEL, dtype: EMBED_DTYPE, pooling: EMBED_POOLING, dim: EMBED_DIM };
  return Object.entries(expected)
    .filter(([key, value]) => manifest[key as keyof typeof expected] !== value)
    .map(([key, value]) => `${key}: index has ${manifest[key as keyof typeof expected]}, app expects ${value}`);
}

export function matchesFilters(chunk: Chunk, filters: Filters): boolean {
  if (filters.types?.length && !filters.types.includes(chunk.type)) return false;
  // Dates are UTC ISO strings, so comparing the YYYY-MM-DD prefix is enough.
  const day = chunk.date.slice(0, 10);
  if (filters.from && day < filters.from) return false;
  if (filters.to && day > filters.to) return false;
  if (filters.file) {
    const needle = filters.file.toLowerCase();
    if (!chunk.files.some((f) => f.toLowerCase().includes(needle))) return false;
  }
  return true;
}

// Vectors are L2-normalized, so cosine similarity is a dot product.
export function searchDense(index: Index, query: Float32Array, k: number, filters: Filters = {}): Hit[] {
  const { vectors, chunks, manifest } = index;
  const dim = manifest.dim;
  const hits: Hit[] = [];
  for (let i = 0; i < chunks.length; i++) {
    if (!matchesFilters(chunks[i], filters)) continue;
    let dot = 0;
    for (let d = 0, o = i * dim; d < dim; d++) dot += vectors[o + d] * query[d];
    hits.push({ chunk: chunks[i], score: dot });
  }
  return hits.sort((a, b) => b.score - a.score).slice(0, k);
}
