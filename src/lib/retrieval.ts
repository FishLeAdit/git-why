// Index loading and search. Isomorphic: used by the browser app, the eval and the Node query script.
// Callers do the I/O (fetch or fs) and hand the raw contents in.
import MiniSearch from 'minisearch';
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

// ---------- keyword search (BM25 via MiniSearch) ----------

// Words that carry no signal in "why was the … changed?" questions. Without this list nearly every
// record matches "why", "was" and "the", and BM25 rewards whichever repeats them most.
const STOP_WORDS = new Set(
  'a an and are as at be been but by can could did do does for from had has have how i if in into is it its of on or should so than that the their them then there these they this to was we were what when where which who why will with would you'.split(' '),
);

export type KeywordIndex = { search: MiniSearch<Chunk>; byId: Map<string, Chunk> };

export function buildKeywordIndex(chunks: Chunk[]): KeywordIndex {
  const search = new MiniSearch<Chunk>({
    fields: ['title', 'text', 'files'],
    // File paths split on "/", "_" and "." too, so "default_experiment_planner.py" matches "planner".
    extractField: (chunk, field) => (field === 'files' ? chunk.files.join(' ') : chunk[field as 'title' | 'text']),
    processTerm: (term) => {
      const t = term.toLowerCase();
      return t.length < 2 || STOP_WORDS.has(t) ? null : t;
    },
    searchOptions: { boost: { title: 2 } },
  });
  search.addAll(chunks);
  return { search, byId: new Map(chunks.map((c) => [c.id, c])) };
}

export function searchKeyword(keyword: KeywordIndex, query: string, k: number, filters: Filters = {}): Hit[] {
  const hits: Hit[] = [];
  for (const result of keyword.search.search(query)) {
    const chunk = keyword.byId.get(result.id)!;
    if (!matchesFilters(chunk, filters)) continue;
    hits.push({ chunk, score: result.score });
    if (hits.length === k) break;
  }
  return hits;
}

// Reciprocal rank fusion (Cormack, Clarke & Buettcher 2009): each list contributes 1 / (K + rank).
// It needs only ranks, so cosine similarities and BM25 scores never have to be put on one scale.
export function fuseRRF(lists: Hit[][], k: number, K = 60): Hit[] {
  const fused = new Map<string, Hit>();
  for (const list of lists) {
    list.forEach(({ chunk }, rank) => {
      const prev = fused.get(chunk.id);
      fused.set(chunk.id, { chunk, score: (prev?.score ?? 0) + 1 / (K + rank + 1) });
    });
  }
  return [...fused.values()].sort((a, b) => b.score - a.score).slice(0, k);
}

// How deep each list goes before fusion; well past the 10 results shown.
const FUSION_DEPTH = 100;

export function searchHybrid(index: Index, keyword: KeywordIndex, query: string, vector: Float32Array, k: number, filters: Filters = {}): Hit[] {
  return fuseRRF([searchDense(index, vector, FUSION_DEPTH, filters), searchKeyword(keyword, query, FUSION_DEPTH, filters)], k);
}

// ---------- dense search ----------

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
