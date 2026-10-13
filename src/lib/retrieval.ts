// Index loading and search. Isomorphic: used by the browser app, the eval and the Node query script.
// Callers do the I/O (fetch or fs) and hand the raw contents in.

export type Chunk = {
  id: string; // "pr-123" | "issue-45" | "commit-ab12cd3"
  type: 'pr' | 'issue' | 'commit';
  title: string;
  date: string; // ISO
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

export function parseIndex(manifest: Manifest, chunks: Chunk[], vectorBytes: ArrayBuffer): Index {
  const vectors = new Float32Array(vectorBytes);
  if (chunks.length !== manifest.count || vectors.length !== manifest.count * manifest.dim) {
    throw new Error(`index mismatch: manifest says ${manifest.count} x ${manifest.dim}, got ${chunks.length} chunks and ${vectors.length} floats`);
  }
  return { manifest, chunks, vectors };
}

// Vectors are L2-normalized, so cosine similarity is a dot product.
export function searchDense(index: Index, query: Float32Array, k: number): Hit[] {
  const { vectors, chunks, manifest } = index;
  const dim = manifest.dim;
  const scores = new Float32Array(chunks.length);
  for (let i = 0; i < chunks.length; i++) {
    let dot = 0;
    for (let d = 0, o = i * dim; d < dim; d++) dot += vectors[o + d] * query[d];
    scores[i] = dot;
  }
  return topK(scores, k).map((i) => ({ chunk: chunks[i], score: scores[i] }));
}

function topK(scores: Float32Array, k: number): number[] {
  return Array.from(scores.keys())
    .sort((a, b) => scores[b] - scores[a])
    .slice(0, k);
}
