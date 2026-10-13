// Embedding, shared by build-index (Node), the browser worker and the eval, so all three use
// exactly the same model, dtype, pooling and normalization.
import { pipeline, type ProgressCallback } from '@huggingface/transformers';
import { EMBED_DIM, EMBED_DTYPE, EMBED_MODEL, EMBED_POOLING, QUERY_PREFIX } from './config.ts';

export type Embedder = (texts: string[]) => Promise<Float32Array[]>;

export async function loadEmbedder(progress_callback?: ProgressCallback): Promise<Embedder> {
  const extractor = await pipeline('feature-extraction', EMBED_MODEL, { dtype: EMBED_DTYPE, progress_callback });
  return async (texts) => {
    const output = await extractor(texts, { pooling: EMBED_POOLING, normalize: true });
    const data = output.data as Float32Array;
    return texts.map((_, i) => data.slice(i * EMBED_DIM, (i + 1) * EMBED_DIM));
  };
}

export async function embedQuery(embed: Embedder, question: string): Promise<Float32Array> {
  const [vector] = await embed([QUERY_PREFIX + question]);
  return vector;
}
