// Embeds queries off the main thread. Uses the same loadEmbedder() as build-index, so browser query
// vectors are comparable with the index (default browser device is wasm, same q8 model file).
import { embedQuery, loadEmbedder } from '../lib/embed.ts';

export type WorkerRequest = { type: 'embed'; id: number; text: string };
export type WorkerResponse =
  | { type: 'progress'; loaded: number; total: number }
  | { type: 'ready' }
  | { type: 'vector'; id: number; vector: Float32Array }
  | { type: 'error'; id?: number; message: string };

// The tsconfig uses DOM types; a Worker has the same postMessage/onmessage shape as the worker scope.
const scope = self as unknown as Worker;
const post = (message: WorkerResponse, transfer: Transferable[] = []) => scope.postMessage(message, transfer);

const embedder = loadEmbedder((p) => {
  if (p.status === 'progress_total') post({ type: 'progress', loaded: p.loaded, total: p.total });
});
embedder.then(
  () => post({ type: 'ready' }),
  (err) => post({ type: 'error', message: `model failed to load: ${err}` }),
);

scope.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const { id, text } = event.data;
  try {
    const vector = await embedQuery(await embedder, text);
    post({ type: 'vector', id, vector }, [vector.buffer]);
  } catch (err) {
    post({ type: 'error', id, message: String(err) });
  }
};
