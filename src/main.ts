import './style.css';
import { config } from './lib/config.ts';
import { manifestMismatches, parseIndex, searchDense, type Chunk, type Filters, type Index, type Manifest } from './lib/retrieval.ts';
import { $, renderExamples, renderHits, setStatus, showWarning } from './ui.ts';
import type { WorkerRequest, WorkerResponse } from './worker/embed.worker.ts';

const EXAMPLES = ['Why was the default patch size changed?', 'Why was ResEnc added?', 'Why did the data loading switch to blosc2?'];
const DATA = `${import.meta.env.BASE_URL}data/`;
const MB = (bytes: number) => (bytes / 1e6).toFixed(0);

// ---------- index ----------

async function loadIndex(): Promise<Index> {
  const get = async (name: string) => {
    const res = await fetch(DATA + name);
    if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`);
    return res;
  };
  const [manifest, chunks, vectors] = await Promise.all([
    get('manifest.json').then((r) => r.json() as Promise<Manifest>),
    get('meta.json').then((r) => r.json() as Promise<Chunk[]>),
    get('vectors.bin').then((r) => r.arrayBuffer()),
  ]);
  return parseIndex(manifest, chunks, vectors);
}

// ---------- embedding worker ----------

const worker = new Worker(new URL('./worker/embed.worker.ts', import.meta.url), { type: 'module' });
const pending = new Map<number, { resolve: (v: Float32Array) => void; reject: (e: Error) => void }>();
let nextId = 0;
let modelReady = false;
let modelError: string | null = null;

worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
  const msg = event.data;
  if (msg.type === 'progress') {
    // Also fires when the model comes from the browser cache, so don't claim it's a download.
    if (!modelReady && msg.total > 0) setStatus(`Loading the embedding model: ${MB(msg.loaded)} / ${MB(msg.total)} MB (downloaded once, then cached)`, msg.loaded / msg.total);
  } else if (msg.type === 'ready') {
    modelReady = true;
    if (!lastQuery && pending.size === 0) setStatus('Ready. Ask a question.');
  } else if (msg.type === 'vector') {
    pending.get(msg.id)?.resolve(msg.vector);
    pending.delete(msg.id);
  } else if (msg.id !== undefined) {
    pending.get(msg.id)?.reject(new Error(msg.message));
    pending.delete(msg.id);
  } else {
    modelError = msg.message;
    setStatus(`Could not load the embedding model: ${msg.message}`);
  }
};

function embed(text: string): Promise<Float32Array> {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ type: 'embed', id, text } satisfies WorkerRequest);
  });
}

// ---------- search ----------

let index: Index | null = null;
let lastQuery: { text: string; vector: Float32Array } | null = null;
let searchSeq = 0;

function readFilters(): Filters {
  const form = $<HTMLFieldSetElement>('filters');
  const value = (name: string) => (form.querySelector<HTMLInputElement>(`[name="${name}"]`)!.value || undefined);
  const types = [...form.querySelectorAll<HTMLInputElement>('[name="type"]:checked')].map((c) => c.value as Chunk['type']);
  return { types, from: value('from'), to: value('to'), file: value('file')?.trim() || undefined };
}

// Ranking a cached query vector is cheap, so filter changes re-rank without re-embedding.
function render() {
  if (!index || !lastQuery) return;
  const filters = readFilters();
  const t0 = performance.now();
  const hits = searchDense(index, lastQuery.vector, config.topK, filters);
  renderHits(hits, filters);
  const ms = (performance.now() - t0).toFixed(0);
  setStatus(hits.length ? `Top ${hits.length} of ${index.chunks.length} records (${ms} ms)` : 'No records match these filters.');
}

async function search(question: string) {
  question = question.trim();
  if (!question) return;
  const seq = ++searchSeq;
  if (!(await indexReady.catch(() => null))) return; // load failure is already shown
  const url = new URL(location.href);
  url.searchParams.set('q', question);
  history.replaceState(null, '', url);

  if (lastQuery?.text !== question) {
    if (modelError) return setStatus(`Search is unavailable: ${modelError}`);
    if (!modelReady) setStatus('Loading the embedding model. Your search will run as soon as it is ready.', 0);
    const vector = await embed(question).catch((err: Error) => {
      setStatus(`Search failed: ${err.message}`);
      return null;
    });
    if (!vector || seq !== searchSeq) return; // a newer search started meanwhile
    lastQuery = { text: question, vector };
  }
  render();
}

// ---------- wiring ----------

const input = $<HTMLInputElement>('q');
$('search').addEventListener('submit', (e) => {
  e.preventDefault();
  search(input.value);
});
$('filters').addEventListener('input', render);
renderExamples(EXAMPLES, (q) => {
  input.value = q;
  search(q);
});
document.addEventListener('keydown', (e) => {
  if (e.key === '/' && document.activeElement !== input && !(e.target instanceof HTMLInputElement)) {
    e.preventDefault();
    input.focus();
  }
});

const repoLink = $<HTMLAnchorElement>('repo-link');
repoLink.textContent = `${config.owner}/${config.repo}`;
repoLink.href = `https://github.com/${config.owner}/${config.repo}`;

setStatus('Loading the index…');
const indexReady = loadIndex();
indexReady.then(
  (loaded) => {
    index = loaded;
    const { manifest } = loaded;
    showWarning(manifestMismatches(manifest));
    $('footer').textContent = `Index: ${manifest.count} records from ${manifest.repo} @ ${manifest.head.slice(0, 7)}, built ${manifest.built_at.slice(0, 10)} with ${manifest.model} (${manifest.dtype}).`;
    if (!lastQuery) setStatus(modelReady ? 'Ready. Ask a question.' : 'Index loaded. Fetching the embedding model…');
    const initial = new URLSearchParams(location.search).get('q');
    if (initial) {
      input.value = initial;
      search(initial);
    }
  },
  (err) => setStatus(`Could not load the index: ${err.message}`),
);

if (import.meta.env.PROD && 'serviceWorker' in navigator) {
  navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`).catch(() => {
    // Offline support is optional; the app works without it.
  });
}
