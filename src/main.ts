import './style.css';
import { config } from './lib/config.ts';
import { manifestMismatches, parseIndex, searchDense, type Chunk, type Filters, type Index, type Manifest } from './lib/retrieval.ts';
import { $, renderExamples, renderHits, setStatus, showWarning } from './ui.ts';
import type { WorkerRequest, WorkerResponse } from './worker/embed.worker.ts';

// One entry per built index, written by build-index (public/data/repos.json).
type RepoEntry = { key: string; name: string; since: string | null; count: number; head: string; built_at: string; bytes: number; examples: string[] };

const DATA = `${import.meta.env.BASE_URL}data/`;
const MB = (bytes: number) => (bytes / 1e6).toFixed(0);

async function get(path: string): Promise<Response> {
  const res = await fetch(DATA + path);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res;
}

// ---------- indexes (loaded lazily, one per repo) ----------

const indexes = new Map<string, Promise<Index>>();

function loadIndex(key: string): Promise<Index> {
  let pending = indexes.get(key);
  if (!pending) {
    pending = Promise.all([
      get(`${key}/manifest.json`).then((r) => r.json() as Promise<Manifest>),
      get(`${key}/meta.json`).then((r) => r.json() as Promise<Chunk[]>),
      get(`${key}/vectors.bin`).then((r) => r.arrayBuffer()),
    ]).then(([manifest, chunks, vectors]) => parseIndex(manifest, chunks, vectors));
    // A failed load (e.g. offline before this repo was ever opened) may be retried later.
    pending.catch(() => indexes.delete(key));
    indexes.set(key, pending);
  }
  return pending;
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
    if (!lastQuery && pending.size === 0 && index) setStatus('Ready. Ask a question.');
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

// ---------- state ----------

let repos: RepoEntry[] = [];
let current: RepoEntry | null = null;
let index: Index | null = null; // the current repo's index, once loaded
let lastQuery: { text: string; vector: Float32Array } | null = null;
let searchSeq = 0;
let repoSeq = 0;

function updateUrl() {
  const url = new URL(location.href);
  if (current) url.searchParams.set('repo', current.name);
  if (lastQuery) url.searchParams.set('q', lastQuery.text);
  else url.searchParams.delete('q');
  history.replaceState(null, '', url);
}

// ---------- search ----------

function readFilters(): Filters {
  const form = $<HTMLFieldSetElement>('filters');
  const value = (name: string) => (form.querySelector<HTMLInputElement>(`[name="${name}"]`)!.value || undefined);
  const types = [...form.querySelectorAll<HTMLInputElement>('[name="type"]:checked')].map((c) => c.value as Chunk['type']);
  return { types, from: value('from'), to: value('to'), file: value('file')?.trim() || undefined };
}

// Ranking a cached query vector is cheap, so filter and repo changes re-rank without re-embedding
// (every repo is embedded with the same model, so one question vector works for all of them).
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
  if (!question || !current) return;
  const seq = ++searchSeq;
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
  updateUrl();
  render(); // no-op until the current repo's index is loaded; selectRepo renders once it is
}

// ---------- repos ----------

async function selectRepo(repo: RepoEntry, notice?: string) {
  const seq = ++repoSeq;
  current = repo;
  index = null;
  updateUrl();
  $<HTMLSelectElement>('repo').value = repo.key;
  const link = $<HTMLAnchorElement>('repo-link');
  link.href = `https://github.com/${repo.name}`;
  renderExamples(repo.examples, (q) => {
    input.value = q;
    search(q);
  });
  $('results').replaceChildren();
  showWarning([]);
  $('footer').textContent = '';
  setStatus(`Loading the ${repo.name} index (${(repo.bytes / 1e6).toFixed(1)} MB)…`);

  let loaded: Index;
  try {
    loaded = await loadIndex(repo.key);
  } catch (err) {
    if (seq === repoSeq) setStatus(`Could not load the ${repo.name} index: ${(err as Error).message}`);
    return;
  }
  if (seq !== repoSeq) return; // the user picked another repo meanwhile
  index = loaded;
  const { manifest } = loaded;
  showWarning(manifestMismatches(manifest));
  const since = repo.since ? ` since ${repo.since.slice(0, 10)}` : '';
  $('footer').textContent = `Index: ${manifest.count} records from ${manifest.repo}${since} @ ${manifest.head.slice(0, 7)}, built ${manifest.built_at.slice(0, 10)} with ${manifest.model} (${manifest.dtype}).`;
  if (lastQuery) render();
  else if (pending.size === 0) setStatus(notice ?? (modelReady ? 'Ready. Ask a question.' : 'Index loaded. Fetching the embedding model…'));
}

// ---------- wiring ----------

const input = $<HTMLInputElement>('q');
$('search').addEventListener('submit', (e) => {
  e.preventDefault();
  search(input.value);
});
$('filters').addEventListener('input', render);
$('repo').addEventListener('change', (e) => {
  const repo = repos.find((r) => r.key === (e.target as HTMLSelectElement).value);
  if (repo) selectRepo(repo);
});
document.addEventListener('keydown', (e) => {
  if (e.key === '/' && document.activeElement !== input && !(e.target instanceof HTMLInputElement)) {
    e.preventDefault();
    input.focus();
  }
});

setStatus('Loading…');
get('repos.json')
  .then((r) => r.json() as Promise<RepoEntry[]>)
  .then((list) => {
    if (list.length === 0) throw new Error('no indexes have been built');
    repos = list;
    $<HTMLSelectElement>('repo').replaceChildren(
      ...repos.map((r) => {
        const option = document.createElement('option');
        option.value = r.key;
        option.textContent = r.name;
        return option;
      }),
    );
    const params = new URLSearchParams(location.search);
    const wanted = params.get('repo')?.toLowerCase();
    const initial = repos.find((r) => r.name.toLowerCase() === wanted) ?? repos[0];
    const unknown = wanted && initial.name.toLowerCase() !== wanted;
    selectRepo(initial, unknown ? `"${params.get('repo')}" isn't indexed here; showing ${initial.name}.` : undefined);
    const q = params.get('q');
    if (q) {
      input.value = q;
      search(q);
    }
  })
  .catch((err: Error) => setStatus(`Could not load the list of indexes: ${err.message}`));

if (import.meta.env.PROD && 'serviceWorker' in navigator) {
  navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`).catch(() => {
    // Offline support is optional; the app works without it.
  });
}
