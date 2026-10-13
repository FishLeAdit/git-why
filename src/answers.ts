// Written answers from a small model that runs on the visitor's GPU (WebLLM), grounded in the top records.
// Opt-in: nothing large downloads until the visitor turns answers on. Search never depends on this module.
import type { WebWorkerMLCEngine } from '@mlc-ai/web-llm';
import { config } from './lib/config.ts';
import { buildMessages, checkAnswer, CITATION_GROUP, idsInGroup } from './lib/prompt.ts';
import type { Hit } from './lib/retrieval.ts';
import { $, el } from './ui.ts';

type Phase = 'checking' | 'unsupported' | 'off' | 'loading' | 'ready' | 'answering' | 'done' | 'gated' | 'stale' | 'error';
type Context = { question: string; hits: Hit[]; threshold: number | null };

const ENABLED_KEY = 'git-why:answers';
const { llm } = config;

let phase: Phase = 'checking';
let model = llm.model;
let unsupportedReason = '';
let gpuLabel = ''; // e.g. "nvidia ampere"
let integratedGpu = false;
let engine: WebWorkerMLCEngine | null = null;
let worker: Worker | null = null;
let cached = false; // the model's files are in this browser's cache
let progress = { fraction: 0, text: '' };
let errorText = '';
let context: Context | null = null;
let answerText = '';
let stopped = false;
let wantAnswer = false; // a question arrived while the model was still loading
let generation = 0; // bumps on every new answer or stop, so a stale stream stops writing
let running: Promise<void> | null = null;

const storage = {
  get: () => {
    try {
      return localStorage.getItem(ENABLED_KEY) === 'on';
    } catch {
      return false;
    }
  },
  set: (on: boolean) => {
    try {
      if (on) localStorage.setItem(ENABLED_KEY, 'on');
      else localStorage.removeItem(ENABLED_KEY);
    } catch {
      // private mode etc.: answers simply won't auto-enable next time
    }
  },
};

// ---------- WebGPU ----------

type GPUAdapterLike = { features: { has(name: string): boolean }; info?: { vendor?: string; architecture?: string } };
type GPULike = { requestAdapter(options?: { powerPreference?: string }): Promise<GPUAdapterLike | null> };

async function detect(): Promise<void> {
  const gpu = (navigator as Navigator & { gpu?: GPULike }).gpu;
  if (!gpu) {
    unsupportedReason = 'This browser has no WebGPU, which the answer model needs (desktop Chrome and Edge have it).';
    return;
  }
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' }).catch(() => null);
  if (!adapter) {
    unsupportedReason = 'WebGPU is there, but no usable GPU was found.';
    return;
  }
  // q4f16 needs the shader-f16 feature; the q4f32 build of the same model works everywhere.
  model = adapter.features.has('shader-f16') ? llm.model : llm.modelWithoutF16;
  gpuLabel = [adapter.info?.vendor, adapter.info?.architecture].filter(Boolean).join(' ');
  // Browsers on Windows ignore powerPreference, so a laptop with two GPUs usually gets the slower
  // integrated one unless the browser is set to "High performance" in Windows graphics settings.
  integratedGpu = adapter.info?.vendor === 'intel';
}

// ---------- model lifecycle ----------

// Hugging Face rate-limits bursts of downloads (HTTP 429), and WebLLM fetches the model's shards in
// parallel, so a first download can fail part-way. Shards already fetched stay cached, so waiting and
// retrying resumes where it stopped. Other failures (no GPU memory, device lost) are not retried.
const RETRY_WAITS_S = [10, 30, 60];
const isNetworkError = (err: unknown) => /request failed|failed to fetch|networkerror|429|network/i.test(String((err as Error)?.message ?? err));
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function loadModel() {
  if (engine || phase === 'loading') return;
  phase = 'loading';
  progress = { fraction: 0, text: 'Starting…' };
  render();
  const webllm = await import('@mlc-ai/web-llm');
  for (let attempt = 0; ; attempt++) {
    try {
      worker = new Worker(new URL('./worker/llm.worker.ts', import.meta.url), { type: 'module' });
      engine = await webllm.CreateWebWorkerMLCEngine(worker, model, {
        initProgressCallback: (report) => {
          progress = { fraction: report.progress, text: report.text };
          if (phase === 'loading') render();
        },
      });
      break;
    } catch (err) {
      worker?.terminate();
      worker = null;
      engine = null;
      const wait = RETRY_WAITS_S[attempt];
      if (!isNetworkError(err) || wait === undefined) {
        errorText = `The answer model could not be loaded: ${(err as Error).message ?? err}`;
        phase = 'error';
        render();
        return;
      }
      for (let s = wait; s > 0; s--) {
        progress = { fraction: progress.fraction, text: `The download was interrupted (Hugging Face may be limiting requests). Resuming in ${s} s… (retry ${attempt + 1} of ${RETRY_WAITS_S.length})` };
        render();
        await sleep(1000);
      }
    }
  }
  cached = true;
  storage.set(true);
  phase = 'ready';
  render();
  if (wantAnswer && context) {
    wantAnswer = false;
    answer();
  }
}

async function deleteModel() {
  const size = `about ${(llm.downloadMB / 1000).toFixed(1)} GB`;
  if (!confirm(`Delete the answer model (${size}) from this browser? Search keeps working, and you can download the model again later.`)) return;
  generation++;
  await stop(false);
  await engine?.unload().catch(() => {});
  worker?.terminate();
  engine = null;
  worker = null;
  const webllm = await import('@mlc-ai/web-llm');
  // Either build may be present (e.g. after a GPU change), but deleting one that isn't would download
  // its file list just to delete it, so check first.
  const ids = [...new Set([llm.model, llm.modelWithoutF16])];
  for (const id of ids) if (await webllm.hasModelInCache(id).catch(() => false)) await webllm.deleteModelAllInfoInCache(id).catch(() => {});
  // WebLLM re-fetches a model's small tensor-cache.json while deleting it; sweep up whatever is left.
  for (const name of await caches.keys()) {
    if (!name.startsWith('webllm')) continue;
    const cache = await caches.open(name);
    for (const request of await cache.keys()) if (ids.some((id) => request.url.includes(id))) await cache.delete(request);
  }
  cached = false;
  storage.set(false);
  answerText = '';
  phase = 'off';
  render(`Deleted. The answer model no longer takes up ${size} in this browser.`);
}

// ---------- answering ----------

async function stop(markStopped = true) {
  if (phase !== 'answering') return;
  generation++;
  stopped = markStopped;
  engine?.interruptGenerate();
  await running?.catch(() => {});
}

async function answer() {
  if (!engine || !context) return;
  await stop(false);
  const { question, hits, threshold } = context;
  const top = hits[0]?.score ?? 0;
  // The gate (threshold measured by npm run eval): below it, the nearest records rarely explain the change.
  if (threshold !== null && top < threshold) {
    phase = 'gated';
    render();
    return;
  }
  const seq = ++generation;
  const messages = buildMessages(question, hits, llm);
  answerText = '';
  stopped = false;
  phase = 'answering';
  render();
  running = (async () => {
    try {
      const stream = await engine!.chat.completions.create({ messages, stream: true, temperature: llm.temperature, max_tokens: llm.maxTokens });
      for await (const chunk of stream) {
        if (seq !== generation) break;
        answerText += chunk.choices[0]?.delta?.content ?? '';
        renderAnswerText();
      }
      if (seq === generation || stopped) {
        phase = 'done';
        render();
      }
    } catch (err) {
      if (seq !== generation) return;
      errorText = `The answer could not be written: ${(err as Error).message ?? err}`;
      phase = 'error';
      render();
    }
  })();
  await running;
}

// ---------- rendering ----------

const panel = () => $('answer');
const gpuTip = 'Your browser is using an integrated GPU. On a laptop with two GPUs, answers are much faster if you set the browser to "High performance" (Windows: Settings → System → Display → Graphics).';

function button(label: string, onClick: () => void, title?: string) {
  const b = el('button', 'button button-small', label);
  b.type = 'button';
  if (title) b.title = title;
  b.addEventListener('click', onClick);
  return b;
}

function note(text: string, kind: 'info' | 'warn' = 'info') {
  return el('p', `answer-note answer-note-${kind}`, text);
}

function citeNode(id: string, given: Set<string>) {
  if (!given.has(id)) {
    const bad = el('span', 'cite cite-invalid', `[${id}]`);
    bad.title = 'Not one of the records the model was given';
    return bad;
  }
  const link = el('a', 'cite', `[${id}]`);
  link.href = `#${id}`;
  link.title = 'Show this record';
  link.addEventListener('click', (e) => {
    e.preventDefault();
    const card = document.getElementById(id);
    if (!card) return;
    card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    card.classList.add('flash');
    setTimeout(() => card.classList.remove('flash'), 1600);
  });
  return link;
}

// Model text goes in only as text nodes; "[pr-123]" groups become links to the result cards.
function renderAnswerText() {
  const box = document.getElementById('answer-text');
  if (!box || !context) return;
  const given = new Set(context.hits.slice(0, llm.records).map((h) => h.chunk.id));
  const paragraphs = answerText.replace(/\*\*/g, '').split(/\n+/).filter((p) => p.trim());
  box.replaceChildren(
    ...paragraphs.map((para) => {
      const p = el('p');
      let last = 0;
      for (const m of para.matchAll(CITATION_GROUP)) {
        const ids = idsInGroup(m[1]);
        if (!ids.length) continue;
        p.append(para.slice(last, m.index));
        ids.forEach((id) => p.append(citeNode(id, given)));
        last = m.index! + m[0].length;
      }
      p.append(para.slice(last));
      return p;
    }),
  );
  if (phase === 'answering') (box.lastElementChild ?? box).append(el('span', 'answer-caret', '▌'));
}

function progressBar(fraction: number) {
  const box = el('div', 'answer-progress');
  const fill = el('div', 'progress-fill');
  fill.style.width = `${Math.round(fraction * 100)}%`;
  box.append(fill);
  return box;
}

function render(message?: string) {
  const body = $('answer-body');
  const actions = $('answer-actions');
  body.replaceChildren();
  actions.replaceChildren();
  const size = `${(llm.downloadMB / 1000).toFixed(1)} GB`;
  const modelName = model.replace(/-MLC$/, '');
  const deleteButton = () => button('Delete model…', deleteModel, `Remove the answer model (${size}) from this browser`);
  panel().hidden = phase === 'checking';

  switch (phase) {
    case 'unsupported':
      body.append(note(`${unsupportedReason} Search works without it.`));
      break;
    case 'off':
      if (message) body.append(note(message));
      body.append(note(`Get a written answer that cites the records below, from a small model (${modelName}) that runs on your GPU. It downloads ${size} once and is then cached in this browser.`));
      if (integratedGpu) body.append(note(gpuTip));
      actions.append(
        button('Turn on answers', () => {
          wantAnswer = context !== null; // answer the question already on screen once the model is ready
          loadModel();
        }),
      );
      break;
    case 'loading':
      body.append(note(progress.text || 'Loading the answer model…'), progressBar(progress.fraction));
      break;
    case 'ready':
      body.append(note(context ? 'Ready.' : 'Ready. Ask a question to get a written answer.'));
      if (context) actions.append(button('Write an answer', answer));
      actions.append(deleteButton());
      break;
    case 'answering':
      body.append(el('div', 'answer-text'));
      body.lastElementChild!.id = 'answer-text';
      actions.append(button('Stop', () => stop()));
      break;
    case 'done': {
      body.append(el('div', 'answer-text'));
      body.lastElementChild!.id = 'answer-text';
      const given = context ? context.hits.slice(0, llm.records).map((h) => h.chunk.id) : [];
      const check = checkAnswer(answerText, given);
      if (stopped) body.append(note('Stopped before the end.', 'warn'));
      else if (!answerText.trim()) body.append(note('The model returned an empty answer.', 'warn'));
      else if (!check.refusal) {
        if (check.invalid.length) body.append(note(`It cites ${check.invalid.join(', ')}, which it was not given. Treat those parts as made up.`, 'warn'));
        if (!check.valid.length) body.append(note('No citations: treat this answer with caution.', 'warn'));
        // Small models sometimes paste the record headers ("[pr-1] (2024-01-01, merged PR) Title") instead of explaining.
        if ((answerText.match(/\]\s*\(\d{4}-\d{2}-\d{2},/g) ?? []).length >= 1) body.append(note('The model mostly copied the records instead of explaining them; read the records below.', 'warn'));
      }
      body.append(el('p', 'answer-meta', `Written by ${modelName}${gpuLabel ? ` on ${gpuLabel}` : ''} from the top ${Math.min(llm.records, given.length)} records below. It can be wrong; check the records.`));
      actions.append(button('Answer again', answer), deleteButton());
      break;
    }
    case 'gated': {
      const top = context?.hits[0]?.score ?? 0;
      body.append(
        el('p', 'answer-refusal', "The history I searched doesn't clearly explain this."),
        note(`The best match scores ${top.toFixed(2)}; written answers need ${context?.threshold?.toFixed(2)} or more for this repo, a cut-off measured on real questions. The nearest records are below.`),
      );
      actions.append(button('Answer anyway', () => answerUngated()), deleteButton());
      break;
    }
    case 'stale':
      body.append(note('The records changed (different filters or repository). The previous answer was cleared.'));
      actions.append(button('Answer again', answer), deleteButton());
      break;
    case 'error':
      body.append(note(errorText, 'warn'));
      actions.append(button('Try again', () => (engine ? answer() : ((phase = 'off'), loadModel()))));
      if (cached) actions.append(deleteButton());
      break;
  }
  if (phase === 'answering' || phase === 'done') renderAnswerText();
}

// "Answer anyway" bypasses the gate once, for this question only.
function answerUngated() {
  if (!context) return;
  const saved = context.threshold;
  context = { ...context, threshold: null };
  answer().finally(() => {
    if (context) context = { ...context, threshold: saved };
  });
}

// ---------- public ----------

export async function initAnswers() {
  await detect();
  if (unsupportedReason) {
    phase = 'unsupported';
    render();
    return;
  }
  // Auto-load only when the visitor turned answers on before and the model is already downloaded.
  const webllm = storage.get() ? await import('@mlc-ai/web-llm') : null;
  cached = webllm ? await webllm.hasModelInCache(model).catch(() => false) : false;
  phase = 'off';
  render();
  if (cached) loadModel();
}

// Called after every ranking. "search" = a new question was asked; "rerank" = filters or repo changed.
export function onResults(question: string, hits: Hit[], threshold: number | null, trigger: 'search' | 'rerank') {
  const changed = trigger === 'rerank' && context !== null && (phase === 'answering' || phase === 'done' || phase === 'gated');
  context = { question, hits, threshold };
  if (phase === 'unsupported' || phase === 'checking' || phase === 'off' || phase === 'error') return;
  if (phase === 'loading') {
    if (trigger === 'search') wantAnswer = true;
    return;
  }
  if (trigger === 'search') {
    answer();
  } else if (changed) {
    stop(false).then(() => {
      answerText = '';
      phase = 'stale';
      render();
    });
  }
}
