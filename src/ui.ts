// DOM rendering. Record text comes from GitHub users, so it only ever goes in via textContent.
import type { Chunk, Filters, Hit } from './lib/retrieval.ts';

const TYPE_LABEL: Record<Chunk['type'], string> = { pr: 'PR', issue: 'Issue', commit: 'Commit' };
const SNIPPET_CHARS = 260;
const FILES_SHOWN = 3;

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, text?: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

export function setStatus(text: string, progress?: number) {
  $('status-text').textContent = text;
  $('progress-panel').hidden = progress === undefined;
  if (progress !== undefined) $('progress').style.width = `${Math.round(Math.min(1, Math.max(0, progress)) * 100)}%`;
}

// The Windows 98 hourglass cursor while the page is waiting on something.
export function setBusy(busy: boolean) {
  document.body.classList.toggle('is-busy', busy);
}

export function showWarning(lines: string[]) {
  const box = $('warning');
  box.hidden = lines.length === 0;
  const body = el('div');
  body.append(el('strong', '', 'The index was built with a different embedding setup, so results will be wrong.'), ...lines.map((l) => el('div', '', l)));
  box.replaceChildren(body);
}

export function renderExamples(questions: string[], onPick: (q: string) => void) {
  $('examples').replaceChildren(
    ...questions.map((q) => {
      const button = el('button', 'example', q);
      button.type = 'button';
      button.addEventListener('click', () => onPick(q));
      return button;
    }),
  );
}

// With a file filter, show the files that matched it; otherwise the first few.
function shownFiles(chunk: Chunk, filters: Filters): { files: string[]; more: number } {
  const needle = filters.file?.toLowerCase();
  const ordered = needle ? chunk.files.filter((f) => f.toLowerCase().includes(needle)) : chunk.files;
  return { files: ordered.slice(0, FILES_SHOWN), more: chunk.files.length - Math.min(ordered.length, FILES_SHOWN) };
}

function snippet(text: string): string {
  const flat = text.replace(/\s+/g, ' ').trim();
  return flat.length <= SNIPPET_CHARS ? flat : `${flat.slice(0, SNIPPET_CHARS).trimEnd()}…`;
}

function card({ chunk, score }: Hit, filters: Filters): HTMLLIElement {
  const item = el('li', `card card-${chunk.type}`);
  item.id = chunk.id;

  const head = el('div', 'card-head');
  head.append(el('span', `badge badge-${chunk.type}`, TYPE_LABEL[chunk.type]));
  if (chunk.state && chunk.type !== 'commit') head.append(el('span', `state state-${chunk.state}`, chunk.state));
  const title = el('a', 'card-title', chunk.title);
  title.href = chunk.url;
  title.target = '_blank';
  title.rel = 'noopener';
  head.append(title, el('span', 'score', score.toFixed(2)));
  head.lastElementChild!.setAttribute('title', 'Cosine similarity to your question');

  const number = chunk.type === 'commit' ? chunk.id.slice('commit-'.length) : `#${chunk.id.split('-')[1]}`;
  const meta = el('div', 'card-meta', `${chunk.date.slice(0, 10)} · ${chunk.author} · ${number}`);

  item.append(head, meta);
  if (chunk.text) item.append(el('p', 'snippet', snippet(chunk.text)));

  const { files, more } = shownFiles(chunk, filters);
  if (files.length) {
    const list = el('div', 'files');
    list.append(...files.map((f) => el('code', '', f)));
    if (more > 0) list.append(el('span', 'more', `+${more} more`));
    item.append(list);
  }
  return item;
}

export function renderHits(hits: Hit[], filters: Filters) {
  $('results').replaceChildren(...hits.map((hit) => card(hit, filters)));
}
