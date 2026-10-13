// Recent questions, kept in this browser only (localStorage), shown in a Windows 98-style drop-down menu.
import { $, el } from './ui.ts';

export type HistoryEntry = { q: string; repo: string; at: number };

const KEY = 'git-why:history';
const MAX = 20;

export function loadHistory(): HistoryEntry[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(KEY) ?? '[]');
    return Array.isArray(parsed) ? parsed.filter((e) => typeof e?.q === 'string' && typeof e?.repo === 'string') : [];
  } catch {
    return [];
  }
}

function save(entries: HistoryEntry[]) {
  try {
    localStorage.setItem(KEY, JSON.stringify(entries.slice(0, MAX)));
  } catch {
    // storage blocked (private mode etc.): history just isn't kept
  }
}

// Newest first; asking the same question in the same repo again moves it to the top.
export function addToHistory(q: string, repo: string) {
  const rest = loadHistory().filter((e) => !(e.q === q && e.repo === repo));
  save([{ q, repo, at: Date.now() }, ...rest]);
}

export function clearHistory() {
  save([]);
}

export function initHistoryMenu(onPick: (entry: HistoryEntry) => void) {
  const toggle = $<HTMLButtonElement>('history-button');
  const menu = $('history-menu');

  const close = (focusButton = false) => {
    menu.hidden = true;
    toggle.setAttribute('aria-expanded', 'false');
    if (focusButton) toggle.focus();
  };

  const item = (label: string, onClick: () => void, detail?: string) => {
    const b = el('button', 'menu-item');
    b.type = 'button';
    b.setAttribute('role', 'menuitem');
    b.append(el('span', 'menu-item-label', label));
    if (detail) b.append(el('span', 'menu-item-detail', detail));
    b.addEventListener('click', onClick);
    return b;
  };

  const open = () => {
    const entries = loadHistory();
    menu.replaceChildren(
      ...(entries.length
        ? entries.map((e) =>
            item(e.q, () => {
              close();
              onPick(e);
            }, e.repo),
          )
        : [el('div', 'menu-empty', 'No questions yet.')]),
      el('div', 'menu-separator'),
      item('Clear history', () => {
        clearHistory();
        close(true);
      }),
    );
    if (!entries.length) menu.querySelector<HTMLButtonElement>('.menu-item')!.disabled = true;
    menu.hidden = false;
    toggle.setAttribute('aria-expanded', 'true');
    menu.querySelector<HTMLButtonElement>('.menu-item:not(:disabled)')?.focus();
  };

  toggle.addEventListener('click', () => (menu.hidden ? open() : close()));
  document.addEventListener('mousedown', (e) => {
    if (!menu.hidden && !menu.contains(e.target as Node) && e.target !== toggle) close();
  });
  menu.addEventListener('keydown', (e) => {
    const items = [...menu.querySelectorAll<HTMLButtonElement>('.menu-item:not(:disabled)')];
    const i = items.indexOf(document.activeElement as HTMLButtonElement);
    if (e.key === 'Escape') close(true);
    else if (e.key === 'ArrowDown') items[(i + 1) % items.length]?.focus();
    else if (e.key === 'ArrowUp') items[(i - 1 + items.length) % items.length]?.focus();
    else return;
    e.preventDefault();
  });
}
