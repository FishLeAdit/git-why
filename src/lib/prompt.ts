// The answer prompt and the check of what the model cited. Isomorphic and pure, so evals and tests can
// use it too.
//
// This is PLAN.md's M6 prompt, revised by measurement: on the 31 reviewed eval questions (Qwen2.5-3B,
// top 4 dense records each), the plain rule list made small models refuse almost everything or answer
// without citations. Asking for short answers in its own words, three worked examples (one cited answer,
// two refusals), and an explicit "mentions the topic but not the reason" rule gave the most honest
// results: no invented motives on unanswerable questions, at the cost of a few extra refusals.
import type { Chunk, Hit } from './retrieval.ts';

export const REFUSAL = "The history I searched doesn't explain this.";

export const SYSTEM_PROMPT = `You explain why a software project changed, using only the records the user gives you.
Answer in one to three sentences, in your own words; do not copy the record headers. After each sentence, cite the record ids it is based on in brackets, like [pr-123].
Use only what the records say; do not guess motives.
If the records mention the topic but do not say why it changed, or the question assumes something the records do not show, reply exactly: "${REFUSAL}"`;

export type ChatMessage = { role: 'system' | 'user' | 'assistant'; content: string };

// Made-up records about an unrelated project, so the examples can't leak answers to real questions.
const EXAMPLES: ChatMessage[] = [
  {
    role: 'user',
    content: `Records:\n[pr-7] (2023-01-05, merged PR) Cache downloaded datasets\nFiles: tools/download.py\nEvery run downloaded the datasets again, which took ten minutes on slow connections. Files are now kept in a local cache.\n\n[issue-9] (2023-02-01, closed issue) Typo in README\nThe word "segmentation" is misspelled.\n\nQuestion: Why are datasets cached now?`,
  },
  { role: 'assistant', content: 'Datasets are cached because every run downloaded them again, which took ten minutes on slow connections [pr-7].' },
  {
    role: 'user',
    content: `Records:\n[issue-3] (2023-03-01, closed issue) Training crashes on Windows\nA user reports a crash; the maintainer asks for the full log.\n\nQuestion: Why was the learning rate lowered?`,
  },
  { role: 'assistant', content: REFUSAL },
  {
    role: 'user',
    content: `Records:\n[issue-12] (2023-04-02, closed issue) Dark mode looks wrong on Safari\nThe sidebar stays white in dark mode on Safari 16. Maintainer: fixed in the next release.\n\n[pr-14] (2023-04-03, merged PR) Fix sidebar colour in dark mode\nUse the theme variable for the sidebar background.\n\nQuestion: Why was dark mode made the default?`,
  },
  { role: 'assistant', content: REFUSAL },
];

const KIND: Record<Chunk['type'], string> = { pr: 'PR', issue: 'issue', commit: 'commit' };

const truncate = (s: string, n: number) => (s.length <= n ? s : `${s.slice(0, n - 1).trimEnd()}…`);

export function formatRecord(chunk: Chunk, maxChars: number): string {
  const kind = chunk.state ? `${chunk.state} ${KIND[chunk.type]}` : KIND[chunk.type];
  const lines = [`[${chunk.id}] (${chunk.date.slice(0, 10)}, ${kind}) ${chunk.title}`];
  // Many commit records are a subject plus file names, so the files are part of the evidence.
  if (chunk.files.length) lines.push(`Files: ${chunk.files.slice(0, 5).join(', ')}`);
  if (chunk.text) lines.push(truncate(chunk.text, maxChars));
  return lines.join('\n');
}

export function buildMessages(question: string, hits: Hit[], opts: { records: number; recordChars: number }): ChatMessage[] {
  const records = hits.slice(0, opts.records).map((h) => formatRecord(h.chunk, opts.recordChars));
  return [
    { role: 'system', content: SYSTEM_PROMPT },
    ...EXAMPLES,
    { role: 'user', content: `Records:\n${records.join('\n\n')}\n\nQuestion: ${question}\n\nAnswer the question "${question}" with citations.` },
  ];
}

// ---------- reading the answer ----------

const ID = /^(?:pr|issue)-\d+$|^commit-[0-9a-f]{7,40}$/i;

// A bracket group like "[pr-123]" or "[pr-123, issue-45]"; small models do both.
export const CITATION_GROUP = /\[([^\]\n]{3,120})\]/g;

export function idsInGroup(group: string): string[] {
  return group
    .split(/[\s,;]+/)
    .map((s) => s.trim().toLowerCase())
    .filter((s) => ID.test(s));
}

export type AnswerCheck = {
  refusal: boolean; // the model said the records don't explain it
  valid: string[]; // cited and among the records it was given
  invalid: string[]; // cited but NOT among the records it was given (made up)
};

const norm = (s: string) => s.replace(/[‘’]/g, "'").replace(/\s+/g, ' ').trim().toLowerCase();

export function checkAnswer(answer: string, givenIds: string[]): AnswerCheck {
  const given = new Set(givenIds.map((id) => id.toLowerCase()));
  const cited = new Set<string>();
  for (const m of answer.matchAll(CITATION_GROUP)) for (const id of idsInGroup(m[1])) cited.add(id);
  // The refusal sentence counts anywhere: the model sometimes adds it after a clause of its own
  // ("… did not switch to blosc2; the history I searched doesn't explain this").
  const refusal = norm(answer).includes(norm(REFUSAL).replace(/\.$/, ''));
  return { refusal, valid: [...cited].filter((id) => given.has(id)), invalid: [...cited].filter((id) => !given.has(id)) };
}
