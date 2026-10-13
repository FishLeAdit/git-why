// Evaluate retrieval on eval/questions.json and write eval/results.md.
//   npm run eval
// For every question: rank the index with dense (embeddings), keyword (BM25) and hybrid (RRF) search,
// then report hit@k and MRR on answerable questions, and how well a cut-off on the top dense score
// separates answerable from unanswerable questions (the gate the in-browser answer model will use).
// Output is deterministic: rerunning on the same index and questions gives the same file.
import { readFileSync, writeFileSync } from 'node:fs';
import { EMBED_DTYPE, EMBED_MODEL, EMBED_POOLING, repoKey } from '../src/lib/config.ts';
import { embedQuery, loadEmbedder } from '../src/lib/embed.ts';
import { buildKeywordIndex, parseIndex, searchDense, searchHybrid, searchKeyword } from '../src/lib/retrieval.ts';

const QUESTIONS = 'eval/questions.json';
const RESULTS = 'eval/results.md';
const DEPTH = 100; // ranks beyond this count as "not found"
const METHODS = ['dense', 'keyword', 'hybrid'];

const questions = JSON.parse(readFileSync(QUESTIONS, 'utf8'));
const unreviewed = questions.filter((q) => !q.reviewed);

// ---------- run every question ----------

const embed = await loadEmbedder();
const indexes = new Map();
function loadRepo(name) {
  if (!indexes.has(name)) {
    const dir = `public/data/${repoKey({ owner: name.split('/')[0], repo: name.split('/')[1] })}`;
    const read = (f) => readFileSync(`${dir}/${f}`);
    const bin = read('vectors.bin');
    const index = parseIndex(JSON.parse(read('manifest.json')), JSON.parse(read('meta.json')), bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
    const ids = new Set(index.chunks.map((c) => c.id));
    indexes.set(name, { index, keyword: buildKeywordIndex(index.chunks), ids });
  }
  return indexes.get(name);
}

const rows = [];
for (const q of questions) {
  const { index, keyword, ids } = loadRepo(q.repo);
  const missing = q.gold.filter((g) => !ids.has(g));
  if (missing.length) throw new Error(`${q.id}: gold ids not in the ${q.repo} index: ${missing.join(', ')}`);
  if (q.answerable !== q.gold.length > 0) throw new Error(`${q.id}: answerable questions need gold ids, unanswerable ones none`);

  const vector = await embedQuery(embed, q.question);
  const ranked = {
    dense: searchDense(index, vector, DEPTH),
    keyword: searchKeyword(keyword, q.question, DEPTH),
    hybrid: searchHybrid(index, keyword, q.question, vector, DEPTH),
  };
  // Rank (1-based) of the first gold record, or null if none is in the top DEPTH.
  const rank = Object.fromEntries(METHODS.map((m) => {
    const i = ranked[m].findIndex((h) => q.gold.includes(h.chunk.id));
    return [m, i === -1 ? null : i + 1];
  }));
  rows.push({ q, rank, top1: ranked.dense[0].score, top1Id: ranked.dense[0].chunk.id });
}

// ---------- retrieval metrics (answerable questions) ----------

const answerable = rows.filter((r) => r.q.answerable);
const hitAt = (m, k) => answerable.filter((r) => r.rank[m] !== null && r.rank[m] <= k).length / answerable.length;
const mrr = (m) => answerable.reduce((s, r) => s + (r.rank[m] !== null && r.rank[m] <= 10 ? 1 / r.rank[m] : 0), 0) / answerable.length;
const metrics = METHODS.map((m) => ({ m, h1: hitAt(m, 1), h5: hitAt(m, 5), h10: hitAt(m, 10), mrr: mrr(m) }));

// ---------- abstention: a cut-off on the top dense cosine ----------
// Refuse (say "the history doesn't explain this") when the best dense score is below the threshold.
// The threshold is chosen on half the questions and reported on the other half, then also on all
// questions with the threshold chosen on all of them (optimistic, since it is tuned on its own test set).

function evaluateGate(set, t) {
  const ans = set.filter((r) => r.q.answerable);
  const un = set.filter((r) => !r.q.answerable);
  const refused = (r) => r.top1 < t;
  return {
    refusal: un.length ? un.filter(refused).length / un.length : NaN, // unanswerable correctly refused
    falseRefusal: ans.length ? ans.filter(refused).length / ans.length : NaN, // answerable wrongly refused
    nAns: ans.length,
    nUn: un.length,
  };
}

// Best balanced accuracy on `set`; ties go to the lower threshold (fewer false refusals).
function chooseThreshold(set) {
  const scores = [...new Set(set.map((r) => r.top1))].sort((a, b) => a - b);
  const candidates = [scores[0] - 1e-6, ...scores.slice(1).map((s, i) => (s + scores[i]) / 2), scores.at(-1) + 1e-6];
  let best = { t: candidates[0], score: -1 };
  for (const t of candidates) {
    const g = evaluateGate(set, t);
    const score = (g.refusal + (1 - g.falseRefusal)) / 2;
    if (score > best.score + 1e-12) best = { t, score };
  }
  return Math.round(best.t * 1000) / 1000;
}

// Deterministic split: alternate questions within each group (answerable / unanswerable), in file order.
const split = (group) => ({ tune: group.filter((_, i) => i % 2 === 0), test: group.filter((_, i) => i % 2 === 1) });
const ansSplit = split(answerable);
const unSplit = split(rows.filter((r) => !r.q.answerable));
const tune = [...ansSplit.tune, ...unSplit.tune];
const test = [...ansSplit.test, ...unSplit.test];
const tTune = chooseThreshold(tune);
const tAll = chooseThreshold(rows);
const gateTest = evaluateGate(test, tTune);
const gateAll = evaluateGate(rows, tAll);

const dist = (set) => {
  const s = set.map((r) => r.top1).sort((a, b) => a - b);
  return s.length ? `${s[0].toFixed(3)} / ${s[Math.floor((s.length - 1) / 2)].toFixed(3)} / ${s.at(-1).toFixed(3)}` : 'n/a';
};

// ---------- write results.md ----------

const pct = (x) => (Number.isNaN(x) ? 'n/a' : `${Math.round(x * 100)}%`);
const fmtRank = (r) => (r === null ? `>${DEPTH}` : String(r));
const repos = [...indexes.entries()].map(([name, { index }]) => `${name} @ ${index.manifest.head.slice(0, 7)} (${index.manifest.count} records, built ${index.manifest.built_at.slice(0, 10)})`);
const md = [
  '# Retrieval evaluation',
  '',
  unreviewed.length
    ? `> **DRAFT, NOT FOR PUBLISHING:** ${unreviewed.length} of ${questions.length} questions have not been reviewed by a human (\`"reviewed": false\` in \`${QUESTIONS}\`). These numbers must not go into the README.`
    : `> All ${questions.length} questions and their gold records were reviewed by a human.`,
  '',
  `Generated by \`npm run eval\` from \`${QUESTIONS}\`: ${answerable.length} answerable and ${rows.length - answerable.length} unanswerable questions.`,
  `Index: ${repos.join('; ')}. Embeddings: ${EMBED_MODEL}, ${EMBED_DTYPE}, ${EMBED_POOLING} pooling.`,
  '',
  '## Ranking (answerable questions)',
  '',
  'A question counts as a hit at k when any of its gold records is in the top k. MRR is computed over the top 10.',
  '',
  '| Method | hit@1 | hit@5 | hit@10 | MRR@10 |',
  '|---|---|---|---|---|',
  ...metrics.map(({ m, h1, h5, h10, mrr }) => `| ${m} | ${pct(h1)} | ${pct(h5)} | ${pct(h10)} | ${mrr.toFixed(3)} |`),
  '',
  `- **dense**: cosine similarity of ${EMBED_MODEL} embeddings (what the site uses today).`,
  '- **keyword**: BM25 over title (2× weight), text and file paths, with English stop words removed.',
  '- **hybrid**: reciprocal rank fusion (K = 60) of the dense and keyword top 100.',
  '',
  '## Saying "the history doesn\'t explain this"',
  '',
  'The gate refuses when the top dense cosine score is below a threshold.',
  '',
  '| Top-1 dense score | min / median / max |',
  '|---|---|',
  `| answerable (${answerable.length}) | ${dist(answerable)} |`,
  `| unanswerable (${rows.length - answerable.length}) | ${dist(rows.filter((r) => !r.q.answerable))} |`,
  '',
  '| Threshold chosen on | Reported on | Threshold | Unanswerable refused | Answerable wrongly refused |',
  '|---|---|---|---|---|',
  `| half the questions (${tune.length}) | the other half (${test.length}) | ${tTune.toFixed(3)} | ${pct(gateTest.refusal)} of ${gateTest.nUn} | ${pct(gateTest.falseRefusal)} of ${gateTest.nAns} |`,
  `| all questions (${rows.length}) | the same questions (optimistic) | ${tAll.toFixed(3)} | ${pct(gateAll.refusal)} of ${gateAll.nUn} | ${pct(gateAll.falseRefusal)} of ${gateAll.nAns} |`,
  '',
  `With this few questions every percentage moves a lot per question, so treat the threshold as a starting point, not a measured constant. The threshold is chosen for the best balanced accuracy (mean of the two rates); ties go to the lower threshold.`,
  '',
  '## Per question',
  '',
  'Rank of the first gold record under each method (lower is better), and the top dense hit.',
  '',
  '| id | question | gold | dense | keyword | hybrid | top-1 dense |',
  '|---|---|---|---|---|---|---|',
  ...rows.map(({ q, rank, top1, top1Id }) => `| ${q.id} | ${q.question.replace(/\|/g, '\\|')} | ${q.gold.join(', ') || '(none)'} | ${q.answerable ? fmtRank(rank.dense) : '–'} | ${q.answerable ? fmtRank(rank.keyword) : '–'} | ${q.answerable ? fmtRank(rank.hybrid) : '–'} | ${top1.toFixed(3)} ${top1Id} |`),
  '',
].join('\n');
writeFileSync(RESULTS, md);

console.log(`\n${questions.length} questions (${answerable.length} answerable)${unreviewed.length ? `, ${unreviewed.length} NOT REVIEWED` : ''}`);
for (const { m, h1, h5, h10, mrr } of metrics) console.log(`  ${m.padEnd(8)} hit@1 ${pct(h1).padStart(4)}  hit@5 ${pct(h5).padStart(4)}  hit@10 ${pct(h10).padStart(4)}  MRR ${mrr.toFixed(3)}`);
console.log(`  gate: threshold ${tTune.toFixed(3)} (tuned on half) → refuses ${pct(gateTest.refusal)} of unanswerable, ${pct(gateTest.falseRefusal)} of answerable on the other half`);
console.log(`wrote ${RESULTS}`);
