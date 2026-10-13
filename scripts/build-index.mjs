// Build the search index from data/raw/ (see fetch.mjs) into public/data/:
//   meta.json      chunks (one per PR, issue or standalone commit)
//   vectors.bin    Float32, count x dim, row-major, L2-normalized
//   manifest.json  model/dtype/pooling/dim/count/repo/head/built_at
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { config, EMBED_DIM, EMBED_DTYPE, EMBED_MODEL, EMBED_POOLING } from '../src/lib/config.ts';
import { loadEmbedder } from '../src/lib/embed.ts';
import { parseIndex } from '../src/lib/retrieval.ts';

const RAW = 'data/raw';
const OUT = 'public/data';
const TEXT_MAX = 1500; // chars of body + extras per chunk
const BODY_MAX = 800; // leave room for commit subjects and comments
const COMMENT_MAX = 300;
const FILES_MAX = 30;
const EMBED_FILES = 10;
const MAINTAINER = new Set(['OWNER', 'MEMBER', 'COLLABORATOR']);
const LABEL = { pr: 'Pull request', issue: 'Issue', commit: 'Commit' };

const raw = (name) => JSON.parse(readFileSync(join(RAW, name), 'utf8'));
const repoInfo = raw('repo.json');
const commits = raw('commits.json');
const merges = raw('merges.json');
const comments = raw('comments.json');

const { botAuthors, dropBotComments, skipSubjects } = config.noise;
const isNoiseBot = (login) => botAuthors.includes(login);
const skipPatterns = skipSubjects.map((s) => new RegExp(s, 'i'));
const pulls = raw('pulls.json').filter((p) => !isNoiseBot(p.author));
const issues = raw('issues.json').filter((i) => !isNoiseBot(i.author));

// ---------- text cleanup ----------

const truncate = (s, n) => (s.length <= n ? s : `${s.slice(0, n - 1).trimEnd()}…`);

function summarizeCode(code) {
  const lines = code.split('\n').map((l) => l.trim()).filter(Boolean);
  if (lines.length === 0) return '';
  // In a traceback the frames are noise; the last line names the error.
  if (lines.some((l) => l.startsWith('Traceback'))) return `\n[traceback] ${lines.at(-1)}\n`;
  return `\n${lines.slice(0, 3).join('\n')}${lines.length > 3 ? ' …' : ''}\n`;
}

function clean(markdown) {
  return markdown
    .replace(/\r/g, '')
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/```[^\n]*\n([\s\S]*?)(```|$)/g, (_, code) => summarizeCode(code))
    .replace(/!\[[^\]]*\]\([^)]*\)/g, '')
    .replace(/<img[^>]*>/gi, '')
    .replace(/<\/?(details|summary|br|p|div)[^>]*>/gi, ' ')
    .replace(/[ \t]+/g, ' ')
    .replace(/\n\s*\n+/g, '\n')
    .trim();
}

// Body first, then extra lines in priority order while they fit.
function compose(body, extras) {
  let text = truncate(clean(body), BODY_MAX);
  for (const line of extras) {
    if (text.length + line.length + 1 > TEXT_MAX) continue;
    text += text ? `\n${line}` : line;
  }
  return text;
}

function commentLines(number) {
  const kept = (comments[number] ?? []).filter((c) => !isNoiseBot(c.author) && !(dropBotComments && c.author.endsWith('[bot]')));
  // Maintainer replies usually carry the "why", so they get the space first.
  const ordered = [...kept.filter((c) => MAINTAINER.has(c.association)), ...kept.filter((c) => !MAINTAINER.has(c.association))];
  return ordered
    .map((c) => ({ c, body: truncate(clean(c.body), COMMENT_MAX) }))
    .filter(({ body }) => body)
    .map(({ c, body }) => `${c.author}${MAINTAINER.has(c.association) ? ' (maintainer)' : ''}: ${body}`);
}

// Git trailers ("Co-Authored-By: ...", "Signed-off-by: ...") say who, not why.
const stripTrailers = (body) => body.replace(/^[A-Za-z-]+-by: .*$/gim, '').trim();

const unique = (xs) => [...new Set(xs)];
const isNoiseSubject = (subject) => skipPatterns.some((re) => re.test(subject.trim()));
// git gives local offsets ("+02:00"), GitHub gives "Z"; one format keeps string sorting/filtering correct.
const utc = (date) => new Date(date).toISOString();

// ---------- fold commits into the PR or issue they belong to ----------

const prByNumber = new Map(pulls.map((p) => [p.number, p]));
const issueByNumber = new Map(issues.map((i) => [i.number, i]));
// Oldest merge first: a commit listed in stacked PRs belongs to the PR that landed it first.
const merged = pulls.filter((p) => p.state === 'merged').sort((a, b) => a.merged_at.localeCompare(b.merged_at));
const owner = new Map(); // commit sha -> { kind: 'pr' | 'issue', number, how }
const assign = (sha, kind, number, how) => owner.has(sha) || owner.set(sha, { kind, number, how });

// 1. Exact SHAs from GitHub: the PR's own commits, plus its merge commit (the squash/rebase result).
for (const p of merged) {
  for (const sha of p.commits) assign(sha, 'pr', p.number, 'pr-commits');
  if (p.merge_commit_sha) assign(p.merge_commit_sha, 'pr', p.number, 'merge-sha');
}
// 2. Local merges ("Merge remote-tracking branch 'user/branch'"): the merged-in side is the PR head.
const prByHead = new Map(merged.map((p) => [p.head_sha, p]));
const prByMergeSha = new Map(merged.map((p) => [p.merge_commit_sha, p]));
for (const m of merges) {
  const viaNumber = prByNumber.get(Number(m.subject.match(/Merge pull request #(\d+)/)?.[1]));
  const p = prByHead.get(m.parents[1]) ?? prByMergeSha.get(m.sha) ?? viaNumber;
  if (p) for (const sha of m.commits) assign(sha, 'pr', p.number, 'merge');
}
// 3. References in the message: "(#N)" / "Merge pull request #N" name a PR; "fix #N" may name either.
for (const c of commits) {
  const prRef = Number(c.subject.match(/\(#(\d+)\)|Merge pull request #(\d+)/)?.slice(1).find(Boolean));
  if (prByNumber.has(prRef)) assign(c.sha, 'pr', prRef, 'subject');
  const fixRef = Number(`${c.subject}\n${c.body}`.match(/\b(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\s*:?\s*#(\d+)/i)?.[1]);
  if (prByNumber.has(fixRef)) assign(c.sha, 'pr', fixRef, 'fix-ref');
  else if (issueByNumber.has(fixRef)) assign(c.sha, 'issue', fixRef, 'fix-ref');
}

const foldedInto = new Map(); // "pr-12" -> commits
for (const c of commits) {
  const o = owner.get(c.sha);
  if (!o) continue;
  const key = `${o.kind}-${o.number}`;
  foldedInto.set(key, [...(foldedInto.get(key) ?? []), c]);
}
const commitLine = (label, list) => {
  const subjects = list.map((c) => c.subject).filter((s) => !isNoiseSubject(s));
  return subjects.length ? truncate(`${label}: ${subjects.join('; ')}`, 300) : null;
};

// ---------- chunks ----------

const prChunks = pulls.map((p) => {
  const folded = foldedInto.get(`pr-${p.number}`) ?? [];
  return {
    id: `pr-${p.number}`,
    type: 'pr',
    title: p.title,
    date: utc(p.merged_at ?? p.created_at),
    author: p.author,
    url: p.url,
    state: p.state,
    files: unique([...p.files, ...folded.flatMap((c) => c.files)]).slice(0, FILES_MAX),
    text: compose(p.body, [commitLine('Commits', folded), ...commentLines(p.number)].filter(Boolean)),
  };
});

const issueChunks = issues.map((i) => {
  const folded = foldedInto.get(`issue-${i.number}`) ?? [];
  return {
    id: `issue-${i.number}`,
    type: 'issue',
    title: i.title,
    date: utc(i.created_at),
    author: i.author,
    url: i.url,
    state: i.state,
    files: unique(folded.flatMap((c) => c.files)).slice(0, FILES_MAX),
    text: compose(i.body, [commitLine('Fixed by commits', folded), ...commentLines(i.number)].filter(Boolean)),
  };
});

const dropped = { bot: 0, subject: 0 };
const commitChunks = commits
  .filter((c) => !owner.has(c.sha))
  .filter((c) => {
    const reason = isNoiseBot(c.author) ? 'bot' : isNoiseSubject(c.subject) ? 'subject' : null;
    if (reason) dropped[reason]++;
    return !reason;
  })
  .map((c) => ({
    id: `commit-${c.sha.slice(0, 7)}`,
    type: 'commit',
    title: c.subject,
    date: utc(c.date),
    author: c.author,
    url: `https://github.com/${repoInfo.owner}/${repoInfo.repo}/commit/${c.sha}`,
    files: c.files.slice(0, FILES_MAX),
    text: truncate(clean(stripTrailers(c.body)), TEXT_MAX),
  }));

const chunks = [...prChunks, ...issueChunks, ...commitChunks].sort((a, b) => b.date.localeCompare(a.date));

// Documents are embedded without the query prefix. Files go before the text so that the ~3% of
// chunks longer than the model's 512 tokens lose the tail of their text, not their file names.
const embedText = (c) => [`${LABEL[c.type]}: ${c.title}`, c.files.length ? `Files: ${c.files.slice(0, EMBED_FILES).join(', ')}` : '', c.text].filter(Boolean).join('\n');

// ---------- embed ----------

console.log(`embedding ${chunks.length} chunks with ${EMBED_MODEL} (${EMBED_DTYPE}, ${EMBED_POOLING} pooling)...`);
const embed = await loadEmbedder();
const vectors = new Float32Array(chunks.length * EMBED_DIM);
// One text per call: q8 quantizes activations per batch, so batching makes a chunk's vector depend
// on its batch-mates (cosine ~0.995 vs embedding it alone). Alone is deterministic, matches how the
// browser embeds, and is no slower on CPU.
const t0 = Date.now();
for (let i = 0; i < chunks.length; i++) {
  const [vector] = await embed([embedText(chunks[i])]);
  vectors.set(vector, i * EMBED_DIM);
  if (i % 200 === 0) process.stdout.write(`  ${i}/${chunks.length}\r`);
}
const embedSeconds = (Date.now() - t0) / 1000;

// ---------- write ----------

const manifest = {
  model: EMBED_MODEL,
  dtype: EMBED_DTYPE,
  pooling: EMBED_POOLING,
  dim: EMBED_DIM,
  count: chunks.length,
  repo: `${repoInfo.owner}/${repoInfo.repo}`,
  head: repoInfo.head,
  built_at: new Date().toISOString(),
};
rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });
writeFileSync(join(OUT, 'meta.json'), JSON.stringify(chunks));
writeFileSync(join(OUT, 'vectors.bin'), Buffer.from(vectors.buffer));
writeFileSync(join(OUT, 'manifest.json'), JSON.stringify(manifest, null, 2));

// ---------- verify what was written ----------

const read = (name) => readFileSync(join(OUT, name));
const bin = read('vectors.bin');
const index = parseIndex(JSON.parse(read('manifest.json')), JSON.parse(read('meta.json')), bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
let minNorm = Infinity;
let maxNorm = 0;
for (let i = 0; i < index.manifest.count; i++) {
  let sq = 0;
  for (let d = 0; d < EMBED_DIM; d++) sq += index.vectors[i * EMBED_DIM + d] ** 2;
  minNorm = Math.min(minNorm, Math.sqrt(sq));
  maxNorm = Math.max(maxNorm, Math.sqrt(sq));
}
const ids = new Set(index.chunks.map((c) => c.id));
if (ids.size !== index.chunks.length) throw new Error('duplicate chunk ids');
if (!(minNorm > 0.999 && maxNorm < 1.001)) throw new Error(`vector norms out of range: ${minNorm}..${maxNorm}`);
// Each stored row must be its own chunk's vector: re-embed a spread of chunks and compare.
for (const i of [0, Math.floor(chunks.length / 2), chunks.length - 1]) {
  const [fresh] = await embed([embedText(index.chunks[i])]);
  let cos = 0;
  for (let d = 0; d < EMBED_DIM; d++) cos += fresh[d] * index.vectors[i * EMBED_DIM + d];
  if (cos < 0.9999) throw new Error(`row ${i} (${index.chunks[i].id}) does not match a fresh embedding: cos ${cos}`);
}
const unsorted = index.chunks.findIndex((c, i) => i > 0 && c.date > index.chunks[i - 1].date);
if (unsorted !== -1) throw new Error(`chunks not newest-first at ${index.chunks[unsorted].id}`);

const byHow = commits.filter((c) => owner.has(c.sha)).reduce((m, c) => ((m[owner.get(c.sha).how] = (m[owner.get(c.sha).how] ?? 0) + 1), m), {});
const kb = (name) => `${(read(name).length / 1024).toFixed(0)} KB`;
console.log(`
${manifest.repo} @ ${manifest.head.slice(0, 7)}: ${chunks.length} chunks
  PRs       ${prChunks.length}
  issues    ${issueChunks.length} (${issueChunks.filter((c) => c.files.length).length} with files from fix commits)
  commits   ${commitChunks.length} standalone; ${commits.filter((c) => owner.has(c.sha)).length} folded (${Object.entries(byHow).map(([k, v]) => `${k} ${v}`).join(', ')}); dropped ${dropped.subject} by subject, ${dropped.bot} bot
  vectors   norms ${minNorm.toFixed(4)}..${maxNorm.toFixed(4)}; ${index.vectors.length / EMBED_DIM} rows = manifest count ${manifest.count}
  output    meta.json ${kb('meta.json')}, vectors.bin ${kb('vectors.bin')}
  embedding ${embedSeconds.toFixed(0)}s (${(chunks.length / embedSeconds).toFixed(1)} chunks/s)`);
