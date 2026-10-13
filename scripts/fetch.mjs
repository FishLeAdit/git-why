// Fetch repo history into data/raw/<owner>__<repo>/ (gitignored).
//   commits:  blobless git clone + git log (no API calls)
//   PRs, issues, comments: GitHub REST API, list endpoints in bulk, every page cached to disk
//   npm run fetch                          every repo in config.json
//   npm run fetch -- --repo owner/name     just one
// Re-running reuses the cache; delete a repo's folder to fetch it fresh.
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { repoKey, repoName } from '../src/lib/config.ts';
import { parseRepoArgs } from './repos.mjs';

// Pinned: 2026-03-10 removes merge_commit_sha from PR payloads, which build-index needs.
const API_VERSION = '2022-11-28';
const PER_PAGE = 100;
const CONCURRENCY = 4;

const token = process.env.GITHUB_TOKEN;
if (!token) console.warn('warning: GITHUB_TOKEN not set; anonymous limit is 60 requests/hour. See .env.example.');

const readJson = (path) => JSON.parse(readFileSync(path, 'utf8'));
const writeJson = (path, data) => writeFileSync(path, JSON.stringify(data, null, 1));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------- GitHub API ----------

async function api(ctx, url, attempt = 0) {
  const res = await fetch(url, {
    headers: {
      Accept: 'application/vnd.github+json',
      'X-GitHub-Api-Version': API_VERSION,
      'User-Agent': 'git-why-fetch',
      ...(token && { Authorization: `Bearer ${token}` }),
    },
  });
  ctx.stats.requests++;
  ctx.stats.rateRemaining = res.headers.get('x-ratelimit-remaining') ?? ctx.stats.rateRemaining;

  if (res.ok) return res.json();

  const body = await res.text();
  const retryAfter = Number(res.headers.get('retry-after'));
  const reset = Number(res.headers.get('x-ratelimit-reset'));
  const exhausted = res.headers.get('x-ratelimit-remaining') === '0';
  // A 403 is only worth waiting on when it's a rate limit; other 403s (e.g. no access) fail fast.
  const rateLimited = res.status === 429 || (res.status === 403 && (exhausted || retryAfter || /rate limit/i.test(body)));
  if (rateLimited && attempt < 5) {
    const waitMs = retryAfter ? retryAfter * 1000 : exhausted ? reset * 1000 - Date.now() + 1000 : 60_000;
    console.warn(`rate limited (${res.status}); waiting ${Math.ceil(waitMs / 1000)}s`);
    await sleep(Math.max(waitMs, 1000));
    return api(ctx, url, attempt + 1);
  }
  if (res.status >= 500 && attempt < 5) {
    await sleep(2 ** attempt * 1000);
    return api(ctx, url, attempt + 1);
  }
  throw new Error(`GET ${url} -> ${res.status} ${body.slice(0, 200)}`);
}

// Fetch one page of `path` (relative to the repo API), or read it from the cache.
// The cache key includes the query, so changing e.g. `since` never reuses pages fetched for another query.
async function cachedPage(ctx, path, query, page) {
  const key = query ? `${path.replaceAll('/', '-')}@${createHash('sha1').update(query).digest('hex').slice(0, 8)}` : path.replaceAll('/', '-');
  const dir = join(ctx.cache, key);
  const file = join(dir, `${page}.json`);
  if (existsSync(file)) {
    ctx.stats.cached++;
    return readJson(file);
  }
  mkdirSync(dir, { recursive: true });
  const data = await api(ctx, `${ctx.api}/${path}?${query}&per_page=${PER_PAGE}&page=${page}`);
  writeJson(file, data);
  return data;
}

// Walk a list endpoint page by page. `done(items)` returns true once we have enough.
async function paginate(ctx, path, query, done = () => false) {
  const all = [];
  for (let page = 1; ; page++) {
    const items = await cachedPage(ctx, path, query, page);
    all.push(...items);
    if (items.length < PER_PAGE || done(all)) return all;
  }
}

async function mapLimit(items, limit, fn) {
  const out = new Array(items.length);
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const i = next++;
      out[i] = await fn(items[i]);
    }
  };
  await Promise.all(Array.from({ length: limit }, worker));
  return out;
}

// Newest-first pages shift when items are opened between fetches (e.g. a run resumed days later),
// which can repeat an item across a page boundary. Keep the first copy.
const uniqueBy = (items, key) => {
  const seen = new Set();
  return items.filter((x) => !seen.has(x[key]) && seen.add(x[key]));
};

// ---------- PRs ----------

async function fetchPulls(ctx) {
  const pages = await paginate(ctx, 'pulls', 'state=all&sort=created&direction=desc', (all) => ctx.pastSince(all) || all.length >= ctx.cap);
  const pulls = uniqueBy(pages, 'number').filter(ctx.createdInRange).slice(0, ctx.cap);

  // For merged PRs: changed file names, and the PR's commit SHAs so build-index can fold
  // fast-forwarded PRs (no merge commit, no "#N" in the subject). First page only: files are capped
  // at ~30 downstream, and PRs with >100 commits are rare.
  const merged = pulls.filter((p) => p.merged_at);
  const files = await mapLimit(merged, CONCURRENCY, (p) => cachedPage(ctx, `pulls/${p.number}/files`, '', 1));
  const prCommits = await mapLimit(merged, CONCURRENCY, (p) => cachedPage(ctx, `pulls/${p.number}/commits`, '', 1));
  const filesByPr = new Map(merged.map((p, i) => [p.number, files[i].map((f) => f.filename)]));
  const commitsByPr = new Map(merged.map((p, i) => [p.number, prCommits[i].map((c) => c.sha)]));

  return pulls.map((p) => ({
    number: p.number,
    title: p.title,
    body: p.body ?? '',
    author: p.user?.login ?? 'ghost',
    created_at: p.created_at,
    closed_at: p.closed_at,
    merged_at: p.merged_at,
    state: p.merged_at ? 'merged' : p.state,
    merge_commit_sha: p.merged_at ? p.merge_commit_sha : null,
    // Some repos (nnUNet) merge PRs locally ("Merge remote-tracking branch 'user/branch'"), so subjects
    // rarely say "#N". The PR's head commit and branch let build-index match those merges.
    head_sha: p.head.sha,
    head_ref: p.head.ref,
    head_owner: p.head.repo?.owner.login ?? null,
    labels: p.labels.map((l) => l.name),
    url: p.html_url,
    files: filesByPr.get(p.number) ?? [],
    commits: commitsByPr.get(p.number) ?? [],
  }));
}

// ---------- Issues ----------

async function fetchIssues(ctx) {
  // The issues list also returns PRs (they carry a `pull_request` field); drop those.
  const isIssue = (i) => !i.pull_request;
  const pages = await paginate(ctx, 'issues', 'state=all&sort=created&direction=desc', (all) => ctx.pastSince(all) || all.filter(isIssue).length >= ctx.cap);
  return uniqueBy(pages, 'number')
    .filter(isIssue)
    .filter(ctx.createdInRange)
    .slice(0, ctx.cap)
    .map((i) => ({
      number: i.number,
      title: i.title,
      body: i.body ?? '',
      author: i.user?.login ?? 'ghost',
      created_at: i.created_at,
      closed_at: i.closed_at,
      state: i.state,
      state_reason: i.state_reason ?? null,
      labels: i.labels.map((l) => l.name),
      url: i.html_url,
    }));
}

// ---------- Comments (bulk, grouped locally by issue/PR number) ----------

async function fetchComments(ctx, items) {
  // A comment is never older than its issue/PR, so the oldest kept item bounds the comments we need.
  // With maxItems set this keeps a first run small even when `since` is null.
  if (items.length === 0) return { byNumber: {}, total: 0 };
  const wanted = new Set(items.map((x) => x.number));
  const oldest = Math.max(ctx.sinceTime, Math.min(...items.map((x) => Date.parse(x.created_at))));
  const sinceQuery = Number.isFinite(oldest) ? `&since=${encodeURIComponent(new Date(oldest).toISOString())}` : '';
  const numberFromUrl = (url) => Number(url.slice(url.lastIndexOf('/') + 1));

  const issueComments = await paginate(ctx, 'issues/comments', `sort=created&direction=asc${sinceQuery}`);
  const reviewComments = await paginate(ctx, 'pulls/comments', `sort=created&direction=asc${sinceQuery}`);

  const byNumber = {};
  const add = (number, comment) => {
    if (!wanted.has(number)) return;
    (byNumber[number] ??= []).push(comment);
  };
  for (const c of issueComments) {
    add(numberFromUrl(c.issue_url), { kind: 'issue', author: c.user?.login ?? 'ghost', association: c.author_association, created_at: c.created_at, body: c.body ?? '' });
  }
  for (const c of reviewComments) {
    add(numberFromUrl(c.pull_request_url), { kind: 'review', author: c.user?.login ?? 'ghost', association: c.author_association, created_at: c.created_at, body: c.body ?? '', path: c.path });
  }
  for (const list of Object.values(byNumber)) list.sort((a, b) => a.created_at.localeCompare(b.created_at));
  return { byNumber, total: issueComments.length + reviewComments.length };
}

// ---------- Commits (local git, no API) ----------

const git = (dir, ...args) => execFileSync('git', ['-C', dir, ...args], { encoding: 'utf8', maxBuffer: 1 << 30 });

function fetchCommits(ctx) {
  const dir = join(ctx.raw, 'repo.git');
  if (!existsSync(dir)) {
    console.log(`cloning ${ctx.name} (blobless)...`);
    execFileSync('git', ['clone', '--bare', '--filter=blob:none', '--quiet', `https://github.com/${ctx.name}.git`, dir], { stdio: 'inherit' });
  }
  const head = git(dir, 'rev-parse', 'HEAD').trim();
  const { since, maxItems } = ctx;
  const sinceArgs = [...(since ? [`--since=${since}`] : []), ...(maxItems ? [`--max-count=${maxItems}`] : [])];

  // Record separator \x1e before each commit, unit separator \x1f between fields, \x1d ends the body.
  // --no-renames keeps git from lazily fetching blobs for rename detection.
  const log = git(dir, 'log', 'HEAD', '--no-merges', '--no-renames', '--name-only', ...sinceArgs, '--format=%x1e%H%x1f%an%x1f%ae%x1f%aI%x1f%s%x1f%b%x1d');
  const commits = log
    .split('\x1e')
    .slice(1)
    .map((record) => {
      const [header, fileBlock] = record.split('\x1d');
      const [sha, author, email, date, subject, body] = header.split('\x1f');
      return { sha, author, email, date, subject, body: body.trim(), files: fileBlock.split('\n').filter(Boolean) };
    });

  // Merge commits are excluded above; record which commits each merge brought in,
  // so build-index can fold them into the PR they came from.
  const merges = git(dir, 'log', 'HEAD', '--merges', ...sinceArgs, '--format=%H%x1f%P%x1f%s')
    .split('\n')
    .filter(Boolean)
    .map((line) => {
      const [sha, parents, subject] = line.split('\x1f');
      const [first, second] = parents.split(' ');
      const brought = second ? git(dir, 'rev-list', '--no-merges', `${first}..${second}`).split('\n').filter(Boolean) : [];
      return { sha, subject, parents: parents.split(' '), commits: brought };
    });

  return { head, commits, merges };
}

// ---------- one repo ----------

async function fetchRepo(r) {
  const raw = join('data/raw', repoKey(r));
  const sinceTime = r.since ? Date.parse(r.since) : -Infinity;
  const createdInRange = (item) => Date.parse(item.created_at) >= sinceTime;
  const ctx = {
    name: repoName(r),
    since: r.since,
    maxItems: r.maxItems,
    raw,
    cache: join(raw, 'cache'),
    api: `https://api.github.com/repos/${repoName(r)}`,
    sinceTime,
    cap: r.maxItems ?? Infinity,
    createdInRange,
    // Lists are sorted newest first, so stop once the oldest item on hand predates `since`.
    pastSince: (items) => items.length > 0 && !createdInRange(items[items.length - 1]),
    stats: { requests: 0, cached: 0, rateRemaining: 'n/a' },
  };
  mkdirSync(ctx.cache, { recursive: true });

  const t0 = Date.now();
  const { head, commits, merges } = fetchCommits(ctx);
  const pulls = await fetchPulls(ctx);
  const issues = await fetchIssues(ctx);
  const { byNumber: comments, total: commentsSeen } = await fetchComments(ctx, [...pulls, ...issues]);

  writeJson(join(raw, 'repo.json'), { owner: r.owner, repo: r.repo, head, since: r.since, maxItems: r.maxItems, fetched_at: new Date().toISOString() });
  writeJson(join(raw, 'commits.json'), commits);
  writeJson(join(raw, 'merges.json'), merges);
  writeJson(join(raw, 'pulls.json'), pulls);
  writeJson(join(raw, 'issues.json'), issues);
  writeJson(join(raw, 'comments.json'), comments);

  const kept = Object.values(comments).reduce((n, list) => n + list.length, 0);
  const { stats } = ctx;
  console.log(`
${ctx.name} @ ${head.slice(0, 7)}${r.since ? ` since ${r.since}` : ''}${r.maxItems ? ` (maxItems ${r.maxItems})` : ''}
  commits   ${commits.length} (+ ${merges.length} merge commits mapped)
  PRs       ${pulls.length} (${pulls.filter((p) => p.state === 'merged').length} merged)
  issues    ${issues.length}
  comments  ${kept} kept of ${commentsSeen} fetched
  API       ${stats.requests} requests, ${stats.cached} pages from cache, ${stats.rateRemaining} remaining
  time      ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}

const { repos } = parseRepoArgs();
for (const r of repos) await fetchRepo(r);
