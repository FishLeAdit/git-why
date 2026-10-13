// Fetch repo history into data/raw/ (gitignored).
//   commits:  blobless git clone + git log (no API calls)
//   PRs, issues, comments: GitHub REST API, list endpoints in bulk, every page cached to disk
// Re-running reuses the cache; delete data/raw/ to fetch fresh.
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { config } from '../src/lib/config.ts';

const { owner, repo, since, maxItems } = config;
const RAW = 'data/raw';
const CACHE = join(RAW, 'cache');
const API = `https://api.github.com/repos/${owner}/${repo}`;
// Pinned: 2026-03-10 removes merge_commit_sha from PR payloads, which build-index needs.
const API_VERSION = '2022-11-28';
const PER_PAGE = 100;
const CONCURRENCY = 4;

const token = process.env.GITHUB_TOKEN;
if (!token) console.warn('warning: GITHUB_TOKEN not set; anonymous limit is 60 requests/hour. See .env.example.');

const sinceTime = since ? Date.parse(since) : -Infinity;
const cap = maxItems ?? Infinity;
const stats = { requests: 0, cached: 0, rateRemaining: 'n/a' };

mkdirSync(CACHE, { recursive: true });

const readJson = (path) => JSON.parse(readFileSync(path, 'utf8'));
const writeJson = (path, data) => writeFileSync(path, JSON.stringify(data, null, 1));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------- GitHub API ----------

async function api(url, attempt = 0) {
  const res = await fetch(url, {
    headers: {
      Accept: 'application/vnd.github+json',
      'X-GitHub-Api-Version': API_VERSION,
      'User-Agent': 'git-why-fetch',
      ...(token && { Authorization: `Bearer ${token}` }),
    },
  });
  stats.requests++;
  stats.rateRemaining = res.headers.get('x-ratelimit-remaining') ?? stats.rateRemaining;

  if (res.ok) return res.json();

  const retryAfter = Number(res.headers.get('retry-after'));
  const reset = Number(res.headers.get('x-ratelimit-reset'));
  if ((res.status === 403 || res.status === 429) && attempt < 5) {
    const waitMs = retryAfter ? retryAfter * 1000 : res.headers.get('x-ratelimit-remaining') === '0' ? reset * 1000 - Date.now() + 1000 : 60_000;
    console.warn(`rate limited (${res.status}); waiting ${Math.ceil(waitMs / 1000)}s`);
    await sleep(Math.max(waitMs, 1000));
    return api(url, attempt + 1);
  }
  if (res.status >= 500 && attempt < 5) {
    await sleep(2 ** attempt * 1000);
    return api(url, attempt + 1);
  }
  throw new Error(`GET ${url} -> ${res.status} ${(await res.text()).slice(0, 200)}`);
}

// Fetch one page of `path` (relative to the repo API), or read it from the cache.
async function cachedPage(path, query, page) {
  const dir = join(CACHE, path.replaceAll('/', '-'));
  const file = join(dir, `${page}.json`);
  if (existsSync(file)) {
    stats.cached++;
    return readJson(file);
  }
  mkdirSync(dir, { recursive: true });
  const data = await api(`${API}/${path}?${query}&per_page=${PER_PAGE}&page=${page}`);
  writeJson(file, data);
  return data;
}

// Walk a list endpoint page by page. `done(items)` returns true once we have enough.
async function paginate(path, query, done = () => false) {
  const all = [];
  for (let page = 1; ; page++) {
    const items = await cachedPage(path, query, page);
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

const createdInRange = (item) => Date.parse(item.created_at) >= sinceTime;
// Lists are sorted newest first, so stop once the oldest item on hand predates `since`.
const pastSince = (items) => items.length > 0 && !createdInRange(items[items.length - 1]);

// ---------- PRs ----------

async function fetchPulls() {
  const pages = await paginate('pulls', 'state=all&sort=created&direction=desc', (all) => pastSince(all) || all.length >= cap);
  const pulls = pages.filter(createdInRange).slice(0, cap);

  const merged = pulls.filter((p) => p.merged_at);
  const files = await mapLimit(merged, CONCURRENCY, (p) => cachedPage(`pulls/${p.number}/files`, '', 1));
  const filesByPr = new Map(merged.map((p, i) => [p.number, files[i].map((f) => f.filename)]));

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
    // nnUNet often merges PRs locally ("Merge remote-tracking branch 'user/branch'"), so subjects
    // rarely say "#N". The PR's head commit and branch let build-index match those merges.
    head_sha: p.head.sha,
    head_ref: p.head.ref,
    head_owner: p.head.repo?.owner.login ?? null,
    labels: p.labels.map((l) => l.name),
    url: p.html_url,
    files: filesByPr.get(p.number) ?? [],
  }));
}

// ---------- Issues ----------

async function fetchIssues() {
  // The issues list also returns PRs (they carry a `pull_request` field); drop those.
  const isIssue = (i) => !i.pull_request;
  const pages = await paginate('issues', 'state=all&sort=created&direction=desc', (all) => pastSince(all) || all.filter(isIssue).length >= cap);
  return pages
    .filter(isIssue)
    .filter(createdInRange)
    .slice(0, cap)
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

async function fetchComments(wanted) {
  const sinceQuery = since ? `&since=${encodeURIComponent(new Date(sinceTime).toISOString())}` : '';
  const numberFromUrl = (url) => Number(url.slice(url.lastIndexOf('/') + 1));

  const issueComments = await paginate('issues/comments', `sort=created&direction=asc${sinceQuery}`);
  const reviewComments = await paginate('pulls/comments', `sort=created&direction=asc${sinceQuery}`);

  const byNumber = {};
  const add = (number, comment) => {
    if (!wanted.has(number)) return;
    (byNumber[number] ??= []).push(comment);
  };
  for (const c of issueComments) {
    add(numberFromUrl(c.issue_url), { kind: 'issue', author: c.user?.login ?? 'ghost', created_at: c.created_at, body: c.body ?? '' });
  }
  for (const c of reviewComments) {
    add(numberFromUrl(c.pull_request_url), { kind: 'review', author: c.user?.login ?? 'ghost', created_at: c.created_at, body: c.body ?? '', path: c.path });
  }
  for (const list of Object.values(byNumber)) list.sort((a, b) => a.created_at.localeCompare(b.created_at));
  return { byNumber, total: issueComments.length + reviewComments.length };
}

// ---------- Commits (local git, no API) ----------

const git = (dir, ...args) => execFileSync('git', ['-C', dir, ...args], { encoding: 'utf8', maxBuffer: 1 << 30 });

function fetchCommits() {
  const dir = join(RAW, 'repo.git');
  if (!existsSync(dir)) {
    console.log(`cloning ${owner}/${repo} (blobless)...`);
    execFileSync('git', ['clone', '--bare', '--filter=blob:none', '--quiet', `https://github.com/${owner}/${repo}.git`, dir], { stdio: 'inherit' });
  }
  const head = git(dir, 'rev-parse', 'HEAD').trim();
  const sinceArgs = since ? [`--since=${since}`] : [];

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

// ---------- main ----------

const t0 = Date.now();
const { head, commits, merges } = fetchCommits();
const pulls = await fetchPulls();
const issues = await fetchIssues();
const { byNumber: comments, total: commentsSeen } = await fetchComments(new Set([...pulls, ...issues].map((x) => x.number)));

writeJson(join(RAW, 'repo.json'), { owner, repo, head, since, maxItems, fetched_at: new Date().toISOString() });
writeJson(join(RAW, 'commits.json'), commits);
writeJson(join(RAW, 'merges.json'), merges);
writeJson(join(RAW, 'pulls.json'), pulls);
writeJson(join(RAW, 'issues.json'), issues);
writeJson(join(RAW, 'comments.json'), comments);

const kept = Object.values(comments).reduce((n, list) => n + list.length, 0);
console.log(`
${owner}/${repo} @ ${head.slice(0, 7)}${since ? ` since ${since}` : ''}${maxItems ? ` (maxItems ${maxItems})` : ''}
  commits   ${commits.length} (+ ${merges.length} merge commits mapped)
  PRs       ${pulls.length} (${pulls.filter((p) => p.state === 'merged').length} merged)
  issues    ${issues.length}
  comments  ${kept} kept of ${commentsSeen} fetched
  API       ${stats.requests} requests, ${stats.cached} pages from cache, ${stats.rateRemaining} remaining
  time      ${((Date.now() - t0) / 1000).toFixed(1)}s`);
