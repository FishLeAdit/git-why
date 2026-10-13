// Debug helper: print the top results for a question against a built index.
//   npm run query -- "why was the default patch size changed"
//   npm run query -- --repo huggingface/transformers.js "why is q8 the default on wasm"
// Without --repo it uses the first repo in config.json.
import { readFileSync } from 'node:fs';
import { repoKey, repoName } from '../src/lib/config.ts';
import { embedQuery, loadEmbedder } from '../src/lib/embed.ts';
import { parseIndex, searchDense } from '../src/lib/retrieval.ts';
import { parseRepoArgs } from './repos.mjs';

const { repos, rest } = parseRepoArgs();
const question = rest.join(' ');
if (!question) {
  console.error('usage: npm run query -- [--repo owner/name] "<question>"');
  process.exit(1);
}
const repo = repos[0];
const read = (name) => readFileSync(`public/data/${repoKey(repo)}/${name}`);
const bin = read('vectors.bin');
const index = parseIndex(JSON.parse(read('manifest.json')), JSON.parse(read('meta.json')), bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));

const embed = await loadEmbedder();
const hits = searchDense(index, await embedQuery(embed, question), 5);
console.log(`\n${repoName(repo)}: ${question}\n`);
for (const { chunk, score } of hits) {
  console.log(`${score.toFixed(3)}  ${chunk.id.padEnd(15)} ${chunk.date.slice(0, 10)}  ${chunk.title.slice(0, 80)}`);
  console.log(`       ${chunk.text.replace(/\s+/g, ' ').slice(0, 140)}`);
}
