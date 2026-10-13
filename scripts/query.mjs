// Debug helper: print the top results for a question against public/data/.
//   npm run query -- "why was the default patch size changed"
import { readFileSync } from 'node:fs';
import { embedQuery, loadEmbedder } from '../src/lib/embed.ts';
import { parseIndex, searchDense } from '../src/lib/retrieval.ts';

const question = process.argv.slice(2).join(' ');
if (!question) {
  console.error('usage: npm run query -- "<question>"');
  process.exit(1);
}
const read = (name) => readFileSync(`public/data/${name}`);
const bin = read('vectors.bin');
const index = parseIndex(JSON.parse(read('manifest.json')), JSON.parse(read('meta.json')), bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));

const embed = await loadEmbedder();
const hits = searchDense(index, await embedQuery(embed, question), 5);
console.log(`\n${question}\n`);
for (const { chunk, score } of hits) {
  console.log(`${score.toFixed(3)}  ${chunk.id.padEnd(15)} ${chunk.date.slice(0, 10)}  ${chunk.title.slice(0, 80)}`);
  console.log(`       ${chunk.text.replace(/\s+/g, ' ').slice(0, 140)}`);
}
