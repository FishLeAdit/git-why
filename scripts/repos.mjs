// `--repo owner/name` (or `--repo=owner/name`) picks one repo from config.json; without it, all repos.
import { config, repoName } from '../src/lib/config.ts';

export function parseRepoArgs(argv = process.argv.slice(2)) {
  const rest = [];
  let name = null;
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--repo') name = argv[++i];
    else if (argv[i].startsWith('--repo=')) name = argv[i].slice('--repo='.length);
    else rest.push(argv[i]);
  }
  if (name === null) return { repos: config.repos, rest };
  const repo = config.repos.find((r) => repoName(r).toLowerCase() === String(name).toLowerCase());
  if (!repo) {
    console.error(`unknown repo "${name}". Add it to "repos" in config.json first. Known: ${config.repos.map(repoName).join(', ')}`);
    process.exit(1);
  }
  return { repos: [repo], rest };
}
