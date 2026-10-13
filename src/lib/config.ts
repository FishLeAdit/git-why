// Single source of truth for settings shared by the build scripts, the web app and the eval.
// Values live in config.json; this module gives them types.
import raw from '../../config.json' with { type: 'json' };

export type NoiseRules = {
  botAuthors: string[]; // PRs, issues and commits by these logins are dropped
  dropBotComments: boolean; // drop comments from any "*[bot]" login (e.g. AI assistant replies)
  skipSubjects: string[]; // regex sources, case-insensitive; standalone commits matching one are dropped
};

export type RepoConfig = {
  owner: string;
  repo: string;
  since: string | null; // ISO date; only fetch history after this
  maxItems: number | null; // cap per item type, for small first runs
  examples: string[]; // example questions shown in the UI
  // Below this top dense cosine, the answer step should say the history doesn't explain the question.
  // Measured by `npm run eval` (eval/results.md); null where no eval has been run for this repo.
  answerThreshold: number | null;
  noise?: Partial<NoiseRules>; // added to the global rules (lists are extended, not replaced)
};

export type Config = {
  repos: RepoConfig[]; // the first one is the default in the UI
  model: string;
  dtype: 'fp32' | 'fp16' | 'q8' | 'int8' | 'uint8' | 'q4' | 'bnb4' | 'q4f16';
  pooling: 'cls' | 'mean';
  topK: number;
  noise: NoiseRules;
};

export const config = raw as Config;

export const EMBED_MODEL = config.model;
export const EMBED_DTYPE = config.dtype;
// bge was trained with CLS pooling (see BAAI/bge-small-en-v1.5 1_Pooling/config.json),
// even though the Xenova model card example shows mean pooling.
export const EMBED_POOLING = config.pooling;
export const EMBED_DIM = 384;

// bge expects this prefix on queries only, never on documents.
export const QUERY_PREFIX = 'Represent this sentence for searching relevant passages: ';

// "MIC-DKFZ/nnUNet" -> "MIC-DKFZ__nnUNet": the folder name under data/raw/ and public/data/.
export const repoKey = (r: { owner: string; repo: string }) => `${r.owner}__${r.repo}`;
export const repoName = (r: { owner: string; repo: string }) => `${r.owner}/${r.repo}`;

export function noiseFor(r: RepoConfig): NoiseRules {
  return {
    botAuthors: [...config.noise.botAuthors, ...(r.noise?.botAuthors ?? [])],
    dropBotComments: r.noise?.dropBotComments ?? config.noise.dropBotComments,
    skipSubjects: [...config.noise.skipSubjects, ...(r.noise?.skipSubjects ?? [])],
  };
}
