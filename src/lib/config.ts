// Single source of truth for settings shared by the build scripts, the web app and the eval.
// Values live in config.json; this module gives them types.
import raw from '../../config.json' with { type: 'json' };

export type Config = {
  owner: string;
  repo: string;
  since: string | null; // ISO date; only fetch history after this
  maxItems: number | null; // cap per item type, for small first runs
  model: string;
  dtype: 'fp32' | 'fp16' | 'q8' | 'int8' | 'uint8' | 'q4' | 'bnb4' | 'q4f16';
  pooling: 'cls' | 'mean';
  topK: number;
  noise: {
    botAuthors: string[]; // PRs, issues and commits by these logins are dropped
    dropBotComments: boolean; // drop comments from any "*[bot]" login (e.g. AI assistant replies)
    skipSubjects: string[]; // regex sources, case-insensitive; standalone commits matching one are dropped
  };
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
