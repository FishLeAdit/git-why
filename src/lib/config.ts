// Single source of truth for settings shared by the build scripts, the web app and the eval.
// Values live in config.json; this module gives them types.
import raw from '../../config.json' with { type: 'json' };

export type Config = {
  owner: string;
  repo: string;
  since: string | null; // ISO date; only fetch history after this
  maxItems: number | null; // cap per item type, for small first runs
  model: string;
  dtype: string;
  topK: number;
  noise: {
    botAuthors: string[];
    skipSubjects: string[]; // regex sources, matched case-insensitively
  };
};

export const config: Config = raw;

export const EMBED_MODEL = config.model;
export const EMBED_DTYPE = config.dtype;
export const EMBED_DIM = 384;

// bge expects this prefix on queries only, never on documents.
export const QUERY_PREFIX = 'Represent this sentence for searching relevant passages: ';
