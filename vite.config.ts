import { defineConfig } from 'vite';

// GitHub Pages serves a project repo under /<repo>/ and a <name>.github.io repo at /.
// In GitHub Actions GITHUB_REPOSITORY is "owner/repo"; locally it is unset, so the site is served at /.
const repo = process.env.GITHUB_REPOSITORY?.split('/')[1];
const base = !repo || repo.endsWith('.github.io') ? '/' : `/${repo}/`;

export default defineConfig({
  base,
  // The embedding worker imports transformers.js, which needs code splitting (ES module workers).
  worker: { format: 'es' },
});
