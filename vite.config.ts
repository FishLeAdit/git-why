import { defineConfig } from 'vite';

export default defineConfig({
  // The embedding worker imports transformers.js, which needs code splitting (ES module workers).
  worker: { format: 'es' },
});
