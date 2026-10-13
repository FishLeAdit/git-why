# git-why

**Live: https://fishleadit.github.io/git-why/**

Ask *why* an open-source repo changed and get the PRs, issues and commits behind it. Search runs entirely in your browser: no backend, no API key, and it keeps working offline after the first visit.

Indexed repositories (pick one in the page; set in `config.json`):
- [`MIC-DKFZ/nnUNet`](https://github.com/MIC-DKFZ/nnUNet), history since 2022
- [`huggingface/transformers.js`](https://github.com/huggingface/transformers.js), history since 2022

> Work in progress: semantic search is live; written answers from a local in-browser model are planned.

## Develop

```sh
npm install
npm run dev
```

Rebuild an index (needs `GITHUB_TOKEN` in `.env`; without `--repo`, every repo in `config.json`):

```sh
npm run fetch -- --repo huggingface/transformers.js
npm run index -- --repo huggingface/transformers.js
npm run query -- --repo huggingface/transformers.js "why is q8 the default on wasm"
```
