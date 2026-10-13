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

## Look

A Windows 98 window, to match [my portfolio](https://fishleadit.github.io/). The styling is plain CSS in `src/style.css`. The Titillium Web font is under the SIL Open Font License (`src/assets/fonts/OFL.txt`), and the Windows 98 cursors come from the portfolio, which is built on [98.portfolio](https://github.com/AlexandreDresch/98.portfolio) by Alexandre Dresch.

## Rebuild an index

Needs `GITHUB_TOKEN` in `.env`. Without `--repo`, every repo in `config.json` is processed.

```sh
npm run fetch -- --repo huggingface/transformers.js
npm run index -- --repo huggingface/transformers.js
npm run query -- --repo huggingface/transformers.js "why is q8 the default on wasm"
```
