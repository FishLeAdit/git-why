import './style.css';
import { config } from './lib/config.ts';

const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `<h1>git-why</h1><p>Ask why <code>${config.owner}/${config.repo}</code> changed.</p>`;
