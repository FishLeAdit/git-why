// Runs WebLLM off the main thread. The page talks to it through CreateWebWorkerMLCEngine (src/answers.ts).
// This worker is only created after the visitor turns answers on, so search-only visits never load it.
import { WebWorkerMLCEngineHandler } from '@mlc-ai/web-llm';

const handler = new WebWorkerMLCEngineHandler();
self.onmessage = (event: MessageEvent) => handler.onmessage(event);
