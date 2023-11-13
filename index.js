import { pipeline } from '@xenova/transformers';

// Allocate a pipeline for sentiment-analysis
let pipe = await pipeline('sentiment-analysis');

let out = await pipe('I love transformers!');
// [{'label': 'POSITIVE', 'score': 0.999817686}]
console.log(out)

let extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
let output = await extractor('This is a simple test.', { pooling: 'mean', normalize: true });

console.log(output)