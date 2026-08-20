# Fixture provenance

Both fixtures are trimmed to exactly **8,000 characters** so that a
token-count comparison between them reflects content rather than length.

## prose.txt

- Source: *Alice's Adventures in Wonderland* by Lewis Carroll
- Project Gutenberg eBook #11, <https://www.gutenberg.org/cache/epub/11/pg11.txt>
- Public domain in the United States.
- Extracted from the start of the narrative, then trimmed to 8,000 characters.
- sha256: `5afa8cb58efa0cadcee88b6f2ae18a2e74de68e6d220e6428569b46995851234`

## code.js

- Source: this repository's own MIT-licensed source, concatenated in this order:
  - `src/hf.js`
  - `src/sigv4.js`
  - `src/registry.js`
  - `src/report.js`
- A frozen snapshot taken when the published numbers were measured. Later edits
  to those source files deliberately do not change this fixture; the hash below
  is the pin, and `pnpm run fixtures` fails if the bytes drift from it.
- Trimmed to 8,000 characters.
- sha256: `2e8ffbb5cf63f3dbd16196dcd1e09caa68305eef51932232a250123fa77f40ce`

## Regenerating

```bash
pnpm run fixtures              # verify both fixtures against this provenance
pnpm run fixtures -- --update  # re-freeze from the current sources
```

The fixtures are inputs to `test/reference.json` and to every number quoted
in the README. After `--update`, regenerate the reference counts with
`pnpm run verify` and re-measure the README tables (the Claude rows need
Bedrock access: `pnpm run audit:bedrock`).
