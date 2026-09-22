// SPDX-License-Identifier: MIT
// Usage: node audit.mjs RESULTS_DIRECTORY STRUCTURAL_DECOMPOSER
// The decoder supplies candidate boundaries only. A recorded witness must
// match the original bytes and recompose exactly through the AGX3B tables.
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';

const [directory, decomposer] = process.argv.slice(2);
if (!directory || !decomposer) throw new Error('missing arguments');
const metadata = JSON.parse(fs.readFileSync(path.join(directory, 'summary.json')));
const selected = new Set(['iadd', 'isub', 'imadd', 'bit_and', 'bit_or',
  'bit_xor', 'bit_nand', 'bit_nor', 'bit_xnor', 'bit_andc',
  'inv', 'isel', 'sll', 'srl', 'sra']);
const cache = new Map();
const rows = [];
for (const shader of metadata) {
  const prefix = path.join(directory, shader.arch, shader.name + '.text');
  const bytes = fs.readFileSync(prefix);
  const decoded = fs.readFileSync(prefix + '.decoded', 'utf8');
  const witnesses = [], unmatched = [];
  for (const line of decoded.split('\n')) {
    const match = /^\s*([0-9a-f]+):\s+([0-9a-f]+)\s+(\w+)\b/.exec(line);
    if (!match || !selected.has(match[3])) continue;
    const [, offsetHex, hex, mnemonic] = match;
    const offset = parseInt(offsetHex, 16);
    if (bytes.subarray(offset, offset + hex.length / 2).toString('hex') !== hex)
      throw new Error(`${shader.name}: candidate bytes disagree at ${offsetHex}`);
    const key = `${mnemonic}:${hex}`;
    if (!cache.has(key)) {
      const p = spawnSync(decomposer, [hex, mnemonic], {encoding: 'utf8'});
      if (p.status !== 0) throw new Error(`decomposer failed: ${key}`);
      const lines = p.stdout.split('\n');
      let pending = [], matches = [];
      for (const l of lines) {
        if (l.startsWith('MATCH ')) pending.push(l);
        if (l.includes('^ agx3b')) { matches.push(...pending); pending = []; }
        if (l.includes('^ agx3c')) pending = [];
      }
      cache.set(key, [...new Set(matches)]);
    }
    const forms = cache.get(key);
    const record = {offset, hex, mnemonic, forms};
    if (forms.length) witnesses.push(record); else unmatched.push(record);
  }
  rows.push({...shader, witnesses, unmatched});
}
fs.writeFileSync(path.join(directory, 'integer-witnesses.json'),
  JSON.stringify({note: 'Exact structural round trips; no execution validation. '
    + 'Candidate boundaries come from a decoder with known failures.', shaders: rows}, null, 2) + '\n');
console.log(JSON.stringify({shaders: rows.length,
  roundTrips: rows.reduce((n, r) => n + r.witnesses.length, 0),
  unmatchedCandidates: rows.reduce((n, r) => n + r.unmatched.length, 0),
  distinctCandidates: cache.size}));
