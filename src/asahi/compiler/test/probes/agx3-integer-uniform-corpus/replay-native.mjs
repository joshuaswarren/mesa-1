// SPDX-License-Identifier: MIT
// Replay captured Apple instructions through the independent probe encoder.
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';

const [directory, encoder] = process.argv.slice(2);
if (!directory || !encoder) throw new Error('usage: replay-native.mjs RESULTS ENCODE_CLI');
const data = JSON.parse(fs.readFileSync(path.join(directory, 'integer-witnesses.json')));
const ops = ['iadd', 'isub', 'imadd', 'bit_and', 'bit_or', 'bit_xor',
  'bit_nand', 'bit_nor', 'bit_andc'];
let matched = 0, excluded = 0;
const cache = new Map();
for (const shader of data.shaders) {
  for (const instruction of shader.witnesses) {
    const op = ops.indexOf(instruction.mnemonic);
    if (op < 0) { ++excluded; continue; }
    const form = instruction.forms[0];
    const fields = new Map([...form.matchAll(/ (\d+):(\d+)/g)].map(m => [+m[1], +m[2]]));
    const f = n => fields.get(n) ?? 0;
    const width = f(10), bitwise = op >= 3, mad = op === 2;
    // Main GPR destinations, matching widths, one varying binary operand.
    if (f(9) !== 2 || ![1, 2].includes(width)) { ++excluded; continue; }
    const sources = [];
    let supported = true;
    for (let i = 0; i < (mad ? 3 : 2); ++i) {
      const base = 15 + i * (bitwise ? 14 : 15), kind = f(base + 8);
      const value = f(base + kind - 1);
      if (![2, 3].includes(kind) && !(mad && i === 2 && kind === 1 && value === 0))
        supported = false;
      if (f(base + 9) !== (kind === 1 ? 1 : width)) supported = false;
      sources.push([kind, value, f(base + 10), f(base + 12)]);
    }
    // Commutative forms intentionally canonicalize to U/R. Exclude Apple's
    // R/U bitops here instead of treating a different equivalent encoding as
    // an exact replay. SUB supports both witnessed orders without swapping.
    if (!(sources[0][0] === 3 && sources[1][0] === 2) &&
        !(op === 1 && sources[0][0] === 2 && sources[1][0] === 3)) supported = false;
    if (!supported) { ++excluded; continue; }
    const shift = op === 0 ? f(54) : 0;
    if (shift !== 0 && shift !== 3) { ++excluded; continue; }
    const waitBase = mad ? 60 : bitwise ? 44 : 45;
    let wait = 0;
    for (let s = 0; s < 6; ++s) wait |= f(waitBase + s) << s;
    if (!mad) sources.push([1, 0, 0, 0]);
    const args = [op, width === 1 ? 1 : 0, f(2), f(11), wait, shift, ...sources.flat()].map(String);
    const key = args.join(',');
    if (!cache.has(key)) {
      const p = spawnSync(encoder, args, {encoding: 'utf8'});
      if (p.status !== 0) throw new Error(`${shader.name}@${instruction.offset}: rejected (${key})`);
      cache.set(key, p.stdout.trim());
    }
    if (cache.get(key) !== instruction.hex)
      throw new Error(`${shader.name}@${instruction.offset}: expected ${instruction.hex}, got ${cache.get(key)}`);
    ++matched;
  }
}
if (!matched) throw new Error('no supported witnesses replayed');
console.log(JSON.stringify({matched, uniqueDescriptors: cache.size, excluded}));
