// SPDX-License-Identifier: MIT
// Usage: node extract.mjs RESULTS_DIRECTORY [AGX3B_DISASSEMBLER]
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawnSync} from 'node:child_process';

const [directory, decoder] = process.argv.slice(2);
if (!directory) throw new Error('missing results directory');
const magic = Buffer.from('cffaedfe', 'hex');
const summary = [];
for (const arch of ['applegpu_g17p', 'applegpu_g16g']) {
  const dir = path.join(directory, arch);
  for (const name of fs.readdirSync(dir).filter(n => n.endsWith('.bin')).sort()) {
    const bytes = fs.readFileSync(path.join(dir, name));
    const texts = [];
    for (let start = bytes.indexOf(magic); start >= 0;
         start = bytes.indexOf(magic, start + 4)) {
      if (start + 32 > bytes.length) continue;
      const count = bytes.readUInt32LE(start + 16);
      const end = start + 32 + bytes.readUInt32LE(start + 20);
      if (end > bytes.length || count > (end - start - 32) / 8) continue;
      let cursor = start + 32;
      for (let i = 0; i < count; i++) {
        if (cursor + 8 > end) throw new Error(`${name}: truncated load command`);
        const cmd = bytes.readUInt32LE(cursor);
        const size = bytes.readUInt32LE(cursor + 4);
        if (size < 8 || cursor + size > end) throw new Error(`${name}: bad load command`);
        if (cmd === 0x19) {
          if (size < 72) throw new Error(`${name}: short segment`);
          const sections = bytes.readUInt32LE(cursor + 64);
          if (72 + sections * 80 > size) throw new Error(`${name}: bad section count`);
          for (let s = cursor + 72; s < cursor + 72 + sections * 80; s += 80) {
            const section = bytes.subarray(s, s + 16).toString().replace(/\0.*$/, '');
            if (section !== '__text') continue;
            const length = Number(bytes.readBigUInt64LE(s + 40));
            const offset = start + bytes.readUInt32LE(s + 48);
            if (!Number.isSafeInteger(length) || offset + length > bytes.length)
              throw new Error(`${name}: invalid text extent`);
            texts.push(bytes.subarray(offset, offset + length));
          }
        }
        cursor += size;
      }
    }
    if (texts.length !== 1) throw new Error(`${name}: expected one text, got ${texts.length}`);
    const text = texts[0];
    const output = path.join(dir, name.replace(/\.bin$/, '.text'));
    fs.writeFileSync(output, text);
    const row = {arch, name: name.slice(0, -4), bytes: text.length,
      sha256: crypto.createHash('sha256').update(text).digest('hex')};
    if (decoder) {
      const result = spawnSync(decoder, [output], {encoding: 'utf8'});
      fs.writeFileSync(output + '.decoded', (result.stdout ?? '') + (result.stderr ?? ''));
      row.decodeStatus = result.status;
    }
    summary.push(row);
  }
}
fs.writeFileSync(path.join(directory, 'summary.json'), JSON.stringify(summary, null, 2) + '\n');
console.log(`Extracted ${summary.length} native shaders; ${summary.filter(r => r.decodeStatus !== undefined && r.decodeStatus !== 0).length} decoder failures.`);
