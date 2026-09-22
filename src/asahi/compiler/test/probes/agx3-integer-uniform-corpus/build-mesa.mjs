// SPDX-License-Identifier: MIT
// Reuse a configured native test build without changing production targets.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';

const [buildArg, outputArg] = process.argv.slice(2);
if (!buildArg || !outputArg)
  throw new Error('usage: node build-mesa.mjs NATIVE_BUILD OUTPUT_DIRECTORY');
const build = path.resolve(buildArg), output = path.resolve(outputArg);
const source = path.join(path.dirname(fileURLToPath(import.meta.url)), 'mesa-corpus.c');
fs.mkdirSync(output, {recursive: true});
const target = 'src/asahi/compiler/agx3_optimization_corpus_test';
execFileSync('ninja', ['-C', build, target], {stdio: 'inherit'});
const entries = JSON.parse(fs.readFileSync(path.join(build, 'compile_commands.json')));
const entry = entries.find(e => e.file.endsWith('/test-agx3-optimization-corpus.c'));
if (!entry?.command) throw new Error('missing arithmetic corpus compile command');
const quote = s => "'" + s.replaceAll("'", "'\\''") + "'";
const object = path.join(output, 'mesa-corpus.o');
let compile = entry.command;
if (!/ -o \S+/.test(compile) || !/ -c \S+$/.test(compile))
  throw new Error('unsupported compile command shape');
compile = compile.replace(/ -o \S+/, ' -o ' + quote(object))
  .replace(/ -c \S+$/, ' -c ' + quote(source))
  .replace(/ -MF \S+/, ' -MF ' + quote(object + '.d'))
  .replace(/ -MQ \S+/, ' -MQ ' + quote(object));
execFileSync('sh', ['-c', compile], {cwd: entry.directory, stdio: 'inherit'});
let link = execFileSync('ninja', ['-C', build, '-t', 'commands', target],
  {encoding: 'utf8', maxBuffer: 64 * 1024 * 1024}).trim().split('\n').at(-1);
const oldObject = target + '.p/test_test-agx3-optimization-corpus.c.o';
if (!link.includes('-o ' + target + ' ') || !link.includes(oldObject))
  throw new Error('unsupported link command shape');
const executable = path.join(output, 'mesa-corpus');
link = link.replace('-o ' + target + ' ', '-o ' + quote(executable) + ' ')
  .replace(oldObject, quote(object));
execFileSync('sh', ['-c', link], {cwd: build, stdio: 'inherit'});
console.log(executable);
