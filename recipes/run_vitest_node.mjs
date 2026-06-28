#!/usr/bin/env node
// Hive runtime red→green test-node runner for VITEST targets (lever L3 enabler).
//
// `hive/verify.py:run_test_node` launches `command + [node_id]`, where node_id is the
// specify-named red-test node in Hive's neutral `<file>::<testName>` form (e.g.
// `src/test/mentioncopy_race.test.ts::test_mentioncopy_last_write_wins_over_stale_async`,
// already rebased relative to the runner cwd). pytest accepts that form verbatim; vitest does
// not — it wants the file as a positional filter and the test name via `-t`. This thin,
// codebase-neutral shim does that translation so the SAME closed-loop verify that drives the
// HTTP-shape / write-sink levers (pytest) also drives the overwrite-race oracle (vitest).
//
// Exit code is forwarded verbatim: 0 = all assertions held (GREEN), 1 = a test failed (a
// legitimate RED) — exactly the convention `classify_returncode` expects.

import { spawnSync } from 'node:child_process'

const node = process.argv[process.argv.length - 1] || ''
const sep = node.indexOf('::')
const file = sep === -1 ? node : node.slice(0, sep)
const name = sep === -1 ? '' : node.slice(sep + 2)

const args = ['vitest', 'run']
if (file) args.push(file)
if (name) args.push('-t', name)

// `npx` resolves the target package's local vitest; cwd is the runner's configured cwd
// (the client/ package). shell:true so `npx`/`npx.cmd` resolves on Windows too.
const r = spawnSync('npx', args, { stdio: 'inherit', shell: true })
process.exit(r.status === null ? 1 : r.status)
