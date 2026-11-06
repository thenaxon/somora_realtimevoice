/* R2 / F12 correction C1 — nothing may be awaited between the Talk click and
 * `session.start()`.
 *
 * Regression: `7bd833e` added `await refreshProfiles()` (a network GET) before
 * `await session.start(model)` in the Test Console's Talk handler. `session.start`
 * creates the AudioContext and requests the microphone; awaiting a fetch first
 * moves that out of the click's user activation, and Safari/iOS can then refuse
 * or suspend audio. Headless Chromium does not catch that. The P0-accepted Talk
 * start (`p0-ref-b052ca5`) called `session.start` without a preceding network
 * await.
 *
 * This is a source-order guard: it reads the SHIPPED module and asserts that in
 * the Talk (start) branch the first `await` is `session.start(...)` and that no
 * profile-refresh / network call is awaited before it. Run:
 *   node tests/ui_talk_start.mjs
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(join(here, "..", "src", "realtimevoice", "web", "ui", "view-test.js"), "utf8");

const start = src.indexOf("async function toggleTalk()");
assert.ok(start >= 0, "toggleTalk() not found in view-test.js");
// the function is followed by the next top-level `  function ` / `  async function `
let end = src.indexOf("\n  function ", start + 1);
const endAsync = src.indexOf("\n  async function ", start + 1);
if (end < 0 || (endAsync >= 0 && endAsync < end)) end = endAsync;
assert.ok(end > start, "could not delimit toggleTalk()");
const body = src.slice(start, end);

// the stop branch (state !== idle) returns early; the start branch is what
// follows. Isolate from the idle guard onward.
const startBranchAt = body.indexOf('if (state.stateName !== "idle")');
assert.ok(startBranchAt >= 0, "idle guard not found");
const startBranch = body.slice(startBranchAt);

// (1) the first await in the start branch is session.start(...)
const firstAwait = startBranch.indexOf("await session.start(");
assert.ok(firstAwait >= 0, "the start branch never awaits session.start");
const beforeFirstAwait = startBranch.slice(0, firstAwait);

assert.ok(
  !/await\s+refreshProfiles\s*\(/.test(beforeFirstAwait),
  "C1: `await refreshProfiles()` (a network GET) must not precede session.start",
);
assert.ok(
  !/await\s+fetch\s*\(/.test(beforeFirstAwait),
  "C1: no awaited fetch may precede session.start",
);
assert.ok(
  !/await\s+ctx\.loadProfiles\s*\(/.test(beforeFirstAwait),
  "C1: no awaited profile load may precede session.start",
);
// no other awaited network call before start (api.* is the REST client)
assert.ok(
  !/await\s+profilesApi\.|await\s+voicesApi\.|await\s+api\./.test(beforeFirstAwait),
  "C1: no awaited REST call may precede session.start",
);

// (2) session.start is the first `await` at all in the branch, so the
// AudioContext/getUserMedia stay inside the click's user activation.
const allAwaits = [...startBranch.matchAll(/await\s+([A-Za-z0-9_.$]+)\s*\(/g)].map((m) => m[1]);
assert.equal(allAwaits[0], "session.start", `first await must be session.start, got ${allAwaits[0]}`);

// (3) the stale-revision refresh must still exist, but fire-and-forget after
// the session started (or on picker focus / visibility) — never awaited before.
assert.ok(/refreshProfiles/.test(src), "the profile refresh must not be removed entirely");

console.log("ok - C1: nothing is awaited between the Talk click and session.start()");
console.log("ok - C1: the profile refresh is not awaited before session.start()");
console.log("\nTALK-START TESTS PASSED (2)");