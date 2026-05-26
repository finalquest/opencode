#!/usr/bin/env bun

import { $ } from "bun"

const args = process.argv.slice(2)

function option(name: string, fallback: string) {
  const index = args.indexOf(`--${name}`)
  if (index === -1) return fallback
  const value = args[index + 1]
  if (!value || value.startsWith("--")) {
    throw new Error(`Missing value for --${name}`)
  }
  return value
}

function flag(name: string) {
  return args.includes(`--${name}`)
}

const branch = option("branch", "dev")
const source = option("source", "upstream")
const target = option("target", "origin")
const dryRun = flag("dry-run")

console.log(`Syncing ${source}/${branch} -> ${target}/${branch}`)

const remotes = (await $`git remote`.text())
  .split("\n")
  .map((line) => line.trim())
  .filter(Boolean)

if (!remotes.includes(source)) {
  throw new Error(`Remote not found: ${source}`)
}

if (!remotes.includes(target)) {
  throw new Error(`Remote not found: ${target}`)
}

const status = (await $`git status --short`.text()).trim()
if (status) {
  throw new Error("Worktree is not clean. Commit or stash your changes before syncing.")
}

await $`git fetch ${source} ${branch}`
await $`git fetch ${target} ${branch}`
await $`git checkout ${branch}`
await $`git reset --hard refs/remotes/${target}/${branch}`

const merge = await $`git merge --no-edit refs/remotes/${source}/${branch}`.nothrow()
if (merge.exitCode !== 0) {
  throw new Error(`Merge failed in the current repo. Resolve conflicts on ${branch}, then push ${target} ${branch}.`)
}

if (dryRun) {
  await $`git push --dry-run ${target} HEAD:refs/heads/${branch}`
  console.log("Dry run completed")
  process.exit(0)
}

await $`git push ${target} HEAD:refs/heads/${branch}`
console.log("Sync completed")
