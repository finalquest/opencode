import { afterEach, describe, expect } from "bun:test"
import { CrossSpawnSpawner } from "@opencode-ai/core/cross-spawn-spawner"
import { Effect, Layer } from "effect"
import { Instance, InstanceStore, instanceStoreDefaultLayer } from "../../src/project/instance"
import { tmpdirScoped } from "../fixture/fixture"
import { testEffect } from "../lib/effect"

const it = testEffect(Layer.mergeAll(instanceStoreDefaultLayer, CrossSpawnSpawner.defaultLayer))

afterEach(async () => {
  await Instance.disposeAll()
})

describe("InstanceStore", () => {
  it.live("loads instance context without installing ALS for the caller", () =>
    Effect.gen(function* () {
      const dir = yield* tmpdirScoped({ git: true })
      const store = yield* InstanceStore
      const ctx = yield* store.load({ directory: dir })

      expect(ctx.directory).toBe(dir)
      expect(ctx.worktree).toBe(dir)
      expect(() => Instance.current).toThrow()
    }),
  )

  it.live("runs load init inside the loaded legacy instance context", () =>
    Effect.gen(function* () {
      const dir = yield* tmpdirScoped({ git: true })
      const store = yield* InstanceStore
      let initializedDirectory: string | undefined

      yield* store.load({
        directory: dir,
        init: async () => {
          initializedDirectory = Instance.directory
        },
      })

      expect(initializedDirectory).toBe(dir)
      expect(() => Instance.current).toThrow()
    }),
  )

  it.live("caches loaded instance context by directory", () =>
    Effect.gen(function* () {
      const dir = yield* tmpdirScoped({ git: true })
      const store = yield* InstanceStore
      let initialized = 0

      const first = yield* store.load({
        directory: dir,
        init: async () => {
          initialized++
        },
      })
      const second = yield* store.load({
        directory: dir,
        init: async () => {
          initialized++
        },
      })

      expect(second).toBe(first)
      expect(initialized).toBe(1)
    }),
  )

  it.live("keeps Instance.provide as the legacy ALS wrapper", () =>
    Effect.gen(function* () {
      const dir = yield* tmpdirScoped({ git: true })

      const directory = yield* Effect.promise(() =>
        Instance.provide({
          directory: dir,
          fn: () => Instance.directory,
        }),
      )

      expect(directory).toBe(dir)
      expect(() => Instance.current).toThrow()
    }),
  )
})
