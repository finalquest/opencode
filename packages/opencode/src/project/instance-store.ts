import { GlobalBus } from "@/bus/global"
import { WorkspaceContext } from "@/control-plane/workspace-context"
import { disposeInstance } from "@/effect/instance-registry"
import { makeRuntime } from "@/effect/run-service"
import { AppFileSystem } from "@opencode-ai/core/filesystem"
import * as Log from "@opencode-ai/core/util/log"
import { Context, Effect, Layer } from "effect"
import { iife } from "@/util/iife"
import { context, type InstanceContext } from "./instance-context"
import * as Project from "./project"

export interface LoadInput {
  directory: string
  init?: () => Promise<unknown>
  worktree?: string
  project?: Project.Info
}

export interface Interface {
  readonly load: (input: LoadInput) => Effect.Effect<InstanceContext>
  readonly reload: (input: LoadInput) => Effect.Effect<InstanceContext>
  readonly dispose: (ctx: InstanceContext) => Effect.Effect<void>
  readonly disposeAll: () => Effect.Effect<void>
}

export class Service extends Context.Service<Service, Interface>()("@opencode/InstanceStore") {}

export const layer: Layer.Layer<Service, never, Project.Service> = Layer.effect(
  Service,
  Effect.gen(function* () {
    const project = yield* Project.Service
    const cache = new Map<string, Promise<InstanceContext>>()
    const disposal = {
      all: undefined as Promise<void> | undefined,
    }

    const boot = Effect.fn("InstanceStore.boot")(function* (input: LoadInput & { directory: string }) {
      const ctx =
        input.project && input.worktree
          ? {
              directory: input.directory,
              worktree: input.worktree,
              project: input.project,
            }
          : yield* project.fromDirectory(input.directory).pipe(
              Effect.map((result) => ({
                directory: input.directory,
                worktree: result.sandbox,
                project: result.project,
              })),
            )
      const init = input.init
      if (init) yield* Effect.promise(() => context.provide(ctx, init))
      return ctx
    })

    function track(directory: string, next: Promise<InstanceContext>) {
      const task = next.catch((error) => {
        if (cache.get(directory) === task) cache.delete(directory)
        throw error
      })
      cache.set(directory, task)
      return task
    }

    const load = Effect.fn("InstanceStore.load")(function* (input: LoadInput) {
      const directory = AppFileSystem.resolve(input.directory)
      const existing = cache.get(directory)
      if (existing) return yield* Effect.promise(() => existing)

      Log.Default.info("creating instance", { directory })
      return yield* Effect.promise(() => track(directory, Effect.runPromise(boot({ ...input, directory }))))
    })

    const reload = Effect.fn("InstanceStore.reload")(function* (input: LoadInput) {
      const directory = AppFileSystem.resolve(input.directory)
      Log.Default.info("reloading instance", { directory })
      yield* Effect.promise(() => disposeInstance(directory))
      cache.delete(directory)
      const next = track(directory, Effect.runPromise(boot({ ...input, directory })))

      GlobalBus.emit("event", {
        directory,
        project: input.project?.id,
        workspace: WorkspaceContext.workspaceID,
        payload: {
          type: "server.instance.disposed",
          properties: {
            directory,
          },
        },
      })

      return yield* Effect.promise(() => next)
    })

    const dispose = Effect.fn("InstanceStore.dispose")(function* (ctx: InstanceContext) {
      Log.Default.info("disposing instance", { directory: ctx.directory })
      yield* Effect.promise(() => disposeInstance(ctx.directory))
      cache.delete(ctx.directory)

      GlobalBus.emit("event", {
        directory: ctx.directory,
        project: ctx.project.id,
        workspace: WorkspaceContext.workspaceID,
        payload: {
          type: "server.instance.disposed",
          properties: {
            directory: ctx.directory,
          },
        },
      })
    })

    const disposeAll = Effect.fn("InstanceStore.disposeAll")(function* () {
      if (disposal.all) return yield* Effect.promise(() => disposal.all!)

      disposal.all = iife(async () => {
        Log.Default.info("disposing all instances")
        const entries = [...cache.entries()]
        for (const [key, value] of entries) {
          if (cache.get(key) !== value) continue

          const ctx = await value.catch((error) => {
            Log.Default.warn("instance dispose failed", { key, error })
            return undefined
          })

          if (!ctx) {
            if (cache.get(key) === value) cache.delete(key)
            continue
          }

          if (cache.get(key) !== value) continue
          await Effect.runPromise(dispose(ctx))
        }
      }).finally(() => {
        disposal.all = undefined
      })

      return yield* Effect.promise(() => disposal.all!)
    })

    yield* Effect.addFinalizer(() => disposeAll().pipe(Effect.ignore))

    return Service.of({
      load,
      reload,
      dispose,
      disposeAll,
    })
  }),
)

export const defaultLayer = layer.pipe(Layer.provide(Project.defaultLayer))

export const runtime = makeRuntime(Service, defaultLayer)

export * as InstanceStore from "./instance-store"
