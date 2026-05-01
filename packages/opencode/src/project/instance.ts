import { GlobalBus } from "@/bus/global"
import { disposeInstance } from "@/effect/instance-registry"
import { makeRuntime } from "@/effect/run-service"
import { AppFileSystem } from "@opencode-ai/core/filesystem"
import { iife } from "@/util/iife"
import * as Log from "@opencode-ai/core/util/log"
import { LocalContext } from "@/util/local-context"
import * as Project from "./project"
import { WorkspaceContext } from "@/control-plane/workspace-context"
import { Context, Effect, Layer } from "effect"

export interface InstanceContext {
  directory: string
  worktree: string
  project: Project.Info
}

const context = LocalContext.create<InstanceContext>("instance")

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

export class InstanceStore extends Context.Service<InstanceStore, Interface>()("@opencode/InstanceStore") {}

export const instanceStoreLayer: Layer.Layer<InstanceStore, never, Project.Service> = Layer.effect(
  InstanceStore,
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

    return InstanceStore.of({
      load,
      reload,
      dispose,
      disposeAll,
    })
  }),
)

export const instanceStoreDefaultLayer = instanceStoreLayer.pipe(Layer.provide(Project.defaultLayer))

const instanceStoreRuntime = makeRuntime(InstanceStore, instanceStoreDefaultLayer)

export const Instance = {
  load(input: LoadInput): Promise<InstanceContext> {
    return instanceStoreRuntime.runPromise((store) => store.load(input))
  },
  async provide<R>(input: { directory: string; init?: () => Promise<any>; fn: () => R }): Promise<R> {
    return context.provide(await Instance.load(input), async () => input.fn())
  },
  get current() {
    return context.use()
  },
  get directory() {
    return context.use().directory
  },
  get worktree() {
    return context.use().worktree
  },
  get project() {
    return context.use().project
  },

  /**
   * Check if a path is within the project boundary.
   * Returns true if path is inside Instance.directory OR Instance.worktree.
   * Paths within the worktree but outside the working directory should not trigger external_directory permission.
   */
  containsPath(filepath: string, ctx?: InstanceContext) {
    const instance = ctx ?? Instance
    if (AppFileSystem.contains(instance.directory, filepath)) return true
    // Non-git projects set worktree to "/" which would match ANY absolute path.
    // Skip worktree check in this case to preserve external_directory permissions.
    if (instance.worktree === "/") return false
    return AppFileSystem.contains(instance.worktree, filepath)
  },
  /**
   * Captures the current instance ALS context and returns a wrapper that
   * restores it when called. Use this for callbacks that fire outside the
   * instance async context (native addons, event emitters, timers, etc.).
   */
  bind<F extends (...args: any[]) => any>(fn: F): F {
    const ctx = context.use()
    return ((...args: any[]) => context.provide(ctx, () => fn(...args))) as F
  },
  /**
   * Run a synchronous function within the given instance context ALS.
   * Use this to bridge from Effect (where InstanceRef carries context)
   * back to sync code that reads Instance.directory from ALS.
   */
  restore<R>(ctx: InstanceContext, fn: () => R): R {
    return context.provide(ctx, fn)
  },
  async reload(input: { directory: string; init?: () => Promise<any>; project?: Project.Info; worktree?: string }) {
    return instanceStoreRuntime.runPromise((store) => store.reload(input))
  },
  async dispose() {
    return instanceStoreRuntime.runPromise((store) => store.dispose(Instance.current))
  },
  async disposeAll() {
    return instanceStoreRuntime.runPromise((store) => store.disposeAll())
  },
}
