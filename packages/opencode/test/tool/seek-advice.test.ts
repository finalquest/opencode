import { afterEach, describe, expect } from "bun:test"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { Database } from "@opencode-ai/core/database/database"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { SessionProjector } from "@opencode-ai/core/session/projector"
import { Cause, Effect, Exit, Fiber, Layer } from "effect"
import { Agent } from "../../src/agent/agent"
import { BackgroundJob } from "@/background/job"
import { EventV2Bridge } from "@/event-v2-bridge"
import { Config } from "@/config/config"
import { CrossSpawnSpawner } from "@opencode-ai/core/cross-spawn-spawner"
import { Ripgrep } from "@opencode-ai/core/ripgrep"
import { Session } from "@/session/session"
import type { SessionPrompt } from "../../src/session/prompt"
import { MessageID, PartID, SessionID } from "../../src/session/schema"
import { SessionRunState } from "@/session/run-state"
import { SessionStatus } from "@/session/status"

import { SeekAdviceTool } from "../../src/tool/seek-advice"
import type { TaskPromptOps } from "../../src/tool/task"
import { Truncate } from "@/tool/truncate"
import { ToolRegistry } from "@/tool/registry"
import { RuntimeFlags } from "@/effect/runtime-flags"
import { disposeAllInstances } from "../fixture/fixture"
import { testEffect } from "../lib/effect"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { ModelV2 } from "@opencode-ai/core/model"

afterEach(async () => {
  await disposeAllInstances()
})

const ref = {
  providerID: ProviderV2.ID.make("test"),
  modelID: ModelV2.ID.make("test-model"),
}

const layer = (flags: Partial<RuntimeFlags.Info> = {}) =>
  LayerNode.compile(
    LayerNode.group([
      Agent.node,
      BackgroundJob.node,
      EventV2Bridge.node,
      Config.node,
      CrossSpawnSpawner.node,
      Session.node,
      SessionProjector.node,
      SessionRunState.node,
      SessionStatus.node,
      Truncate.node,
      ToolRegistry.node,
      Database.node,
      RuntimeFlags.node,
      Ripgrep.node,
    ]),
    [[RuntimeFlags.node, RuntimeFlags.layer(flags)]],
  )

const it = testEffect(layer())

function defer<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

const seed = Effect.fn("SeekAdviceTest.seed")(function* (title = "Advice") {
  const session = yield* Session.Service
  const chat = yield* session.create({ title })
  const user = yield* session.updateMessage({
    id: MessageID.ascending(),
    role: "user",
    sessionID: chat.id,
    agent: "build",
    model: ref,
    time: { created: Date.now() },
  })
  const assistant: SessionV1.Assistant = {
    id: MessageID.ascending(),
    role: "assistant",
    parentID: user.id,
    sessionID: chat.id,
    mode: "build",
    agent: "build",
    cost: 0,
    path: { cwd: "/tmp", root: "/tmp" },
    tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
    modelID: ref.modelID,
    providerID: ref.providerID,
    variant: "xhigh",
    time: { created: Date.now() },
  }
  yield* session.updateMessage(assistant)
  return { chat, assistant }
})

function reply(input: SessionPrompt.PromptInput, text: string): SessionV1.WithParts {
  const id = MessageID.ascending()
  return {
    info: {
      id,
      role: "assistant",
      parentID: input.messageID ?? MessageID.ascending(),
      sessionID: input.sessionID,
      mode: input.agent ?? "general",
      agent: input.agent ?? "general",
      cost: 0,
      path: { cwd: "/tmp", root: "/tmp" },
      tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
      modelID: input.model?.modelID ?? ref.modelID,
      providerID: input.model?.providerID ?? ref.providerID,
      time: { created: Date.now() },
      finish: "stop",
    },
    parts: [
      {
        id: PartID.ascending(),
        messageID: id,
        sessionID: input.sessionID,
        type: "text",
        text,
      },
    ],
  }
}

const adviceJson = JSON.stringify({
  summary: "Option A is more likely",
  recommendation: "Investigate the cache key path first",
  reasoning: "The error log points to a stale cache entry",
  risks: ["Fixing B may mask the real issue"],
  next_steps: ["Add a log statement at the cache lookup", "Reproduce with a cold cache"],
  confidence: "medium",
})

function stubOps(opts?: { onPrompt?: (input: SessionPrompt.PromptInput) => void; text?: string }): TaskPromptOps {
  return {
    cancel: () => Effect.void,
    resolvePromptParts: (template) => Effect.succeed([{ type: "text" as const, text: template }]),
    prompt: (input) =>
      Effect.sync(() => {
        opts?.onPrompt?.(input)
        return reply(input, opts?.text ?? adviceJson)
      }),
  }
}

const baseParams = {
  question: "Which of the two cache hypotheses is correct?",
  goal: "Choose the most likely root cause before editing code",
  context: "Error: stale entry at key foo. Observed on cold restart.",
}

function assertFailsWith(effect: Effect.Effect<unknown, unknown, never>, substring: string) {
  return Effect.gen(function* () {
    const exit = yield* effect.pipe(Effect.exit)
    expect(Exit.isFailure(exit)).toBe(true)
    if (Exit.isFailure(exit)) expect(Cause.pretty(exit.cause)).toContain(substring)
  })
}

describe("tool.seek_advice", () => {
  it.instance("returns rendered advice on valid JSON", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seed()
      const tool = yield* SeekAdviceTool
      const def = yield* tool.init()
      const promptOps = stubOps()

      const result = yield* def.execute(baseParams, {
        sessionID: chat.id,
        messageID: assistant.id,
        agent: "build",
        abort: new AbortController().signal,
        extra: { promptOps },
        messages: [],
        metadata: () => Effect.void,
        ask: () => Effect.void,
      })

      expect(result.output).toContain(`<advice confidence="medium">`)
      expect(result.output).toContain("<recommendation>Investigate the cache key path first</recommendation>")
      expect(result.output).toContain("<summary>Option A is more likely</summary>")
      expect(result.metadata).toHaveProperty("call", 1)
    }),
  )

  it.instance("includes conversation transcript in the advisor prompt", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seed()
      const tool = yield* SeekAdviceTool
      const def = yield* tool.init()
      let seen: SessionPrompt.PromptInput | undefined
      const promptOps = stubOps({ onPrompt: (input) => (seen = input) })

      const messages: SessionV1.WithParts[] = [
        {
          info: {
            id: MessageID.ascending(),
            role: "user",
            sessionID: chat.id,
            agent: "build",
            model: ref,
            time: { created: Date.now() },
          },
          parts: [
            {
              id: PartID.ascending(),
              messageID: MessageID.ascending(),
              sessionID: chat.id,
              type: "text",
              text: "please fix the cache bug",
            },
          ],
        },
      ]

      yield* def.execute(baseParams, {
        sessionID: chat.id,
        messageID: assistant.id,
        agent: "build",
        abort: new AbortController().signal,
        extra: { promptOps },
        messages,
        metadata: () => Effect.void,
        ask: () => Effect.void,
      })

      const first = seen?.parts[0]
      const promptText = first && first.type === "text" ? first.text : ""
      expect(promptText).toContain("Conversation transcript")
      expect(promptText).toContain("please fix the cache bug")
      expect(promptText).toContain("## user")
    }),
  )

  it.instance("fails when advisor is disabled", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seed()
      const tool = yield* SeekAdviceTool
      const def = yield* tool.init()
      const promptOps = stubOps()

      yield* assertFailsWith(
        def.execute(baseParams, {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "build",
          abort: new AbortController().signal,
          extra: { promptOps },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        }),
        "disabled",
      )
    }),
    { config: { advisor: { enabled: false } } },
  )

  it.instance("enforces the per-session call limit", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seed()
      const tool = yield* SeekAdviceTool
      const def = yield* tool.init()
      const promptOps = stubOps()

      yield* def.execute(baseParams, {
        sessionID: chat.id,
        messageID: assistant.id,
        agent: "build",
        abort: new AbortController().signal,
        extra: { promptOps },
        messages: [],
        metadata: () => Effect.void,
        ask: () => Effect.void,
      })
      yield* assertFailsWith(
        def.execute(baseParams, {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "build",
          abort: new AbortController().signal,
          extra: { promptOps },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        }),
        "call limit reached",
      )
    }),
    { config: { advisor: { max_calls_per_session: 1 } } },
  )

  it.instance("fails on invalid JSON from advisor", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seed()
      const tool = yield* SeekAdviceTool
      const def = yield* tool.init()
      const promptOps = stubOps({ text: "I think you should try option A." })

      yield* assertFailsWith(
        def.execute(baseParams, {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "build",
          abort: new AbortController().signal,
          extra: { promptOps },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        }),
        "valid JSON",
      )
    }),
  )

  it.instance("fails when advisor JSON fails schema validation", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seed()
      const tool = yield* SeekAdviceTool
      const def = yield* tool.init()
      const promptOps = stubOps({
        text: JSON.stringify({ summary: "s", recommendation: "r", reasoning: "re", risks: [], next_steps: [] }),
      })

      yield* assertFailsWith(
        def.execute(baseParams, {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "build",
          abort: new AbortController().signal,
          extra: { promptOps },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        }),
        "schema validation",
      )
    }),
  )

  it.instance("cancels child session when abort signal fires", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seed()
      const tool = yield* SeekAdviceTool
      const def = yield* tool.init()
      const ready = defer<SessionPrompt.PromptInput>()
      const cancelled = defer<SessionID>()
      const abort = new AbortController()
      const promptOps: TaskPromptOps = {
        cancel: (sessionID) =>
          Effect.sync(() => {
            cancelled.resolve(sessionID)
          }),
        resolvePromptParts: (template) => Effect.succeed([{ type: "text" as const, text: template }]),
        prompt: (input) =>
          Effect.promise(() => {
            ready.resolve(input)
            return cancelled.promise
          }).pipe(Effect.as(reply(input, "cancelled"))),
      }

      const fiber = yield* def
        .execute(baseParams, {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "build",
          abort: abort.signal,
          extra: { promptOps },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        })
        .pipe(Effect.forkChild)

      const input = yield* Effect.promise(() => ready.promise)
      abort.abort()
      expect(yield* Effect.promise(() => cancelled.promise)).toBe(input.sessionID)

      const exit = yield* Fiber.await(fiber)
      expect(Exit.isFailure(exit)).toBe(true)
    }),
  )
})
