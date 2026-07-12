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
import { FSUtil } from "@opencode-ai/core/fs-util"
import { Ripgrep } from "@opencode-ai/core/ripgrep"
import { Session } from "@/session/session"
import type { SessionPrompt } from "../../src/session/prompt"
import { MessageID, PartID, SessionID } from "../../src/session/schema"
import { SessionRunState } from "@/session/run-state"
import { SessionStatus } from "@/session/status"

import { AnalyzeImageTool } from "../../src/tool/analyze-image"
import type { TaskPromptOps } from "../../src/tool/task"
import { Truncate } from "@/tool/truncate"
import { ToolRegistry } from "@/tool/registry"
import { RuntimeFlags } from "@/effect/runtime-flags"
import { disposeAllInstances } from "../fixture/fixture"
import { testEffect } from "../lib/effect"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { ModelV2 } from "@opencode-ai/core/model"
import * as os from "os"
import * as path from "path"

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
      FSUtil.node,
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

const seed = Effect.fn("AnalyzeImageTest.seed")(function* (title = "Image Analysis") {
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

const ANALYSIS_TEXT = "This is a 1x1 transparent PNG image. It contains a single pixel with no visible content."

function stubOps(opts?: { onPrompt?: (input: SessionPrompt.PromptInput) => void; text?: string }): TaskPromptOps {
  return {
    cancel: () => Effect.void,
    resolvePromptParts: (template) => Effect.succeed([{ type: "text" as const, text: template }]),
    prompt: (input) =>
      Effect.sync(() => {
        opts?.onPrompt?.(input)
        return reply(input, opts?.text ?? ANALYSIS_TEXT)
      }),
  }
}

function assertFailsWith(effect: Effect.Effect<unknown, unknown, never>, substring: string) {
  return Effect.gen(function* () {
    const exit = yield* effect.pipe(Effect.exit)
    expect(Exit.isFailure(exit)).toBe(true)
    if (Exit.isFailure(exit)) expect(Cause.pretty(exit.cause)).toContain(substring)
  })
}

const PNG_BYTES = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPh8DwAKDQ0=",
  "base64",
)

function tmpFile(name: string, content: Buffer | string) {
  const filePath = path.join(os.tmpdir(), `opencode-test-${Math.random().toString(36).slice(2)}-${name}`)
  return Effect.promise(() => Bun.write(filePath, content)).pipe(Effect.as(filePath))
}

describe("tool.analyze_image", () => {
  it.instance("returns text description on valid image", () =>
    Effect.gen(function* () {
      const imagePath = yield* tmpFile("test.png", PNG_BYTES)
      const { chat, assistant } = yield* seed()
      const tool = yield* AnalyzeImageTool
      const def = yield* tool.init()
      const promptOps = stubOps()

      const result = yield* def.execute(
        { image_path: imagePath },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "build",
          abort: new AbortController().signal,
          extra: { promptOps },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      expect(result.output).toContain("1x1 transparent PNG")
      expect(result.metadata).toHaveProperty("call", 1)
      expect(result.metadata.image as string).toEndWith("test.png")
    }),
  )

  it.instance("passes image FilePart to child session prompt", () =>
    Effect.gen(function* () {
      const imagePath = yield* tmpFile("test.png", PNG_BYTES)
      const { chat, assistant } = yield* seed()
      const tool = yield* AnalyzeImageTool
      const def = yield* tool.init()
      let seen: SessionPrompt.PromptInput | undefined
      const promptOps = stubOps({ onPrompt: (input) => (seen = input) })

      yield* def.execute(
        { image_path: imagePath },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "build",
          abort: new AbortController().signal,
          extra: { promptOps },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      const filePart = seen?.parts.find((p) => p.type === "file")
      expect(filePart).toBeDefined()
      expect(filePart?.type === "file" && filePart.mime).toBe("image/png")
      const textPart = seen?.parts.find((p) => p.type === "text")
      expect(textPart).toBeDefined()
      expect(textPart?.type === "text" && textPart.text).toContain("describe in detail")
    }),
  )

  it.instance("fails when image_analyzer is disabled", () =>
    Effect.gen(function* () {
      const imagePath = yield* tmpFile("test.png", PNG_BYTES)
      const { chat, assistant } = yield* seed()
      const tool = yield* AnalyzeImageTool
      const def = yield* tool.init()
      const promptOps = stubOps()

      yield* assertFailsWith(
        def.execute(
          { image_path: imagePath },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "build",
            abort: new AbortController().signal,
            extra: { promptOps },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        ),
        "disabled",
      )
    }),
    { config: { image_analyzer: { enabled: false } } },
  )

  it.instance("enforces the per-session call limit", () =>
    Effect.gen(function* () {
      const imagePath = yield* tmpFile("test.png", PNG_BYTES)
      const { chat, assistant } = yield* seed()
      const tool = yield* AnalyzeImageTool
      const def = yield* tool.init()
      const promptOps = stubOps()

      yield* def.execute(
        { image_path: imagePath },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "build",
          abort: new AbortController().signal,
          extra: { promptOps },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )
      yield* assertFailsWith(
        def.execute(
          { image_path: imagePath },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "build",
            abort: new AbortController().signal,
            extra: { promptOps },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        ),
        "call limit reached",
      )
    }),
    { config: { image_analyzer: { max_calls_per_session: 1 } } },
  )

  it.instance("fails on unsupported file type", () =>
    Effect.gen(function* () {
      const textPath = yield* tmpFile("not-an-image.txt", "hello world")
      const { chat, assistant } = yield* seed()
      const tool = yield* AnalyzeImageTool
      const def = yield* tool.init()
      const promptOps = stubOps()

      yield* assertFailsWith(
        def.execute(
          { image_path: textPath },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "build",
            abort: new AbortController().signal,
            extra: { promptOps },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        ),
        "Unsupported image type",
      )
    }),
  )

  it.instance("fails on missing file", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seed()
      const tool = yield* AnalyzeImageTool
      const def = yield* tool.init()
      const promptOps = stubOps()

      yield* assertFailsWith(
        def.execute(
          { image_path: "/nonexistent/path/to/image.png" },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "build",
            abort: new AbortController().signal,
            extra: { promptOps },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        ),
        "not found",
      )
    }),
  )

  it.instance("cancels child session when abort signal fires", () =>
    Effect.gen(function* () {
      const imagePath = yield* tmpFile("test.png", PNG_BYTES)
      const { chat, assistant } = yield* seed()
      const tool = yield* AnalyzeImageTool
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
        .execute(
          { image_path: imagePath },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "build",
            abort: abort.signal,
            extra: { promptOps },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )
        .pipe(Effect.forkChild)

      const input = yield* Effect.promise(() => ready.promise)
      abort.abort()
      expect(yield* Effect.promise(() => cancelled.promise)).toBe(input.sessionID)

      const exit = yield* Fiber.await(fiber)
      expect(Exit.isSuccess(exit)).toBe(true)
    }),
  )
})
