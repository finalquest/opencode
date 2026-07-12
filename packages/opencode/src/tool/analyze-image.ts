import * as Tool from "./tool"
import DESCRIPTION from "./analyze-image.txt"
import { ToolJsonSchema } from "./json-schema"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { Session } from "@/session/session"
import { MessageID } from "../session/schema"
import { MessageV2 } from "../session/message-v2"
import { Agent } from "../agent/agent"
import { deriveSubagentSessionPermission } from "../agent/subagent-permissions"
import type { TaskPromptOps } from "./task"
import { Config } from "@/config/config"
import { parseModel } from "@/provider/provider"
import { Effect, Schema } from "effect"
import * as Duration from "effect/Duration"
import { EffectBridge } from "@/effect/bridge"
import { Database } from "@opencode-ai/core/database/database"
import { FSUtil } from "@opencode-ai/core/fs-util"
import { InstanceState } from "@/effect/instance-state"
import { sniffAttachmentMime } from "@/util/media"
import * as path from "path"

const id = "analyze_image"

const Parameters = Schema.Struct({
  image_path: Schema.String.annotate({
    description: "Absolute or project-relative path to the image file to analyze",
  }),
  question: Schema.optional(Schema.String).annotate({
    description: "Optional question about the image. If omitted, the analyzer describes everything it sees.",
  }),
})

const DEFAULT_MAX_CALLS = 10
const DEFAULT_TIMEOUT_MS = 60_000
const SUPPORTED_IMAGE_MIMES = new Set(["image/jpeg", "image/png", "image/gif", "image/webp"])

export const AnalyzeImageTool = Tool.define(
  id,
  Effect.gen(function* () {
    const agent = yield* Agent.Service
    const config = yield* Config.Service
    const sessions = yield* Session.Service
    const database = yield* Database.Service
    const fs = yield* FSUtil.Service

    const callCounts = new Map<string, number>()

    const run = Effect.fn("AnalyzeImageTool.execute")(function* (
      params: Schema.Schema.Type<typeof Parameters>,
      ctx: Tool.Context,
    ) {
      const cfg = yield* config.get()
      const imageCfg = cfg.image_analyzer
      if (imageCfg?.enabled === false) {
        return yield* Effect.fail(new Error("analyze_image is disabled via config (image_analyzer.enabled = false)"))
      }

      const maxCalls = imageCfg?.max_calls_per_session ?? DEFAULT_MAX_CALLS
      const used = callCounts.get(ctx.sessionID) ?? 0
      if (used >= maxCalls) {
        return yield* Effect.fail(
          new Error(`analyze_image call limit reached for this session (${used}/${maxCalls}).`),
        )
      }
      callCounts.set(ctx.sessionID, used + 1)

      const instance = yield* InstanceState.context
      let filepath = params.image_path
      if (!path.isAbsolute(filepath)) {
        filepath = path.resolve(instance.directory, filepath)
      }

      const stat = yield* fs.stat(filepath).pipe(
        Effect.catchIf(
          (err) => "reason" in err && err.reason._tag === "NotFound",
          () => Effect.succeed(undefined),
        ),
      )
      if (!stat) return yield* Effect.fail(new Error(`Image file not found: ${filepath}`))

      const bytes = yield* fs.readFile(filepath)
      const mime = sniffAttachmentMime(bytes.subarray(0, 4096), FSUtil.mimeType(filepath))
      if (!SUPPORTED_IMAGE_MIMES.has(mime)) {
        return yield* Effect.fail(
          new Error(`Unsupported image type: ${mime}. Supported: jpeg, png, gif, webp.`),
        )
      }

      const base64 = Buffer.from(bytes).toString("base64")
      const dataUrl = `data:${mime};base64,${base64}`
      const filename = path.basename(filepath)

      const next = yield* agent.get("image_analyzer")
      if (!next) return yield* Effect.fail(new Error("image_analyzer agent is not available"))

      const parent = yield* sessions.get(ctx.sessionID)
      const childPermission = deriveSubagentSessionPermission({
        parentSessionPermission: parent.permission ?? [],
        subagent: next,
      })
      const childDenies = [
        { permission: id, pattern: "*" as const, action: "deny" as const },
        { permission: "task" as const, pattern: "*" as const, action: "deny" as const },
      ]
      const childSession = yield* sessions.create({
        parentID: ctx.sessionID,
        title: `analyze_image (@${next.name})`,
        agent: next.name,
        permission: [
          ...childPermission,
          ...childDenies.filter(
            (deny) =>
              !childPermission.some(
                (rule) =>
                  rule.permission === deny.permission && rule.pattern === deny.pattern && rule.action === deny.action,
              ),
          ),
        ],
      })

      const msg = yield* MessageV2.get({ sessionID: ctx.sessionID, messageID: ctx.messageID }).pipe(
        Effect.provideService(Database.Service, database),
        Effect.orDie,
      )
      if (msg.info.role !== "assistant") return yield* Effect.fail(new Error("Not an assistant message"))
      const variant = msg.info.variant

      const model = imageCfg?.model
        ? parseModel(imageCfg.model)
        : { modelID: msg.info.modelID, providerID: msg.info.providerID }

      const metadata = {
        parentSessionId: ctx.sessionID,
        sessionId: childSession.id,
        model,
        call: used + 1,
        image: filename,
      }
      yield* ctx.metadata({ title: `Image: ${filename}`.slice(0, 80), metadata })

      const ops = ctx.extra?.promptOps as TaskPromptOps
      if (!ops) return yield* Effect.fail(new Error("AnalyzeImageTool requires promptOps in ctx.extra"))

      const instructionText = params.question
        ? `Analyze the following image and answer this question: ${params.question}`
        : "Analyze the following image and describe in detail everything you see."

      const runAnalysis = Effect.fn("AnalyzeImageTool.runAnalysis")(function* () {
        const result = yield* ops.prompt({
          messageID: MessageID.ascending(),
          sessionID: childSession.id,
          model: { modelID: model.modelID, providerID: model.providerID },
          variant: imageCfg?.variant ?? (imageCfg?.model ? undefined : variant),
          agent: next.name,
          parts: [
            { type: "file", mime, url: dataUrl, filename },
            { type: "text", text: instructionText },
          ],
        })
        return result.parts.findLast((item) => item.type === "text")?.text ?? ""
      })

      const timeoutMs = imageCfg?.timeout ?? DEFAULT_TIMEOUT_MS
      const runCancel = yield* EffectBridge.make()
      const cancel = ops.cancel(childSession.id)
      function onAbort() {
        runCancel.fork(cancel)
      }

      const text = yield* Effect.acquireUseRelease(
        Effect.sync(() => {
          ctx.abort.addEventListener("abort", onAbort)
        }),
        () => runAnalysis().pipe(Effect.timeout(Duration.millis(timeoutMs))),
        (_, exit) =>
          Effect.gen(function* () {
            if (exit._tag === "Failure") yield* cancel.pipe(Effect.ignore)
          }).pipe(
            Effect.ensuring(
              Effect.sync(() => {
                ctx.abort.removeEventListener("abort", onAbort)
              }),
            ),
          ),
      )

      return {
        title: `Image: ${filename}`.slice(0, 80),
        metadata,
        output: text || "The image analyzer returned an empty response.",
      }
    })

    return {
      description: DESCRIPTION,
      parameters: Parameters,
      jsonSchema: ToolJsonSchema.fromSchema(Parameters),
      execute: (params: Schema.Schema.Type<typeof Parameters>, ctx: Tool.Context) =>
        run(params, ctx).pipe(Effect.orDie),
    }
  }),
)
