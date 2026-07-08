import * as Tool from "./tool"
import DESCRIPTION from "./seek-advice.txt"
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

const id = "seek_advice"

const ExpectedOutput = Schema.Literals(["analysis", "decision", "plan", "risk_review", "debugging"])
const Confidence = Schema.Literals(["low", "medium", "high"])

const ParameterFields = {
  question: Schema.String.annotate({
    description: "The specific question or doubt you want resolved",
  }),
  goal: Schema.String.annotate({
    description: "What you intend to do with the advice, eg choose between alternatives, validate a plan",
  }),
  context: Schema.String.annotate({
    description: "Relevant context: what you tried, what you observed, code snippets, error messages. The advisor cannot read files, so inline anything relevant here.",
  }),
  constraints: Schema.optional(Schema.mutable(Schema.Array(Schema.String))).annotate({
    description: "Constraints the recommendation must respect, eg do not change architecture",
  }),
  files: Schema.optional(Schema.mutable(Schema.Array(Schema.String))).annotate({
    description: "Relevant file paths for reference. The advisor cannot read them; inline their content in context if needed.",
  }),
  diff: Schema.optional(Schema.String).annotate({
    description: "A diff or proposed change to review",
  }),
  logs: Schema.optional(Schema.String).annotate({
    description: "Relevant logs or command output",
  }),
  expected_output: Schema.optional(ExpectedOutput).annotate({
    description: "The kind of response you want: analysis, decision, plan, risk_review, or debugging",
  }),
}

const Parameters = Schema.Struct(ParameterFields)

const AdviceOutput = Schema.Struct({
  summary: Schema.String,
  recommendation: Schema.String,
  reasoning: Schema.String,
  risks: Schema.mutable(Schema.Array(Schema.String)),
  next_steps: Schema.mutable(Schema.Array(Schema.String)),
  confidence: Confidence,
})

const decodeAdvice = Schema.decodeUnknownEffect(AdviceOutput)

const DEFAULT_MAX_CALLS = 10
const DEFAULT_TIMEOUT_MS = 60_000
const TOOL_OUTPUT_LIMIT = 2000

function truncate(text: string): string {
  return text.length > TOOL_OUTPUT_LIMIT ? text.slice(0, TOOL_OUTPUT_LIMIT) + "\n... (truncated)" : text
}

function renderConversation(messages: SessionV1.WithParts[]): string {
  return messages
    .map((message) => {
      const role = message.info.role
      const body = message.parts
        .flatMap((part) => {
          if (part.type === "text" && !part.synthetic) return [part.text]
          if (part.type === "tool") {
            const state = part.state
            if (state.status === "completed") return [`[tool ${part.tool}]\n${truncate(state.output)}`]
            if (state.status === "error") return [`[tool ${part.tool} ERROR]\n${truncate(state.error)}`]
            return [`[tool ${part.tool} ${state.status}]`]
          }
          return []
        })
        .join("\n\n")
      if (!body) return null
      return `## ${role}\n\n${body}`
    })
    .filter(Boolean)
    .join("\n\n")
}

function buildPrompt(input: Schema.Schema.Type<typeof Parameters>, transcript: string): string {
  const lines: string[] = []
  lines.push(`Conversation transcript\n\n${transcript}`)
  lines.push(`Question\n\n${input.question}`)
  lines.push(`Goal\n\n${input.goal}`)
  lines.push(`Context\n\n${input.context}`)
  if (input.constraints?.length) lines.push(`Constraints\n\n${input.constraints.map((c) => `- ${c}`).join("\n")}`)
  if (input.files?.length) lines.push(`Files\n\n${input.files.map((f) => `- ${f}`).join("\n")}`)
  if (input.diff) lines.push(`Diff\n\n${input.diff}`)
  if (input.logs) lines.push(`Logs\n\n${input.logs}`)
  if (input.expected_output) lines.push(`Expected output\n\n${input.expected_output}`)
  return lines.join("\n\n")
}

function extractJson(text: string): unknown {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i)
  const candidate = fenced ? fenced[1] : text
  const start = candidate.lastIndexOf("{")
  const end = candidate.indexOf("}", start)
  if (start === -1 || end === -1) return JSON.parse(candidate.trim())
  return JSON.parse(candidate.slice(start, end + 1))
}

function renderAdvice(output: Schema.Schema.Type<typeof AdviceOutput>): string {
  return [
    `<advice confidence="${output.confidence}">`,
    `<summary>${output.summary}</summary>`,
    `<recommendation>${output.recommendation}</recommendation>`,
    `<reasoning>${output.reasoning}</reasoning>`,
    output.risks.length ? `<risks>\n${output.risks.map((r) => `- ${r}`).join("\n")}\n</risks>` : "",
    output.next_steps.length ? `<next_steps>\n${output.next_steps.map((s) => `- ${s}`).join("\n")}\n</next_steps>` : "",
    "</advice>",
  ]
    .filter(Boolean)
    .join("\n")
}

export const SeekAdviceTool = Tool.define(
  id,
  Effect.gen(function* () {
    const agent = yield* Agent.Service
    const config = yield* Config.Service
    const sessions = yield* Session.Service
    const database = yield* Database.Service

    const callCounts = new Map<string, number>()

    const run = Effect.fn("SeekAdviceTool.execute")(function* (
      params: Schema.Schema.Type<typeof Parameters>,
      ctx: Tool.Context,
    ) {
      const cfg = yield* config.get()
      const advisorCfg = cfg.advisor
      if (advisorCfg?.enabled === false) {
        return yield* Effect.fail(new Error("seek_advice is disabled via config (advisor.enabled = false)"))
      }

      const maxCalls = advisorCfg?.max_calls_per_session ?? DEFAULT_MAX_CALLS
      const used = callCounts.get(ctx.sessionID) ?? 0
      if (used >= maxCalls) {
        return yield* Effect.fail(
          new Error(`seek_advice call limit reached for this session (${used}/${maxCalls}). Proceed without advice.`),
        )
      }
      callCounts.set(ctx.sessionID, used + 1)

      const next = yield* agent.get("advisor")
      if (!next) return yield* Effect.fail(new Error("advisor agent is not available"))

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
        title: `seek_advice (@${next.name})`,
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

      const model = advisorCfg?.model
        ? parseModel(advisorCfg.model)
        : { modelID: msg.info.modelID, providerID: msg.info.providerID }

      const metadata = {
        parentSessionId: ctx.sessionID,
        sessionId: childSession.id,
        model,
        call: used + 1,
      }
      yield* ctx.metadata({ title: params.question.slice(0, 80), metadata })

      const ops = ctx.extra?.promptOps as TaskPromptOps
      if (!ops) return yield* Effect.fail(new Error("SeekAdviceTool requires promptOps in ctx.extra"))

      const transcript = renderConversation(ctx.messages)

      const runAdvice = Effect.fn("SeekAdviceTool.runAdvice")(function* () {
        const parts = yield* ops.resolvePromptParts(buildPrompt(params, transcript))
        const result = yield* ops.prompt({
          messageID: MessageID.ascending(),
          sessionID: childSession.id,
          model: { modelID: model.modelID, providerID: model.providerID },
          variant: advisorCfg?.model ? undefined : variant,
          agent: next.name,
          parts,
        })
        return result.parts.findLast((item) => item.type === "text")?.text ?? ""
      })

      const timeoutMs = advisorCfg?.timeout ?? DEFAULT_TIMEOUT_MS
      const runCancel = yield* EffectBridge.make()
      const cancel = ops.cancel(childSession.id)
      function onAbort() {
        runCancel.fork(cancel)
      }

      const text = yield* Effect.acquireUseRelease(
        Effect.sync(() => {
          ctx.abort.addEventListener("abort", onAbort)
        }),
        () => runAdvice().pipe(Effect.timeout(Duration.millis(timeoutMs))),
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

      const parsed = yield* Effect.try({
        try: () => extractJson(text),
        catch: (error) => new Error(`Advisor did not return valid JSON: ${String(error)}`),
      })
      const advice = yield* decodeAdvice(parsed).pipe(
        Effect.mapError((error) => new Error(`Advisor response failed schema validation: ${String(error)}`)),
      )

      return {
        title: params.question.slice(0, 80),
        metadata,
        output: renderAdvice(advice),
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
