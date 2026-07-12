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
import { parseModel, Provider } from "@/provider/provider"
import { ProviderTransform } from "@/provider/transform"
import { Auth } from "@/auth"
import { Effect, Schema, Exit } from "effect"
import * as Option from "effect/Option"
import * as Duration from "effect/Duration"
import { EffectBridge } from "@/effect/bridge"
import { Database } from "@opencode-ai/core/database/database"
import { generateObject, streamObject, type ModelMessage } from "ai"
import PROMPT_ADVISOR from "../agent/prompt/advisor.txt"

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

type AdviceOutputType = Schema.Schema.Type<typeof AdviceOutput>

const decodeAdvice = Schema.decodeUnknownEffect(AdviceOutput)

const adviceSchema = Object.assign(
  Schema.toStandardSchemaV1(AdviceOutput),
  Schema.toStandardJSONSchemaV1(AdviceOutput),
)

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
  const candidate = (fenced ? fenced[1] : text).trim()
  try {
    return JSON.parse(candidate)
  } catch {}
  const start = candidate.indexOf("{")
  if (start === -1) return JSON.parse(candidate)
  let depth = 0
  let end = start
  for (let i = start; i < candidate.length; i++) {
    if (candidate[i] === "{") depth++
    else if (candidate[i] === "}") depth--
    if (depth === 0) {
      end = i
      break
    }
  }
  return JSON.parse(candidate.slice(start, end + 1))
}

function renderAdvice(output: AdviceOutputType): string {
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

      const msg = yield* MessageV2.get({ sessionID: ctx.sessionID, messageID: ctx.messageID }).pipe(
        Effect.provideService(Database.Service, database),
        Effect.orDie,
      )
      if (msg.info.role !== "assistant") return yield* Effect.fail(new Error("Not an assistant message"))
      const variant = msg.info.variant

      const model = advisorCfg?.model
        ? parseModel(advisorCfg.model)
        : { modelID: msg.info.modelID, providerID: msg.info.providerID }

      const baseMetadata = {
        parentSessionId: ctx.sessionID,
        model,
        call: used + 1,
      }

      const timeoutMs = advisorCfg?.timeout ?? DEFAULT_TIMEOUT_MS
      const transcript = renderConversation(ctx.messages)
      const promptText = buildPrompt(params, transcript)

      const tryStructured = Effect.gen(function* () {
        const providerOpt = yield* Effect.serviceOption(Provider.Service)
        if (Option.isNone(providerOpt)) return yield* Effect.fail(new Error("Provider.Service not available"))
        const provider = providerOpt.value

        const resolved = yield* provider.getModel(model.providerID, model.modelID).pipe(
          Effect.mapError((err) => new Error(`Model not found: ${String(err)}`)),
        )
        const language = yield* provider.getLanguage(resolved).pipe(
          Effect.mapError((err) => new Error(`Language model not available: ${String(err)}`)),
        )

        const authOpt = yield* Effect.serviceOption(Auth.Service)
        const authInfo = Option.isSome(authOpt)
          ? yield* authOpt.value.get(model.providerID).pipe(Effect.orElseSucceed(() => undefined))
          : undefined
        const isOpenaiOauth = model.providerID === "openai" && authInfo?.type === "oauth"

        const messages: ModelMessage[] = [
          ...(isOpenaiOauth
            ? []
            : [{ role: "system" as const, content: PROMPT_ADVISOR } satisfies ModelMessage]),
          { role: "user" as const, content: promptText } satisfies ModelMessage,
        ]

        const genParams = {
          temperature: advisorCfg?.temperature ?? 0.2,
          messages,
          model: language,
          schema: adviceSchema,
        } satisfies Parameters<typeof generateObject>[0]

        if (isOpenaiOauth) {
          return yield* Effect.promise(async () => {
            const result = streamObject({
              ...genParams,
              providerOptions: ProviderTransform.providerOptions(resolved, {
                instructions: PROMPT_ADVISOR,
                store: false,
              }),
              onError: () => {},
            })
            for await (const part of result.fullStream) {
              if (part.type === "error") throw part.error
            }
            return result.object
          })
        }

        return yield* Effect.promise(() => generateObject(genParams).then((r) => r.object))
      })

      const structuredExit = yield* tryStructured.pipe(
        Effect.timeout(Duration.millis(timeoutMs)),
        Effect.exit,
      )

      if (Exit.isSuccess(structuredExit)) {
        const advice = structuredExit.value
        const metadata = { ...baseMetadata, path: "structured" }
        yield* ctx.metadata({ title: params.question.slice(0, 80), metadata })
        return {
          title: params.question.slice(0, 80),
          metadata,
          output: renderAdvice(advice),
        }
      }

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

      const metadata = { ...baseMetadata, sessionId: childSession.id, path: "fallback" }
      yield* ctx.metadata({ title: params.question.slice(0, 80), metadata })

      const ops = ctx.extra?.promptOps as TaskPromptOps
      if (!ops) return yield* Effect.fail(new Error("SeekAdviceTool requires promptOps in ctx.extra"))

      const runAdvice = Effect.fn("SeekAdviceTool.runAdvice")(function* () {
        const parts = yield* ops.resolvePromptParts(promptText)
        const result = yield* ops.prompt({
          messageID: MessageID.ascending(),
          sessionID: childSession.id,
          model: { modelID: model.modelID, providerID: model.providerID },
          variant: advisorCfg?.variant ?? (advisorCfg?.model ? undefined : variant),
          agent: next.name,
          parts,
        })
        return result.parts.findLast((item) => item.type === "text")?.text ?? ""
      })

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

      const parseExit = yield* Effect.gen(function* () {
        const parsed = yield* Effect.try({
          try: () => extractJson(text),
          catch: (error) => new Error(`Advisor did not return valid JSON: ${String(error)}`),
        })
        return yield* decodeAdvice(parsed).pipe(
          Effect.mapError((error) => new Error(`Advisor response failed schema validation: ${String(error)}`)),
        )
      }).pipe(Effect.exit)

      if (Exit.isSuccess(parseExit)) {
        return {
          title: params.question.slice(0, 80),
          metadata,
          output: renderAdvice(parseExit.value),
        }
      }

      return {
        title: params.question.slice(0, 80),
        metadata,
        output: [
          `<advice confidence="unknown">`,
          `<summary>Advisor response (unparsed)</summary>`,
          `<raw>${text}</raw>`,
          `</advice>`,
        ].join("\n"),
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
