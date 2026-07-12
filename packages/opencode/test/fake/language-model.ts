import type { LanguageModelV3, LanguageModelV3CallOptions, LanguageModelV3GenerateResult, LanguageModelV3StreamResult } from "@ai-sdk/provider"

type MockResponse = { json: string } | { error: Error } | { hang: true }

export function mockLanguageModel(response: MockResponse): LanguageModelV3 {
  const result: LanguageModelV3GenerateResult = {
    content: [{ type: "text" as const, text: "json" in response ? response.json : "" }],
    finishReason: { unified: "stop" as const, raw: "stop" },
    usage: {
      inputTokens: { total: undefined, noCache: undefined, cacheRead: undefined, cacheWrite: undefined },
      outputTokens: { total: undefined, text: undefined, reasoning: undefined },
    },
    warnings: [],
  }

  return {
    specificationVersion: "v3" as const,
    provider: "test",
    modelId: "test-model",
    supportedUrls: {},
    doGenerate(_options: LanguageModelV3CallOptions): PromiseLike<LanguageModelV3GenerateResult> {
      if ("error" in response) return Promise.reject(response.error)
      if ("hang" in response) return new Promise(() => {})
      return Promise.resolve(result)
    },
    doStream(_options: LanguageModelV3CallOptions): PromiseLike<LanguageModelV3StreamResult> {
      if ("error" in response) return Promise.reject(response.error)
      throw new Error("mockLanguageModel.doStream not implemented")
    },
  }
}
