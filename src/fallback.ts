import { ProviderError, type ModelProvider, type ModelRequest, type ModelResponse } from "./types.js";

export interface FallbackOptions {
  /** Default: only ProviderError with retryable = true moves to the next provider. */
  shouldFallback?: (error: unknown) => boolean;
  onFallback?: (from: ModelProvider, to: ModelProvider, error: unknown) => void;
}

/**
 * Try providers in order. A retryable failure — rate limit, overload, network —
 * moves to the next one; anything else is thrown as-is, because a 400 from
 * the first provider will be a 400 from the second and the caller needs to
 * see it. The response's `model` says who actually answered.
 */
export class FallbackProvider implements ModelProvider {
  readonly name = "fallback";

  constructor(
    private readonly providers: readonly ModelProvider[],
    private readonly options: FallbackOptions = {},
  ) {
    if (providers.length === 0) throw new Error("FallbackProvider needs at least one provider");
  }

  get model(): string {
    return this.providers[0]!.model;
  }

  async complete(request: ModelRequest): Promise<ModelResponse> {
    const shouldFallback = this.options.shouldFallback ?? ((e) => e instanceof ProviderError && e.retryable);
    let lastError: unknown;
    for (let i = 0; i < this.providers.length; i++) {
      const provider = this.providers[i]!;
      try {
        return await provider.complete(request);
      } catch (err) {
        lastError = err;
        const next = this.providers[i + 1];
        if (!next || !shouldFallback(err)) throw err;
        this.options.onFallback?.(provider, next, err);
      }
    }
    throw lastError;
  }
}
