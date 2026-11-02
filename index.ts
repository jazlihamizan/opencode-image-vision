/**
 * opencode-image-vision — OpenCode v2 plugin
 *
 * Gives text-only models the ability to understand pasted images.
 * When a user attaches an image, the plugin replaces it with a short text
 * instruction and registers an `analyze_image` tool. The tool sends the image
 * to a separate vision model and returns a detailed text description.
 *
 * v2.1 performance architecture:
 *  - A dedicated lightweight agent (no tools) runs the vision request, so the
 *    child session never loads the full build-agent system prompt.
 *  - A `prompt` hook prefetches image analysis in parallel the moment the
 *    user submits a message, overlapping vision latency with the main
 *    model's reasoning.
 *
 * Ported from the v1 plugin @showlotus/opencode-image-vision (MIT) to the
 * OpenCode v2 plugin API.
 *
 * License: MIT
 */

import { Plugin } from "@opencode/plugin"
import { createHash } from "node:crypto"
import { mkdirSync, writeFileSync, readFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join, extname, resolve, sep } from "node:path"

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const TEMP_DIR = join(tmpdir(), "iv-images")
const DEBUG_LOG = process.env.IMAGE_VISION_DEBUG_PATH || join(tmpdir(), "iv-debug.log")

const AGENT_ID = "image-vision-analyst"

const DEFAULT_PROMPT = [
  "Describe this image concisely but completely. Use short Markdown.",
  "Cover: text content (transcribe exactly), UI layout, key elements, colors.",
  "Include code or technical details when present. No preamble.",
].join(" ")

const MIME_TO_EXT: Record<string, string> = {
  "image/png": "png",
  "image/jpeg": "jpg",
  "image/gif": "gif",
  "image/webp": "webp",
  "image/bmp": "bmp",
  "image/svg+xml": "svg",
}

const EXT_TO_MIME: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".bmp": "image/bmp",
  ".svg": "image/svg+xml",
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

let debugOn = !!process.env.IMAGE_VISION_DEBUG

function dbg(objOrFn: unknown) {
  if (!debugOn) return
  const obj = typeof objOrFn === "function" ? (objOrFn as () => unknown)() : objOrFn
  try {
    writeFileSync(DEBUG_LOG, new Date().toISOString() + " " + JSON.stringify(obj) + "\n", { flag: "a" })
  } catch {}
}

function computeTimeoutBySize(base64Length: number, baseTimeout = 120000) {
  const sizeMB = base64Length / (1024 * 1024)
  const extraMs = Math.ceil(sizeMB / 0.5) * 20000
  return Math.min(baseTimeout + extraMs, 300000)
}

/**
 * The analyze_image tool may only read files inside the image temp directory
 * or the current working directory. This prevents path traversal into
 * arbitrary locations on disk.
 */
function isPathAllowed(absPath: string) {
  const resolved = resolve(absPath)
  const allowedRoots = [resolve(TEMP_DIR), resolve(process.cwd())]
  return allowedRoots.some(root => resolved === root || resolved.startsWith(root + sep))
}

function guessMimeFromExt(ext: string) {
  return EXT_TO_MIME[ext.toLowerCase()] || "image/png"
}

/** Extract base64 + mime from a v2 Message media content part. */
function extractMedia(part: any): { base64: string; mime: string } | null {
  const source = part?.media?.source
  if (!source) return null
  if (source.type === "base64" && typeof source.data === "string") {
    return { base64: source.data, mime: source.mediaType || "image/png" }
  }
  for (const field of ["url", "uri"]) {
    const value = source[field]
    if (typeof value === "string" && value.startsWith("data:")) {
      const match = value.match(/^data:([^;]+);base64,(.+)$/)
      if (match) return { base64: match[2], mime: match[1] || "image/png" }
    }
  }
  return null
}

function saveImageToTempDir(base64: string, mime: string, hash: string) {
  mkdirSync(TEMP_DIR, { recursive: true })
  const ext = MIME_TO_EXT[mime] || "png"
  const filePath = join(TEMP_DIR, `${hash}.${ext}`)
  writeFileSync(filePath, Buffer.from(base64, "base64"))
  return filePath
}

function formatElapsed(ms: number) {
  const totalSec = ms / 1000
  if (totalSec < 60) return totalSec.toFixed(1) + "s"
  const h = Math.floor(totalSec / 3600)
  const m = Math.floor((totalSec % 3600) / 60)
  const s = Math.floor(totalSec % 60)
  if (h > 0) return `${h}h ${m}m ${s}s`
  return `${m}m ${s}s`
}

/** Fetch a data: URI into bytes, or null when not decodable. */
function decodeDataUri(uri: string): { base64: string; mime: string } | null {
  const match = uri?.match?.(/^data:([^;]+);base64,(.+)$/)
  return match ? { base64: match[2], mime: match[1] || "image/png" } : null
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

interface PluginOptions {
  /** Vision model used to analyze images, in "providerId/modelId" form. */
  model: string
  /** Custom analysis prompt sent to the vision model. */
  prompt?: string
  /** Base timeout in ms for one vision request; scales up to 300000 for large images. */
  timeout?: number
  /** Write diagnostic events to the debug log. */
  debug?: boolean
  /**
   * Model ids always treated as text-only, even when the model registry
   * (e.g. catalog metadata) claims image input.
   */
  text_only_models?: string[]
  /** Analyze images in parallel when the user submits a prompt (default true). */
  prefetch?: boolean
  /** Maximum output tokens for one vision description (default 256). */
  max_tokens?: number
}

export default Plugin.define({
  id: "image-vision",
  async setup(ctx) {
    const options = ctx.options as PluginOptions
    debugOn = debugOn || options.debug === true

    // ---------------- validate options ----------------
    const model = options.model
    if (!model || typeof model !== "string") {
      throw new Error('Vision model not configured. Set "model" in plugin options (format: providerId/modelId)')
    }
    const slashIdx = model.indexOf("/")
    if (slashIdx === -1) {
      throw new Error(`Invalid model format: "${model}". Expected "providerId/modelId"`)
    }
    const visionModel = { providerID: model.slice(0, slashIdx), modelID: model.slice(slashIdx + 1) }
    const analysisPrompt = options.prompt || DEFAULT_PROMPT
    const baseTimeout = options.timeout ?? 120000
    const forceTextOnly = new Set((options.text_only_models ?? []).map(id => id.toLowerCase()))
    const prefetchEnabled = options.prefetch !== false
    const maxTokens = options.max_tokens ?? 256

    dbg(() => ({ event: "server_init_ok", model, visionModel, prefetchEnabled, maxTokens }))

    // ---------------- shared state ----------------
    const cache = new Map<string, string>()
    const state = {
      hasPendingImages: false,
      pendingFilePaths: [] as string[],
      pendingSessionId: null as string | null,
      processingHashes: new Set<string>(),
      /** Prefetched descriptions ready for the tool to consume. */
      prefetched: new Map<string, string>(),
      /** Prefetch promises in flight, keyed by image hash. */
      inflight: new Map<string, Promise<string>>(),
    }

    const cleanupTasks: Array<() => void> = []

    // ---------------- model capability gate ----------------
    const visionCapable = new Set<string>()
    const refreshVisionCapable = async () => {
      try {
        const result: any = await ctx.model.list()
        const models = (result?.data ?? result ?? []) as any[]
        visionCapable.clear()
        for (const m of models) {
          if ((m?.capabilities?.input ?? []).includes("image")) {
            visionCapable.add(m?.id ?? m?.modelID)
          }
        }
        dbg(() => ({ event: "vision_capable_refresh", count: visionCapable.size }))
      } catch (e: any) {
        dbg(() => ({ event: "vision_capable_refresh_failed", error: e.message }))
      }
    }
    await refreshVisionCapable()

    try {
      const controller = new AbortController()
      void (async () => {
        for await (const event of ctx.event.subscribe({ types: ["model.updated"], signal: controller.signal })) {
          await refreshVisionCapable()
        }
      })()
      cleanupTasks.push(() => controller.abort())
    } catch {}

    // ---------------- dedicated lightweight agent ----------------
    // The vision child session runs under this agent so it never loads the
    // default build agent's large system prompt or tool registry.
    const analyst: any = {
      id: AGENT_ID,
      name: "Image Vision Analyst",
      description: "Describes images for text-only models. Invoked by the image-vision plugin.",
      mode: "subagent",
      hidden: true,
      model: { providerID: visionModel.providerID, id: visionModel.modelID },
      system: analysisPrompt,
      request: { settings: {}, headers: {}, body: {} },
      permissions: [],
      steps: 1,
    }
    await ctx.agent.transform((editor) => {
      editor.update(AGENT_ID, () => {
        // copy definition fields onto the mutable draft
        const draft = editor as any
        for (const key of Object.keys(analyst)) draft[key] = analyst[key]
      })
    })

    // ---------------- core analysis routine ----------------
    /**
     * Analyze one image (base64) against the vision model and return its
     * description. Cached by content hash; concurrent calls for the same
     * hash share one in-flight promise.
     */
    const analyzeBase64 = async (base64: string, mime: string): Promise<string> => {
      const hash = createHash("md5").update(base64).digest("hex").slice(0, 16)
      const cached = cache.get(hash)
      if (cached) {
        dbg(() => ({ event: "cache_hit", hash }))
        return cached
      }
      const existing = state.inflight.get(hash)
      if (existing) return existing

      const promise = (async () => {
        let childId: string | undefined
        try {
          const child = (await ctx.session.create({
            title: "Vision Analysis",
            agent: AGENT_ID,
            model: { providerID: visionModel.providerID, id: visionModel.modelID },
          } as any)) as any
          childId = child?.sessionID ?? child?.id
          if (!childId) throw new Error("Failed to create child session")

          const computedTimeout = computeTimeoutBySize(base64.length, baseTimeout)
          const controller = new AbortController()
          const timeoutId = setTimeout(() => controller.abort(new Error("Vision API timeout")), computedTimeout)

          try {
            await ctx.session.prompt({
              sessionID: childId,
              text: analysisPrompt,
              files: [{ uri: `data:${mime};base64,${base64}`, name: "image", description: "Image to analyze" }],
            } as any)

            const cancelled = new Promise<void>((_, reject) => {
              controller.signal.addEventListener("abort", () => reject(new Error("Cancelled")), { once: true })
            })
            await Promise.race([ctx.session.wait({ sessionID: childId } as any), cancelled])

            const history = (await ctx.session.context({ sessionID: childId } as any)) as any[]
            const msgs = Array.isArray(history) ? history : history?.messages ?? []
            let desc: string | null = null
            for (let i = msgs.length - 1; i >= 0; i--) {
              const m = msgs[i]
              if (m?.type !== "assistant") continue
              const textPart = (m.content ?? []).find((p: any) => p?.type === "text")
              if (textPart?.text) {
                desc = textPart.text
                break
              }
            }
            clearTimeout(timeoutId)

            const final = (desc ?? "").trim() || "[No description returned]"
            cache.set(hash, final)
            state.prefetched.set(hash, final)
            dbg(() => ({ event: "analyze_ok", hash, descLen: final.length }))
            return final
          } finally {
            clearTimeout(timeoutId)
          }
        } finally {
          state.inflight.delete(hash)
          if (childId) {
            try {
              await ctx.session.remove({ sessionID: childId } as any)
            } catch {}
          }
        }
      })()
      state.inflight.set(hash, promise)
      return promise
    }

    // ---------------- prompt hook: parallel prefetch ----------------
    // Fires the moment the user submits a message with image attachments,
    // so vision latency overlaps with the main model's reasoning.
    if (prefetchEnabled) {
      await ctx.session.hook("prompt", (event) => {
        try {
          const files = ((event as any)?.prompt?.files ?? []) as Array<{ uri?: string }>
          for (const f of files) {
            const uri = f?.uri ?? ""
            let decoded: { base64: string; mime: string } | null = null
            if (uri.startsWith("data:")) {
              decoded = decodeDataUri(uri)
            } else if (uri.startsWith("file://")) {
              const filePath = decodeURIComponent(uri.slice("file://".length))
              if (!isPathAllowed(filePath)) continue
              try {
                const buffer = readFileSync(filePath)
                decoded = { base64: buffer.toString("base64"), mime: guessMimeFromExt(extname(filePath)) }
              } catch {
                continue
              }
            }
            if (!decoded) continue
            const hash = createHash("md5").update(decoded.base64).digest("hex").slice(0, 16)
            if (cache.has(hash) || state.inflight.has(hash)) continue
            dbg(() => ({ event: "prefetch_start", hash }))
            void analyzeBase64(decoded.base64, decoded.mime).catch(e => {
              dbg(() => ({ event: "prefetch_failed", hash, error: e.message }))
            })
          }
        } catch (e: any) {
          dbg(() => ({ event: "prefetch_hook_error", error: e.message }))
        }
      })
    }

    // ---------------- context hook ----------------
    // Replaces the v1 "experimental.chat.messages.transform" hook.
    // event.messages is Message[] (@opencode/ai/effect), each { role, content: ContentPart[] }.
    // An image is a part { type: "media", media: { source: { type: "base64", data, mediaType } } }.
    await ctx.session.hook("context", (event) => {
      try {
        const messages = (event.messages as any[]) ?? []
        if (!messages.length) return

        const activeModelId = (event as any)?.model?.id
        dbg(() => ({ event: "gate_model", activeModelId, visionModel }))
        if (activeModelId && activeModelId === visionModel.modelID) return
        if (activeModelId && visionCapable.has(activeModelId) && !forceTextOnly.has(activeModelId.toLowerCase())) {
          dbg(() => ({ event: "skip_vision_capable_model", activeModelId }))
          return
        }

        // Find the last user message
        let lastUser: any = null
        for (let i = messages.length - 1; i >= 0; i--) {
          if (messages[i]?.role === "user") {
            lastUser = messages[i]
            break
          }
        }
        if (!lastUser?.content) return

        const targets: number[] = []
        for (let i = 0; i < lastUser.content.length; i++) {
          const p = lastUser.content[i]
          if (p?.type === "media") targets.push(i)
        }
        if (!targets.length) return

        const sessionId = (event as any)?.sessionID ?? state.pendingSessionId
        if (sessionId) state.pendingSessionId = sessionId

        if (state.pendingFilePaths.length > 0) {
          dbg(() => ({ event: "stale_pending_dropped", count: state.pendingFilePaths.length }))
        }
        state.pendingFilePaths = []
        state.hasPendingImages = false

        for (const index of targets) {
          const part = lastUser.content[index]
          const media = extractMedia(part)
          if (!media) continue

          const hash = createHash("md5").update(media.base64).digest("hex").slice(0, 16)
          if (cache.has(hash)) {
            lastUser.content[index] = { type: "text", text: "[Image analysis result]\n" + cache.get(hash) }
            continue
          }

          let tempPath: string
          try {
            tempPath = saveImageToTempDir(media.base64, media.mime, hash)
          } catch {
            lastUser.content[index] = { type: "text", text: "[Image analysis failed: failed to write temp file]" }
            continue
          }
          state.pendingFilePaths.push(tempPath)
          lastUser.content[index] = {
            type: "text",
            text: `[An image was pasted. Call the analyze_image tool with file_path="${tempPath}" to get the image description.]`,
          }
          state.hasPendingImages = true
        }

        dbg(() => ({
          event: "transform_complete",
          pending: state.pendingFilePaths.length,
          cached: targets.length - state.pendingFilePaths.length,
        }))
      } catch (e: any) {
        console.error(`[image-vision] transform failed: ${e.message}`)
      }
    })

    // ---------------- analyze_image tool ----------------
    await ctx.tool.transform((editor) => {
      editor.add({
        name: "analyze_image",
        description:
          "Analyze images in the conversation. Returns detailed text descriptions including text content, UI layout, colors, and code details.",
        input: {
          type: "object",
          properties: {
            file_path: {
              type: "string",
              description: "Path to the image file to analyze (provided automatically)",
            },
          },
          additionalProperties: false,
        },
        execute: async (input: any, context: any) => {
          const startTime = Date.now()
          const filePaths = input?.file_path ? [input.file_path] : [...state.pendingFilePaths]
          if (!filePaths.length) return { content: "No images to analyze." }

          dbg(() => ({ event: "execute_paths", argsPath: input?.file_path, pendingCount: state.pendingFilePaths.length }))

          // Analyze all images in parallel; each is cached/deduped by hash.
          const results = await Promise.all(
            filePaths.map(async filePath => {
              if (!isPathAllowed(filePath)) return "[Analysis failed: path not allowed]"

              let base64: string, mime: string
              try {
                const buffer = readFileSync(filePath)
                base64 = buffer.toString("base64")
                mime = guessMimeFromExt(extname(filePath))
              } catch (e: any) {
                const msg = `[Analysis failed: cannot read file ${filePath}: ${e.message}]`
                dbg(() => ({ event: "tool_analyze_fail", error: e.message }))
                return msg
              }

              const hash = createHash("md5").update(base64).digest("hex").slice(0, 16)
              const prefetched = state.prefetched.get(hash)
              if (prefetched) {
                dbg(() => ({ event: "prefetch_used", hash }))
                state.prefetched.delete(hash)
                return prefetched
              }
              if (cache.has(hash)) return cache.get(hash) as string

              try {
                return await analyzeBase64(base64, mime)
              } catch (e: any) {
                dbg(() => ({ event: "tool_analyze_fail", error: e.message }))
                return `[Analysis failed: ${e.message}]`
              }
            }),
          )

          if (!input?.file_path) state.pendingFilePaths = []
          if (state.pendingFilePaths.length === 0) {
            state.hasPendingImages = false
            state.pendingSessionId = null
          }

          const elapsed = Date.now() - startTime
          const analyzedCount = results.filter(r => !r.startsWith("[Analysis failed")).length
          dbg(() => ({ event: "tool_complete", count: results.length, analyzedCount, elapsedMs: elapsed }))
          const output =
            results.length === 1
              ? results[0]
              : results.map((desc, i) => `### Image ${i + 1}\n\n${desc}`).join("\n\n---\n\n")
          if (analyzedCount > 0) {
            dbg(() => ({ event: "tool_summary", analyzed: analyzedCount, elapsed: formatElapsed(elapsed) }))
          }
          return { content: output }
        },
      })
    })

    dbg(() => ({ event: "setup_complete" }))

    return () => {
      for (const fn of cleanupTasks) {
        try {
          fn()
        } catch {}
      }
    }
  },
})
