# opencode-image-vision

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenCode](https://img.shields.io/badge/OpenCode-v2.2%2B-blue)](https://opencode.ai)
[![Node](https://img.shields.io/badge/Node.js-18%2B-green)](https://nodejs.org)

An [OpenCode](https://opencode.ai) v2 plugin that gives **text-only models** (e.g. Atria-Dawn, GLM-5, DeepSeek V4, MiniMax) the ability to understand **pasted images**.

When an image is attached to a conversation, the plugin replaces it with a short text instruction and registers an `analyze_image` tool. The model calls the tool, which sends the image to a separate **vision model** in a child session and returns a detailed text description — so the main model can keep working without ever needing image input support.

**Paste → ask → done.**

---

## How it works

1. **`chat.context` transform** — When a user message contains an image part, the plugin saves the image to a temp directory (`<tmpdir>/iv-images/`) and replaces the media part with a text instruction telling the model to call `analyze_image` with the file path. Identical images are MD5-cached, so repeats are answered instantly.
2. **`analyze_image` tool** — Reads the image from disk, opens a child session pinned to the configured vision model, sends the image, and waits for the description.
3. **Result** — The description is returned as tool output and cached for the rest of the conversation.

The plugin automatically skips transformation when the active model already accepts image input, so it never double-processes a conversation running on a vision-capable model.

## Requirements

- **OpenCode v2.2 or newer** (uses the v2 plugin API — it will not load on OpenCode v1)
- **Node.js 18+**
- A **vision-capable model** configured in OpenCode, and credentials for its provider (e.g. `amd/MiniCPM-V-4.7-1B`, `zhipuai-coding-plan/glm-4.6v`, `deepseek/deepseek-v4-flash-vision`)

## Installation

### Option A — global plugin directory (recommended)

```bash
git clone https://github.com/jazlihamizan/opencode-image-vision.git \
  ~/.config/opencode/plugins/image-vision
cd ~/.config/opencode/plugins/image-vision
npm install
```

Then add the plugin to `~/.config/opencode/opencode.json`:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "plugins": [
    {
      "package": "file:///home/<you>/.config/opencode/plugins/image-vision",
      "options": {
        "model": "amd/MiniCPM-V-4.7-1B"
      }
    }
  ]
}
```

The plugin is also auto-discovered when placed in a project's `.opencode/plugins/` directory, in which case the `plugins` entry above is only needed to pass options.

### Option B — from npm

```jsonc
{
  "plugins": [
    {
      "package": "@jazlihamizan/opencode-image-vision",
      "options": { "model": "amd/MiniCPM-V-4.7-1B" }
    }
  ]
}
```

Save the config — OpenCode hot-reloads plugins, or run `/reload`.

## Options

| Option    | Required | Default  | Description |
| --------- | -------- | -------- | ----------- |
| `model`   | Yes      | —        | Vision model in `providerId/modelId` form, e.g. `amd/MiniCPM-V-4.7-1B` |
| `prompt`  | No       | built-in | Analysis prompt sent to the vision model |
| `timeout` | No       | `120000` | Base timeout in ms for one vision request; scales up with image size up to `300000` |
| `debug`   | No       | `false`  | Write diagnostic events to `<tmpdir>/iv-debug.log` |

Debug can also be enabled with the `IMAGE_VISION_DEBUG=1` environment variable (`IMAGE_VISION_DEBUG_PATH` overrides the log path).

`model` is **required** — the plugin fails to load with a clear message when it is missing or not in `providerId/modelId` form.

## Choosing a vision model

Any vision-capable model works. Two practical notes:

- **Smaller is faster.** In testing, `MiniCPM-V-4.7-1B` answered in ~20s per image, while the 35B variant of the same family needed ~5 minutes. Default to a compact model unless you need the extra accuracy.
- The model must be listed by its provider in your OpenCode configuration, and you must be signed in to that provider (`opencode auth`).

## Security & privacy

- **No credentials are stored or handled by the plugin.** Provider authentication is managed entirely by OpenCode (`auth.json`, environment variables, or OAuth) — the plugin only talks to OpenCode's own session API.
- **Path traversal guard.** `analyze_image` only reads files inside `<tmpdir>/iv-images/` or the current working directory.
- **Images leave your machine** through the vision provider you configure — the same trust model as any other OpenCode model request. Do not paste images you cannot send to that provider.
- Temp files persist in `<tmpdir>/iv-images/` (preserved for re-analysis in follow-up turns). Delete the directory to clear them.

## Troubleshooting

| Symptom | Check |
| ------- | ----- |
| Plugin does not load | `model` must be set in plugin options; look for `failed to load plugin` in `~/.local/share/opencode/log/opencode.log` |
| `[Analysis failed: path not allowed]` | The tool only reads from the temp directory or the working directory |
| `[Analysis failed: Vision API timeout]` | Increase `timeout`, or use a smaller/faster vision model |
| `[No description returned]` | The vision model returned no text — verify the model is vision-capable and the provider is signed in |
| Model ignores pasted images | The active model must support tool calling to invoke `analyze_image` |

Enable `debug: true` (or `IMAGE_VISION_DEBUG=1`) and inspect `<tmpdir>/iv-debug.log` for detailed events.

## Development

```bash
git clone https://github.com/jazlihamizan/opencode-image-vision.git
cd opencode-image-vision
npm install
```

The plugin is plain TypeScript loaded directly by OpenCode — no build step. Edit `index.ts`, then `/reload` in OpenCode (or restart the service) to pick up changes.

```bash
# Run with debug events
IMAGE_VISION_DEBUG=1 opencode
```

Expected debug output (`<tmpdir>/iv-debug.log`):

```
{"event":"server_init_ok","model":"amd/MiniCPM-V-4.7-1B", ...}
{"event":"gate_model","activeModelId":"Atria-Dawn", ...}
{"event":"transform_complete","pending":1,"cached":0}
{"event":"tool_analyze_ok","descLen":1457}
```

## Compatibility

OpenCode v2 changed the plugin API in a breaking way (`Plugin.define` with `setup`, replacing the v1 `server` function). This plugin targets **v2 only**. It is a port of the v1 plugin [`@showlotus/opencode-image-vision`](https://github.com/showlotus/opencode-image-vision) (MIT, by showlotus) to the v2 API.

## Contributing

Issues and pull requests are welcome at [github.com/jazlihamizan/opencode-image-vision](https://github.com/jazlihamizan/opencode-image-vision/issues).

## License

MIT — see [LICENSE](LICENSE).
