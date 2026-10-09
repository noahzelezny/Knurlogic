# Images

Qwen 3.5 / 3.6, Qwen 3.8, Gemma 4 and GLM-5 models read images as well as
text. DeepSeek-V4 is text only. `GET /v1/models` lists `vision` in
`capabilities` when the loaded model takes images.

## Sending an image

OpenAI shape, one or more `image_url` parts per message:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "local", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "What is in this picture?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0..."}}]}]}'
```

Anthropic shape: an `image` block with `"source": {"type": "base64",
"media_type": "image/png", "data": "..."}`. Ollama: `images` in a chat
message.

In the page's chat, attach an image with the paperclip button beside the
message box.

## Limits

| | |
|---|---|
| image URLs (`http://`, `https://`) | not fetched; send the bytes as a `data:` URL |
| encoded size | at most 32 MiB per image |
| very large images | larger than 4096 x 4096 pixels are scaled down; a decompression bomb is refused |
| an image over the model's own pixel limit | refused with 413, naming the limit |
| image memory | a request's images must fit the image store together (0.25 GiB by default; `knurlogic serve --image-store-gib`) |

## Images and the cache

An image is encoded once per conversation. The prompt cache keeps it: the
next turn of the same conversation does not read the image again. The
same picture sent again (even re-encoded) is recognised by its pixels.

## Turning vision off

A model you only send text can launch without its vision tower: that
memory (the tower, the image store and the image cache allowance) goes to
room for conversations.

- the page: **Vision Off** in Load model;
- MCP: `load(..., vision=false)`, and `fit(..., vision=false)` to ask first;
- the CLI: `knurlogic serve <model> --set KNURLOGIC_VISION=off`.

A request with an image then gets a 400 saying vision is off for this
launch.

## On a cluster

On a model split across Macs, rank 0 (the machine that answers requests)
holds the vision tower and encodes the images. See [clusters.md](clusters.md).

Design: [../design/vision.md](../design/vision.md),
[../design/vision-contracts.md](../design/vision-contracts.md).
