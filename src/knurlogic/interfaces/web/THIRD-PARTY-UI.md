# Third-party UI credit

The Chat panel in `interfaces/web/index.html` ports several small,
well-scoped pieces of behaviour from **exo's dashboard**
(`exo/dashboard/src/lib`), licensed Apache-2.0:

* the SSE line parser (`stores/app.svelte.ts:2148-2220`, `parseSSEStream`);
* the `<think>...</think>` streaming splitter (`:2107-2140`);
* the PDF-to-page-images routine and its constants (`types/files.ts`);
* the prefill-progress-bar percent/ETA math (`PrefillProgressBar.svelte:17-30`);
* the history-to-OpenAI-multimodal message mapping (`:2402-2446`);
* the image-lightbox download-extension-from-MIME-type logic
  (`ImageLightbox.svelte`);
* the paste-over-2500-chars-becomes-a-file rule (`ChatForm.svelte`).

These are the near-verbatim ports named in
`docs/design/vision-evidence/report-chat-ui.md` section 5. Everything else
in the panel (storage, the markdown renderer, the attachment pipeline, the
model picker) is knurlogic's own.

Per exo's Apache License 2.0, this file records that the above pieces are
derived from exo's dashboard source and carry its copyright; no NOTICE file
changes were required by exo's own repository at the time of porting.

See also `engine/architectures/THIRD-PARTY.md` for architecture-level
attributions (a separate concern: model code, not UI).
