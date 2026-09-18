# Third-party source vendored here

Every `.py` in this directory is a copy of a model implementation from
another project, taken deliberately and recorded in `PROVENANCE.md`. They are
loaded into `sys.modules` by `knurlogic.register`; nothing is written into
anyone's `site-packages`.

## Licenses

| file | upstream project | license |
|---|---|---|
| `gemma4_text.py` | mlx-lm | MIT — Copyright © 2025 Apple Inc. |
| `qwen3_5.py` | mlx-lm | MIT — Copyright © 2026 Apple Inc. |
| `qwen3_5_moe.py` | mlx-lm | MIT — Copyright © 2026 Apple Inc. |
| `qwen4_exp.py` | mlx-lm (via github.com/eauchs/mlx-lm) | MIT |
| `glm5_next/` | mlx-vlm | MIT |

Some files carry no copyright header upstream; they are covered by their
project's MIT license regardless, and are listed here so the obligation is
discharged in one place.

## MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Replacing these

The plan is to own these implementations rather than carry copies. A file
replaced by an independent implementation should be removed from the table
above in the same commit that replaces it — leaving a stale attribution is
its own kind of wrong.
