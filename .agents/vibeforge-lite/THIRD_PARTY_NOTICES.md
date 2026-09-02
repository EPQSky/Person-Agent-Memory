# Third-Party Notices

## mattpocock/skills

Parts of `skills/` are derived from the `mattpocock/skills` repository. The authoritative release baseline is `v1.2.3`, resolved to commit `6acc160e4e0cd062dbbbd7a1b26ae92855edf07e`. VibeForge Lite exposes that release's batched `productivity/grilling` behavior through the explicit `batch-grill-me` name while retaining its sequential `grilling` default.

Copyright (c) 2026 Matt Pocock

Licensed under the MIT License. The complete upstream license is preserved at `licenses/MIT-mattpocock.txt`; imported paths, local names, and modifications are recorded in `UPSTREAM.lock`.

`batch-grill-with-docs`, `execute-spec-tickets`, `vibe-guide`, and the executable `vibe-init` behavior are maintained by this project. `execute-spec-tickets` was supplied for inclusion as archive SHA-256 `b33360071acffb8ad7613394c4935b6e28dfab33ce9e8e30004562a61477e3f1`; its repository integration and test placement are recorded in `UPSTREAM.lock`. VibeForge Lite imports a curated subset of `v1.2.3`; paths outside that product boundary are listed as not imported in `UPSTREAM.lock`. All vendored skills include local Codex compatibility changes and are not presented as unmodified upstream copies. Local adaptations also preserve a sequential default grilling rhythm with separate batch skills, make code review cover the full working tree, add explicit trust boundaries for external tracker content, constrain spec synthesis to one testing-seam confirmation, keep parent specs out of implementation-ready state, and let TDD reuse seams already approved in a spec or ticket.

## OpenAI Codex Plugin validator

CI downloads the Plugin validator and its identifier helper from a fixed commit of the OpenAI `codex` repository and executes them without redistributing them in this project's Plugin archive. The repository, commit, and paths are recorded in `UPSTREAM.lock`; the upstream project is licensed under Apache-2.0.
