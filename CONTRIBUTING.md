# Contributing

Run the relevant offline tests before changing runtime, registry or adapter behavior. Keep tool names, scientific input types and artifact identity/provenance contracts explicit. A failed tool result must not register a usable downstream artifact.

Historical datasets, official-source hashes, frozen manifests and archived results are immutable. Changes to methods, budgets, prompts, schemas or evaluation rules require a separately versioned configuration and corresponding documentation. Keep natural recovery and controlled repair evidence separate.

Do not commit credentials, uploaded inputs, model weights, local indexes, runtime logs or manuscript-management files. Retain upstream license/provenance notices for imported code. Describe the change and the relevant offline checks in a pull request.
