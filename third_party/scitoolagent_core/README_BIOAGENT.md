# SciToolAgent official core

Repository: https://github.com/HICAI-ZJU/SciToolAgent

Pinned commit: ac1cf19fcef84e69f149db6da424afe4e9b2f11f

Files listed in UPSTREAM.json are exact Git blob bytes. Hash verification occurs
before loading. MIT license retained. No upstream evaluation dataset is loaded.

The original KGTableRetriever module is imported directly. Original function
bodies for graph construction, planning, parameter extraction, plan/answer
parsing, execution/retry and run_SciToolAgent are compiled from their AST without
rewriting their bodies. This avoids upstream global SDK/config initialization.

The protein adapter supplies a public 20-tool KG through original create_triplets
and build_knowledge_graph. Local existing BGE-small-zh weights replace upstream
OpenAI embeddings; the original graph traversal and similarity weighting remain.
Embedding worker concurrency is capped at five. LlamaIndex 0.14.18 is the local
compatible core, rather than the original meta-package 0.11.3; runtime versions
are archived.

Original Input: protocol is retained. The API input format is one JSON object
with public parameter names, replacing the original domain-specific single
string/dot-separated multiple inputs. A lone string is accepted only for tools
with one required argument. Tool results are literal JSON text for the original
executor to extract from. No typed artifact pool or automatic conversion is used.

All entries are existing read-only computational analysis tools (Safety=Yes);
the same backend safety checks apply as to other baselines. Extra chemical-tool
safety hooks are not configured and raise if reached. Original get_tool_info
security matching is not repaired or repurposed for identity control.

Provider and resource controls are external hooks. All online LLM calls use the
designated backbone and common budget, including retries and summarization.
BudgetStop interrupts upstream retry loops without changing their algorithms.
No scientific success is inferred from native IsCompleted:Completed.
