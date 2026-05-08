📝 Additional Notes
===================

🧠 System and Prompt Tokens
---------------------------

System messages include both explicit system turns and internal prompts added during model setup.

⚙️ Memory and Performance
-------------------------

During explanation, package stores combined embeddings (after normalization) and mask data used for SHAP sampling.
Most intermediate data is dropped when ``verbose=False``. Even then, long conversations and large models can require significant memory.

For larger workloads, prefer approximation strategies (for example Monte Carlo variants with controlled sample budgets).

🔗 External Embeddings
----------------------

To compare behaviors across models, use external embedding implementations from ``mllm_shap.shap.embeddings``.
Embedding calls run sequentially after each generation step (no batching by default), so embedding model throughput directly impacts total runtime.

.. warning::

    Some features are experimental and may change in future releases. Always check the `examples/` folder for updated usage patterns.
