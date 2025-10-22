Additional Notes
================

System messages include not only system turns defined on input, but also any predefined prompts used to initialize the model and any steering tokens or instructions added by the model itself.

Package during explaination stores each result combined embedding (after processing with normalizer, refer to `mllm_shap.shap.normalizers` module) and mask of tokens to exclude from SHAP calculation. All other data is discarded to save memory (if using `verbose=False`). Yet it still might be memory-intensive for long conversations or large models, so monitor your memory usage accordingly and consider using less accurate approximations like Monte Carlo SHAP with limited number of samples for larger workloads.

If you'd like to compare different models behaviour on same inputs, consider calculation of SHAP values using 3rd party model's embeddings - refer to `mllm_shap.shap.embeddings` implementations for that. Note that for memory efficiency reasons, call to embedding calculation is done after each step of generation, (no batching). All operations are sequential, therefore embedding model speed affects overall explaination time directly.

.. warning::
    Some features are experimental and may change in future releases. Always check the `examples/` folder for updated usage patterns.
