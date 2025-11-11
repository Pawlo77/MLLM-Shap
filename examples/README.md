This folder contains example use cases of the `audio-shap` package. Most examples utilize the `liquid_audio` model (see the [connector](./../mllm_shap/src/mllm_shap/connectors/liquid/)).

Each notebook can also serve as a **manual test** to verify the package’s correct functionality—especially useful for those developing custom connectors or extending the package’s capabilities.

- **[Multi Turn Text example](./text_multi_turn.ipynb)** — a recommended starting point demonstrating module usage with both exact Shapley value computation and Monte Carlo approximation. It includes a two-turn conversation without system prompts.
- **[Monte Carlo Text example](./text_monte_carlo.ipynb)** — a continuation of the *Multi Turn Text example*, showing both turns computed using a minimal Monte Carlo setup (very limited calls, resulting in low accuracy).
- **[Monte Carlo Internal Audio example](./audio_internal.ipynb)** — a showcase how the package can be used for expandability in multi modal multi turn conversation, where model is feed with text data from user and returns audio. Example covers how to force model to return audio only and how to include that output in next turn expandability.
- **[Monte Carlo Audio example](./audio_external.ipynb)** - example that combines all possible modalities (audio in and out, text in).
