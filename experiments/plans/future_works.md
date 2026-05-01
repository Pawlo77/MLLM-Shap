# Analysis of SV for text

- create model that real-time (every call or k calls) predicts what are the best coalitions to be sampled given the sizes, an extension to the Neyman allocation that learns to predict which coalitions are most informative for SV estimation, rather than relying on a fixed heuristic. This would be a more complex fix but could yield better sample efficiency if successful. Model might include info on the feasibility scores - there many different approaches should be explored, from simple regression to more complex bandit algorithms.

- Analyst more T2T models, possibly with different architectures, sizes or training data, to see if the identified issues are consistent across models or if some models are more robust to certain issues.

- Analyze different languages, to see if the issues identified are language-specific or if they generalize across languages. This could involve testing on models trained on different languages or multilingual models.

- Analyze the approximation methods in more depth, to see if there are specific conditions under which certain methods perform better or worse. For example, does the Neyman allocation perform better for certain types of features or coalitions? Are there specific patterns in the data that lead to higher variance in the MC estimator?

- Analyze different grouping strategies for the coalitions, to see if certain groupings lead to more stable or interpretable SV estimates. For example, grouping by linguistic units (e.g., words, phrases). Larger groups might also bring significant computational savings, but the tradeoff between granularity and interpretability would need to be explored. *Sentence-level alignment might be lean towards the Hierarchical Shapley approach, where we first compute SVs for larger groups (sentences) and then drill down into smaller units (words) within the most important groups.*

- Add utility score that will compare all output distributions (not just the base sequence) to capture distributional shifts under coalition perturbations. This could be a KL divergence or Wasserstein distance between the full output distributions, providing a richer signal than just the base sequence similarity.

# GUI enhancements

- Add a feature to visualise faithfulness results in a more interactive way.

# How the LLM's understand different languages?

# How the LLM's understand different modalities?

# How fine-tuning on different datasets affects the LLM's understanding of prescribed tasks?

# How reframing the input affects the LLM's understanding of the task?
