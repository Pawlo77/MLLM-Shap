References
==========

MLLM-SHAP is used for research purposes on behaviour of large language models (LLMs) and their
explainability using SHAP values. Research paper and report is available at `arXiv <https://arxiv.org/im_not_exist_yet>`_. All code is publicly available at `official Github repository <https://github.com/Pawlo77/MLLM-Shap/tree/main/experiments>`_ and can be used as further example on how to use package and how to interpret its results.

Package is fully tested, including analysis of correctness on different approximations algorithms of SHAP values, available `here <https://github.com/Pawlo77/MLLM-Shap/tree/main/mllm_shap/tests/approximations>`_. Following image shows comparison of different SHAP approximations on same input and model - Liquid Audio LLM.

.. warning::

    Approximation methods might not work well with very small number of samples / input lengths, as many of them are limited from generating not unique masks. This might lead to "dead-locks" in sampling and wrong results.

# TODO: add image
