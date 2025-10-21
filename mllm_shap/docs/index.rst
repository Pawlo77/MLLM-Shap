.. mllm-shap documentation master file, created by
   sphinx-quickstart on Tue Oct 21 11:09:55 2025.

Welcome to MLLM-SHAP
=====================

MLLM-SHAP is a Python package designed to **interpret the predictions of large language models (LLMs) using SHAP (SHapley Additive exPlanations) values**.
It helps you understand the contribution of input features to model outputs, enabling **transparent and explainable AI workflows**.

Key features:

- Integration with **audio and text models**, supporting multi-modal inputs and outputs.
- Flexible aggregation strategies: mean, sum, max, min, etc.
- Multiple similarity metrics (cosine, euclidean, etc.) for embedding analysis.
- Customizable SHAP calculation algorithms: exact, Monte Carlo approximations, and more.
- Examples showcasing common explainability pipelines in `examples/` on official GitHub repository.

If you are interested with gui visualization of SHAP values, please check `Extension - GUI Visualization <#id1>`_. For more advanced cli usages, refer to `official Github repository examples <https://github.com/Pawlo77/MLLM-Shap/tree/main/examples>`_ or more advanced pipelines developed as part of exemplary `research projects <https://github.com/Pawlo77/MLLM-Shap/tree/main/experiments>`_.

Supported LLM integrations:

- `Liquid-Audio <https://github.com/Liquid4All/liquid-audio/>`_

Getting Started
===============

.. include:: getting_started.rst

Documentation
=============

.. x
.. ----
.. .. automodule:: mllm_shap.x
..    :members:
..    :undoc-members:
..    :show-inheritance:

Release Notes
=============

.. include:: release_notes.rst

Additional Notes
================

.. include:: addidional_notes.rst

References
==========

.. include:: references.rst

Extension - GUI Visualization
=============================

.. include:: extension_gui.rst

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
