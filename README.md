# MLLM SHAP

# Developer notes

#### Deploying

Deployment to PyPi is handled from github workflow automatically.

```bash
git tag vx.x.x main
git push origin --tags
```

#### Updating docs for new modules

From `~/mllm_shap` directory:

```bash
sphinx-apidoc -o docs/ src     
```

This will generate `.rst` files with definition of new modules. To update and locally analyze their docs, do from `~/mllm_shap/docs` directory:

```bash
make clean && make html
```

Home page file is `~/mllm_shap/docs/_build/html/index.html`
