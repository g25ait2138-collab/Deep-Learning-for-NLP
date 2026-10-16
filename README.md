# Deep Learning for NLP

Python/Jupyter learning examples for foundational neural networks. The current repository contains an educational single-neuron flower classifier using two numeric inputs. The red/blue flower classes are synthetic teaching labels, **not real flower species**. It is **not** a trained deep network or an image classification system.

## Currently available files

| File | Purpose |
|---|---|
| [`flowers.ipynb`](flowers.ipynb) | Original single-neuron flower classification classroom notebook; run-specific outputs may vary |
| [`flowers-checkpoint.ipynb`](flowers-checkpoint.ipynb) | Earlier checkpoint preserved for comparison; not the recommended final reproducible version |
| [`12.png`](12.png) | Previously uploaded image asset |
| [`.gitignore`](.gitignore) | Python/Jupyter ignore rules |

## How to run the existing demonstration

1. Open [`flowers.ipynb`](flowers.ipynb) and choose **Open in Colab**, or download the notebook and upload it to [Google Colab](https://colab.research.google.com/).
2. Select **Runtime → Run all**.
3. Review the input features, sigmoid neuron, training loss, plots and example prediction.

The model uses numeric flower **length** and **width** as its two features and predicts illustrative class 0 (blue) or class 1 (red). It does not learn from flower photographs. The training set is very small; training performance does not establish real-world generalization.

## Planned reproducible version (not yet uploaded)

A reviewed upload package has been prepared separately, with `notebooks/flowers_classification.ipynb`, a tested NumPy implementation under `src/`, `tests/`, `docs/`, and a `requirements.txt`. Those paths **will not work on GitHub until the package is uploaded**. Publish those files before switching this README to their links.

## Coursework and privacy

This public repository is a learning demonstration. Review any notebook output, file path, and uploaded source material for private information or course restrictions before publishing. Do not treat previously saved notebook outputs as proof of a fresh reproducible run.
