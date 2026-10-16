# Deep Learning for NLP 

Python/Jupyter learning examples for foundational neural networks and language-processing tools. **This repository currently demonstrates a single-neuron flower classifier, not a trained deep network or a flower-species recognition system.** The red/blue labels are synthetic classroom categories.

## Contents

| Path | Purpose |
|---|---|
| `notebooks/flowers_classification.ipynb` | Self-contained, reproducible single-neuron flower classification walkthrough (recommended) |
| `src/flower_classifier.py` | Tested NumPy implementation (stable sigmoid, binary cross-entropy, full-batch gradient descent) |
| `tests/test_flower_classifier.py` | Unit tests for numerical stability, determinism and prediction behavior |
| `docs/technical_review.md` | Technical review, corrections, limits and verification evidence |
| `docs/Flowers_Classification_Using_Artificial_Neural_Network.pdf` | Original supporting explanation (Hindi) |
| `archive/flowers_original.ipynb` | Original uploaded class notebook, preserved for reference |
| `archive/flowers_checkpoint_original.ipynb` | Original checkpoint, preserved only for comparison |
| `requirements.txt` | Runtime dependencies for the classifier |
| `docs/gitignore_additions.txt` | Optional new ignore patterns; preserve existing repository `.gitignore` |

The current GitHub root also contains earlier `flowers.ipynb`, `flowers-checkpoint.ipynb`, and `12.png`. Keep those existing files until the new version is reviewed; the recommended clean notebook is in `notebooks/`.

## Quick start: Google Colab

1. Open `notebooks/flowers_classification.ipynb` on GitHub and choose **Open in Colab**, or upload it at https://colab.research.google.com/.
2. Choose **Runtime → Run all**. It uses built-in NumPy/Matplotlib and needs no external dataset.
3. Review the scatter plot, training loss, prediction and decision-boundary plot.
4. This notebook is self-contained. You do **not** need to clone the repository for the Colab exercise.

## Local setup

```bash
python -m venv .venv
# On Linux/macOS: source .venv/bin/activate
# On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

### Model

A single sigmoid neuron computes `p = sigmoid(w1*length + w2*width + bias)` and classifies with threshold 0.5. Training uses seeded, full-batch gradient descent with numerically stable binary cross-entropy (BCE). Class 0 = blue and class 1 = red.

The **eight training examples** are illustrative only. Training-set accuracy does not estimate test accuracy. The mystery flower is `[4.5, 1.0]`, which can be classified after fitting; its score is **not** a benchmark accuracy result. The previously saved `0.991729...` probability came from a *randomly initialized original notebook run* and need not be reproduced exactly.

For formulas, comparative observations, and test limitations, see [`docs/technical_review.md`](docs/technical_review.md).

## Limitations and next steps

- Not a multi-layer perceptron or full image classification workflow; inputs are two numerical features.
- Dataset is extremely small and has no held-out test set; add real data, validation, and appropriate evaluation metrics before reporting generalization.

## Coursework and privacy

This repository is public. Review notebook outputs, file paths and annotations before publishing, and avoid uploading confidential datasets, credentials or graded coursework without permission. Uploaded source notebooks are retained as references; their existing outputs are **not** evidence of fresh reproducible execution.
