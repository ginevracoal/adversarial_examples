# Master thesis project on adversarial examples

Implementations of classifiers robust to adversarial examples, based on random projections.

- `baseline_convnet` is a basic CNN trained on mnist
- `random_ensemble` computes n_proj random projections of the training data in a lower dimensional space,
then classifies the original high dimensional data with a voting technique on the single classifications.

## Folders structure

- data
- notebooks
- src
- trained_models
    - IBM-art

