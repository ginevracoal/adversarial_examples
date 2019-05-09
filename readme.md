# On the robustness of adversarial examples
## Master thesis project 

Implementations of robust classifiers in the context of adversarial examples.
The proposed methods are based on the computation of random projections and the use of `Keras` and 
IBM `adversarial-robustness-toolbox`.

Models:
- The baseline for our tests is `baseline_convnet`, a basic CNN trained on mnist.
- The model `random_ensemble` computes n_proj random projections of the training data in a lower dimensional space,
then classifies the original high dimensional data with a voting technique on the single classifications.

## Folders structure

- data
- notebooks
- src
- trained_models
    - IBM-art

