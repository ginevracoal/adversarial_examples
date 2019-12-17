## Adversarially Robust Training using Random Projections
Focusing on image classification tasks, we use an intuition behind random projections of the inputs as a defence and we 
propose a training technique, called **Random Projections Ensemble**, which improves the robustness to adversarial examples
 of any given classifier. This method projects the input data in multiple lower dimensional spaces, each one corresponding 
 to a random selection of directions in the space. Then it trains a new classifier in each subspace, using the corresponding 
 projected version of the data. Finally, it performs an ensemble classification on the original high dimensional data. 
Then we propose a regularization method for training an adversarially robust classifier, called **Random Projections 
Regularizer**. Both methods are meant to be attack independent.

We evaluate adversarial vulnerability of the resulting trained models and compare them to robust models given by
adversarial training technique. 
We compare the ensemble method to the regularization method in terms of performances and computational efficiency. 
Scalability and parallelizability.

## Implementations specs

**Models**:
- the baseline for our tests is `baseline_convnet`, a basic CNN trained on mnist.
- `random_ensemble` computes `n_proj` random projections of the training data in a lower dimensional space 
(whose dimension is `size_proj`^2), then classifies the original high dimensional data with a voting technique on the 
single classifications.
- `random_regularizer` is a regularization method based on the computation of loss gradients over input data 
projections.

**Main libraries**: 
- `Tensorflow`, `Keras` for model training
- IBM `adversarial-robustness-toolbox` and `cleverhans` for attacks generation

## Plots and results

Inverse projections

<img src="../../results/images/mnist_randens_proj=1_size=25.png" width="420"/> <img src="../../results/images/cifar_randens_proj=1_size=28.png" width="420">

Adversarial robustness

<img src="../../results/images/mnist_randens_adversarial_accuracy.png" width="400"/> <img src="../../results/images/cifar_randens_adversarial_accuracy.png" width="400">


Computational complexity

<img src="../../results/images/mnist_randens_complexity.png" width="400"/> <img src="../../results/images/cifar_randens_complexity.png" width="400">

Perturbed images

<img src="../../results/images/cifar_perturbations.png" width="400"/> <img src="../../results/images/cifar_perturbations_plot.png" width="400">
