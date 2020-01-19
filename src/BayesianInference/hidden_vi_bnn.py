import sys
sys.path.append(".")
from directories import *
import pyro
from BayesianInference.hidden_bnn import HiddenBNN
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from utils import *
import pyro.optim as pyroopt
from BayesianInference.adversarial_attacks import *
from BayesianInference.pyro_utils import data_loaders
import time
# from utils import execution_time
import argparse


class VI_BNN(HiddenBNN):
    def __init__(self, dataset_name, input_shape, device):
        self.input_size = input_shape[0]*input_shape[1]*input_shape[2]
        self.n_classes = 10
        self.dataset_name = dataset_name
        super(VI_BNN, self).__init__(input_size=self.input_size, device=device)

    def get_filename(self, n_inputs, lr, n_epochs):
        return "hidden_vi_" + str(self.dataset_name) + "_inputs=" + str(n_inputs) + \
                "_lr=" + str(lr) + "_epochs=" + str(n_epochs)

    def infer_parameters(self, train_loader, lr, n_epochs):
        random.seed(0)

        filename = self.get_filename(n_inputs=len(train_loader.dataset), lr=lr, n_epochs=n_epochs)
        print("\nSVI inference for ", filename)
        # optim = pyroopt.SGD({'lr': lr, 'momentum': 0.9, 'nesterov': True})
        optim = pyroopt.Adam({"lr": lr})
        elbo = Trace_ELBO()
        svi = SVI(self.model, self.guide, optim, loss=elbo)

        loss_list = []
        accuracy_list = []
        start = time.time()
        for i in range(n_epochs):
            accuracy = 0
            total_loss = 0.0
            total = 0.0
            correct = 0.0

            for images, labels in train_loader:
                images.to(self.device)
                labels.to(self.device)
                # svi.step() = take a gradient step on the loss function
                # images and labels are passed to model() and guide()
                loss = svi.step(inputs=images.view(-1,self.input_size), labels=labels)
                total_loss += loss / len(train_loader.dataset)
                total += labels.size(0)
                # forward computes the average output on n_samples samples of the network
                pred = self.forward(images.to(self.device), n_samples=1).mean(0).argmax(-1)
                correct += (pred == labels.argmax(-1).to(self.device)).sum().item()
                accuracy = 100 * correct / total

            print(f"[Epoch {i + 1}]\t loss: {total_loss:.2f} \t accuracy: {accuracy:.2f}")
            loss_list.append(total_loss)
            accuracy_list.append(accuracy)
        execution_time(start=start, end=time.time())

        print("\nlearned params =", list(pyro.get_param_store().get_all_param_names()))
        self.save(filename=filename)

        plot_loss_accuracy({'loss':loss_list, 'accuracy':accuracy_list}, path=RESULTS + "bnn/" + filename + ".png")

    # def predict(self, inputs, n_samples):
    #     predictive = Predictive(self.model, guide=self.guide, num_samples=n_samples)
    #     svi_samples = {k: v.reshape(n_samples).detach().to(self.device).numpy()
    #                    for k, v in predictive(inputs).items()
    #                    if k != "obs"}
    #     return svi_samples

    def load_posteriors(self, posteriors_names, relative_path=TRAINED_MODELS):
        posteriors_list = []

        for posterior in posteriors_names:
            posterior = self.load(filename=posterior, relative_path=relative_path)
            posteriors_list.append(posterior)

        return posteriors_list



# === MAIN EXECUTIONS ===


def test_conjecture(posteriors, data_loader, dataset_name, n_samples, n_inputs, device, mode, baseclass=None):
    random.seed(0)
    pyro.clear_param_store()

    loss_gradients = expected_loss_gradients(posteriors=posteriors, baseclass=baseclass,
                                                 n_samples=n_samples,
                                                 data_loader=data_loader,
                                                 device=device, mode=mode)

    filename = "expLossGradients_inputs=" + str(len(data_loader.dataset)) \
               + "_samples=" + str(len(posteriors) * n_samples) + "_mode=" + str(mode) + ".pkl"

    # loss_gradients = load_from_pickle(RESULTS + "bnn/"+ filename+".pkl").cpu().detach().numpy()

    plot_heatmap(columns=loss_gradients, path=RESULTS + "bnn/", filename=filename + "_heatmap.png",
                 xlab="pixel idx", ylab="image idx",
                 title=f"Expected loss gradients on {n_samples*len(posteriors)} posterior samples")


def ensemble_evaluation(posteriors, data_loader, n_samples, device):
    total = 0.0
    correct = 0.0
    for images, labels in data_loader:
        total += labels.size(0)
        predictions = []
        for posterior in posteriors:
            post_predictions = posterior.forward(images.to(device), n_samples=n_samples)
            predictions.append(post_predictions.mean(0))
        predictions = torch.stack(predictions)
        ensemble_prediction = predictions.mean(0).argmax(-1)
        correct += (ensemble_prediction == labels.to(device).argmax(-1)).sum().item()
    # print("\npreds[:10]=", ensemble_prediction[:10],"\nlabels[:10]=", labels[:10].to(device).argmax(-1))
    accuracy = 100 * correct / total
    print(f"\n === Ensemble accuracy on {n_samples} sampled models from {len(posteriors)} posteriors = {accuracy:.2f}")


def main(args):

    batch_size=10
    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=args.dataset, batch_size=batch_size, n_inputs=args.inputs)
    pyro.clear_param_store()
    bayesnn = VI_BNN(input_shape=input_shape, device=args.device, dataset_name=args.dataset)

    # bayesnn.infer_parameters(train_loader=train_loader, lr=args.lr, n_epochs=args.epochs)
    # bayesnn.evaluate(data_loader=train_loader, n_samples=args.samples)

    posteriors_names = [
        # bayesnn
        # "hidden_vi_mnist_inputs=60000_lr=0.02_epochs=200", # dropout + log softmax
        # "hidden_vi_mnist_inputs=60000_lr=0.02_epochs=400", # dropout + log softmax
        "hidden_vi_mnist_inputs=60000_lr=0.002_epochs=100",  # log softmax
        "hidden_vi_mnist_inputs=60000_lr=0.02_epochs=80",  # log softmax
        "hidden_vi_mnist_inputs=60000_lr=0.0002_epochs=100",  # log softmax
        "hidden_vi_mnist_inputs=1000_lr=0.002_epochs=200" # log softmax
    ]

    posteriors = bayesnn.load_posteriors(posteriors_names=posteriors_names, relative_path=TRAINED_MODELS)

    # attack_network(dataset_name=args.dataset, n_inputs=args.inputs, device=args.device, n_samples=args.samples)

    # ensemble_evaluation(posteriors=posteriors, data_loader=train_loader, n_samples=args.samples, device=args.device)

    # test_conjecture(posteriors_list=posteriors, data_loader=train_loader, n_samples=args.samples,
    #                 n_inputs=args.inputs, device=args.device, dataset_name=args.dataset)

    n_samples_list=[1]#,5,10,30,50]

    for n_samples in n_samples_list:
        test_conjecture(posteriors=posteriors, data_loader=test_loader, n_samples=n_samples,
                        n_inputs=args.inputs, device=args.device, dataset_name=args.dataset, mode="vi")

    # plot_expectation_over_images(dataset_name=args.dataset, n_inputs=args.inputs, n_samples_list=n_samples_list)
    plot_exp_loss_gradients_norms(dataset_name=args.dataset, n_inputs=args.inputs, n_samples_list=n_samples_list,
                                  n_posteriors=len(posteriors))
    # plot_partial_derivatives(dataset_name=args.dataset, n_inputs=args.inputs, n_samples_list=n_samples_list,
    #                          n_posteriors=len(posteriors))


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.1.0')
    parser = argparse.ArgumentParser(description="VI Bayesian Neural Network using Pyro HiddenLayer module.")

    parser.add_argument("-n", "--inputs", nargs="?", default=10, type=int)
    parser.add_argument("--epochs", nargs='?', default=10, type=int)
    parser.add_argument("--samples", nargs='?', default=3, type=int)
    parser.add_argument("--dataset", nargs='?', default="mnist", type=str)
    parser.add_argument("--lr", nargs='?', default=0.002, type=float)
    parser.add_argument("--device", default='cuda', type=str, help='use "cpu" or "cuda".')

    main(args=parser.parse_args())