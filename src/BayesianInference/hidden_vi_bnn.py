import sys
sys.path.append(".")
import argparse

import pyro
from pyro.infer import SVI, Trace_ELBO
import pyro.optim as pyroopt

from BayesianInference.plot_utils import *
from BayesianInference.hidden_bnn import HiddenBNN
from BayesianInference.adversarial_attacks import *
from BayesianInference.loss_gradients import *


DEBUG=False


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

        if DEBUG:
            print("a1_mean", pyro.get_param_store()["a1_mean"])
            print("a2_scale", pyro.get_param_store()["a2_scale"])

        plot_loss_accuracy({'loss':loss_list, 'accuracy':accuracy_list}, path=RESULTS + "bnn/" + filename + ".png")
        return self

    # def predict(self, inputs, n_samples):
    #     predictive = Predictive(self.model, guide=self.guide, num_samples=n_samples)
    #     svi_samples = {k: v.reshape(n_samples).detach().to(self.device).numpy()
    #                    for k, v in predictive(inputs).items()
    #                    if k != "obs"}
    #     return svi_samples

    def load_posterior(self, posterior_name, relative_path=TRAINED_MODELS):
        posterior = self.load(filename=posterior_name, relative_path=relative_path)
        return posterior



# === MAIN EXECUTIONS ===



def test_conjecture(posterior, data_loader, dataset_name, n_samples_list, n_inputs, device, mode, baseclass=None):
    random.seed(0)
    posterior = copy.deepcopy(posterior)
    filename = "expLossGradients_inputs=" + str(len(data_loader.dataset)) + "_mode=" + str(mode) + ".pkl"

    exp_loss_gradients_samples = []
    for n_samples in n_samples_list:
        filename = "expLossGradients_inputs=" + str(len(data_loader.dataset)) \
                   + "_samples=" + str(n_samples) + "_mode=" + str(mode)
        loss_gradients = expected_loss_gradients(posterior=posterior,
                                                     n_samples=n_samples,
                                                     data_loader=data_loader,
                                                     device=device, mode=mode)
        exp_loss_gradients_samples.append(loss_gradients)
    exp_loss_gradients_samples = np.array(exp_loss_gradients_samples)
    save_to_pickle(exp_loss_gradients_samples, relative_path=RESULTS + "bnn/", filename=filename)

    # exp_loss_gradients_samples = load_from_pickle(RESULTS + "bnn/"+ filename)

    plot_exp_loss_gradients_norms(exp_loss_gradients=exp_loss_gradients_samples, n_inputs=n_inputs,
                                  n_samples_list=n_samples_list)



def main(args):

    batch_size=128
    train_loader, _, data_format, input_shape = \
        data_loaders(dataset_name=args.dataset, batch_size=batch_size, n_inputs=args.inputs, shuffle=True)
    bayesnn = VI_BNN(input_shape=input_shape, device=args.device, dataset_name=args.dataset)

    posterior = bayesnn.infer_parameters(train_loader=train_loader, lr=args.lr, n_epochs=args.epochs)
    posterior.evaluate(data_loader=train_loader, n_samples=args.samples)
    # exit()

    ## log softmax dim = -1
    # posterior_name = "hidden_vi_mnist_inputs=60000_lr=0.02_epochs=200", # dropout + log softmax
    # posterior_name = "hidden_vi_mnist_inputs=60000_lr=0.02_epochs=400", # dropout + log softmax
    # posterior_name = "hidden_vi_mnist_inputs=60000_lr=0.002_epochs=100",  # log softmax #0
    # posterior_name = "hidden_vi_mnist_inputs=60000_lr=0.02_epochs=80",  # log softmax #1
    # posterior_name = "hidden_vi_mnist_inputs=60000_lr=0.0002_epochs=100",  # log softmax #2
    # posterior_name = "hidden_vi_mnist_inputs=1000_lr=0.002_epochs=200" # log softmax #3

    ## log softmax dim = 1
    # posterior_name = "hidden_vi_mnist_inputs=10_lr=0.002_epochs=10" # modello al 20% sui primi 10 input #4
    # posterior_name = "hidden_vi_mnist_inputs=10_lr=0.002_epochs=200" # modello al 60% sul train set #5
    # posterior_name = "hidden_vi_mnist_inputs=10_lr=0.02_epochs=100" # modello al 100% sul train set #6

    # posterior_name="hidden_vi_mnist_inputs=10_lr=0.2_epochs=100"
    # posterior = bayesnn.load_posterior(posterior_name=posterior_name, relative_path=RESULTS)

    posterior.evaluate(data_loader=train_loader, n_samples=args.samples)
    # attack_network(dataset_name=args.dataset, n_inputs=args.inputs, device=args.device, n_samples=args.samples)

    test_conjecture(posterior=posterior, data_loader=train_loader, n_samples_list=[5,10,30],
                    n_inputs=args.inputs, device=args.device, dataset_name=args.dataset, mode="vi")

    # ===== AVERAGE OVER IMAGES =====
    n_samples_list = [5, 10, 30]
    n_inputs_list = [10,20,30]
    filename="avgOverImages.pkl"
    average_over_images(posterior, n_inputs_list=n_inputs_list, n_samples_list=n_samples_list, device=args.device,
                        dataset_name=args.dataset, filename=filename)
    plot_avg_over_images_grid(filename=filename)
    distplot_avg_gradients_over_inputs(filename=filename)
    for n_samples in n_samples_list:
        plot_gradients_increasing_inputs(posterior, n_samples, device=args.device)
    exit()
    ##################

    plot_gradients_on_single_image(posterior=posterior, n_samples_list=n_samples_list,
                                   device=args.device, data_loader=train_loader)



    for n_samples in n_samples_list:
        test_conjecture(posterior=posterior, data_loader=train_loader, n_samples=n_samples,
                        n_inputs=args.inputs, device=args.device, dataset_name=args.dataset, mode="vi")
    exit()
    # plot_expectation_over_images(dataset_name=args.dataset, n_inputs=args.inputs, n_samples_list=n_samples_list)

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