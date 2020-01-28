import sys


sys.path.append(".")
import argparse

import pyro
from pyro.infer import SVI, Trace_ELBO
import pyro.optim as pyroopt

from BayesianInference.plots.plot_utils import *
from BayesianInference.hidden_bnn import HiddenBNN
from BayesianInference.adversarial_attacks import *
from BayesianInference.loss_gradients import *


DEBUG=False


class VI_BNN(HiddenBNN):
    def __init__(self, dataset_name, input_shape, device, activation="softmax"):
        self.input_size = input_shape[0]*input_shape[1]*input_shape[2]
        self.activation = activation
        self.loss = "crossentropy" if activation == "softmax" else "nllloss"
        self.hidden_size = 512 if dataset_name == "mnist" else 1024
        self.n_classes = 10
        super(VI_BNN, self).__init__(input_size=self.input_size, device=device, activation=self.activation,
                                     hidden_size=self.hidden_size, dataset_name="mnist")
        self.dataset_name = dataset_name

    def get_filename(self, n_inputs, lr, n_epochs):
        return "hidden_vi_" + str(self.dataset_name) + "_inputs=" + str(n_inputs) + \
                "_lr=" + str(lr) + "_epochs=" + str(n_epochs)

    def infer_parameters(self, train_loader, lr, n_epochs, seed=0):
        random.seed(seed)
        filename = self.get_filename(n_inputs=len(train_loader.dataset), lr=lr, n_epochs=n_epochs)
        print("\nSVI BNN:", filename)
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
                avg_pred = self.forward(images.to(self.device), n_samples=3).mean(0)
                pred = avg_pred.argmax(-1)
                correct += (pred == labels.argmax(-1).to(self.device)).sum().item()
                accuracy = 100 * correct / total

                if DEBUG:
                    print(images.shape)
                    print("\nimages.shape = ", images.view(-1, self.input_size).shape)
                    print("\ncheck prob dist:", avg_pred.sum(1))
                    exit()

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

    def load_posterior(self, posterior_name, activation="softmax", relative_path=TRAINED_MODELS):
        posterior = self.load(filename=posterior_name, relative_path=relative_path)
        posterior.activation = activation
        return posterior


# === MAIN EXECUTIONS ===

def produce_attacks(dataset_name, n_inputs, activation, lr, device):

    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=dataset_name, batch_size=32, n_inputs=n_inputs, shuffle=True)

    relative_path = TRAINED_MODELS

    models_list = [{"idx": 0, "filename": "hidden_vi_mnist_inputs=10000_lr=0.0002_epochs=100", "activation": "softmax",
                    "dataset": "mnist"},
                   {"idx": 1, "filename": "hidden_vi_mnist_inputs=60000_lr=0.0002_epochs=100", "activation": "softmax",
                    "dataset": "mnist"},
                   {"idx": 2, "filename": "hidden_vi_mnist_inputs=60000_lr=0.0002_epochs=11", "activation": "softmax",
                    "dataset": "mnist"}]

    plot_accuracy = []
    plot_robustness = []
    plot_eps = []

    # === load models ===
    for model_dict in models_list:
        idx = model_dict["idx"]

        bayesnn = VI_BNN(input_shape=input_shape, device=device, dataset_name=model_dict["dataset"],
                         activation=model_dict["activation"])
        posterior = bayesnn.load_posterior(posterior_name=model_dict["filename"], relative_path=relative_path,
                                               activation=model_dict["activation"])

    # === train models ===
    # idx = None
    # for epochs in [100]:
    #     bayesnn = VI_BNN(input_shape=input_shape, device=device, dataset_name=dataset_name,
    #                      activation=activation)
    #     # train_loader = slice_data_loader(data_loader=train_loader, slice_size=30)
    #     posterior = bayesnn.infer_parameters(train_loader=train_loader, lr=lr, n_epochs=epochs)

    #######################
        for epsilon in [0.1, 0.3, 0.6]:
            for n_samples in [50, 100]:

                posterior.evaluate(data_loader=test_loader, n_samples=n_samples)

                filename = str(dataset_name)+"_inputs="+str(n_inputs)+"_epsilon=" + str(epsilon)\
                           + "_samples=" + str(n_samples)

                if idx != None:
                    filename = filename+"_model="+str(idx)+"_attack.pkl"
                else:
                    filename = filename+"_attack.pkl"

                attack_dict = bayesian_attack(model=posterior, data_loader=test_loader, epsilon=epsilon,
                                              device=device,
                                              n_attack_samples=n_samples, n_pred_samples=n_samples)
                save_to_pickle(relative_path=RESULTS + "bnn/", filename=filename, data=attack_dict)

                # attack_dict = load_from_pickle(path=RESULTS + "nn/"+filename)
                plot_robustness.append(attack_dict["softmax_robustness"])
                plot_accuracy.append(attack_dict["original_accuracy"])
                plot_eps.append(epsilon)

    # model_type = np.repeat("bnn", len(plot_accuracy))
    # scatterplot_accuracy_robustness(accuracy=plot_accuracy, robustness=plot_robustness, model_type=model_type,
    #                                 epsilon=plot_eps)



def test_conjecture(posterior, data_loader, idx, n_samples_list, n_inputs, device, baseclass=None):
    random.seed(0)
    posterior = copy.deepcopy(posterior)

    filename="distPlot_expLossGradients_inputs="+str(n_inputs)+"_samples="+str(n_samples_list)+"_model="+str(idx)

    # data_loader_slice = slice_data_loader(data_loader=data_loader, slice_size=n_inputs)
    # exp_loss_gradients_samples = []
    # for n_samples in n_samples_list:
    #     posterior.evaluate(data_loader=data_loader_slice, n_samples=n_samples)
    #     loss_gradients = expected_loss_gradients(posterior=posterior,
    #                                                  n_samples=n_samples,
    #                                                  data_loader=data_loader_slice,
    #                                                  device=device, mode=mode)
    #     exp_loss_gradients_samples.append(loss_gradients)
    #
    # exp_loss_gradients_samples = np.array(exp_loss_gradients_samples)
    #
    # return exp_loss_gradients_samples

    # exp_loss_gradients = test_conjecture(posterior=posterior, data_loader=data_loader, device=device,
    #                                      n_samples_list=n_samples_list, n_inputs=n_inputs, idx=idx)
    # save_to_pickle(exp_loss_gradients, relative_path=RESULTS + "bnn/",filename=filename+".pkl")
    #
    # exp_loss_gradients = load_from_pickle(RESULTS + "bnn/"+ filename+".pkl")

    plot_exp_loss_gradients_norms(exp_loss_gradients=exp_loss_gradients, n_inputs=n_inputs,
                                  n_samples_list=n_samples_list, model_idx=idx, filename=filename)
    plot_gradients_on_images(loss_gradients=exp_loss_gradients, max_n_images=10, n_samples_list=n_samples_list, filename=filename)
    # catplot_partial_derivatives(filename=filename, n_inputs=n_inputs, n_samples_list=n_samples_list)
    # exit()


def avg_over_images(n_inputs_list, n_samples_list, posterior, idx, data_loader, device):

    filename = "avgOverImages_inputs="+str(n_inputs_list)+"_samples="+str(n_samples_list)+"_model="+str(idx)
    average_over_images(posterior, n_inputs_list=n_inputs_list, n_samples_list=n_samples_list, device=args.device,
                        data_loader=test_loader, filename=filename)
    plot_avg_over_images_grid(filename=filename)
    distplot_avg_gradients_over_inputs(filename=filename)
    plot_gradients_increasing_inputs(posterior=posterior, n_samples_list=n_samples_list, n_inputs_list=n_inputs_list,
                                     data_loader=data_loader, device=device)

def main(args):

    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=args.dataset, batch_size=32, n_inputs=args.inputs, shuffle=True)
    bayesnn = VI_BNN(input_shape=input_shape, device=args.device, dataset_name=args.dataset, activation=args.activation)

    # train_loader = slice_data_loader(data_loader=train_loader, slice_size=100)
    # test_loader = slice_data_loader(data_loader=test_loader, slice_size=1000)

    relative_path=TRAINED_MODELS
    models_list = [{"idx":0,"filename":"hidden_vi_mnist_inputs=10000_lr=0.0002_epochs=100","activation":"softmax",
                    "dataset":"mnist"},
                   {"idx":1,"filename":"hidden_vi_mnist_inputs=60000_lr=0.0002_epochs=100","activation":"softmax",
                    "dataset":"mnist"},
                   {"idx":2,"filename":"hidden_vi_mnist_inputs=60000_lr=0.0002_epochs=11","activation":"softmax",
                    "dataset":"mnist"}]

    # posterior = bayesnn.infer_parameters(train_loader=train_loader, lr=args.lr, n_epochs=args.epochs)

    # idx = 0
    # posterior = bayesnn.load_posterior(posterior_name=models_list[idx]["filename"],
    #                                    relative_path=relative_path, activation=models_list[idx]["activation"])

    # posterior.evaluate(data_loader=train_loader, n_samples=args.samples)
    #
    produce_attacks(dataset_name=args.dataset, lr=args.lr, device=args.device, activation=args.activation,
                    n_inputs=args.inputs)

    # test_conjecture(posterior=posterior, data_loader=test_loader, idx=idx, device=args.device,
    #                 n_samples_list=[5,10,30], n_inputs=args.inputs)

    avg_over_images(n_inputs_list=[1,10,100], n_samples_list=[10,100,1000], posterior=posterior, idx=idx,
                    data_loader=test_loader, device=args.device)



if __name__ == "__main__":
    assert pyro.__version__.startswith('1.1.0')
    parser = argparse.ArgumentParser(description="VI Bayesian Neural Network using Pyro HiddenLayer module.")

    parser.add_argument("-n", "--inputs", nargs="?", default=10, type=int)
    parser.add_argument("--epochs", nargs='?', default=10, type=int)
    parser.add_argument("--samples", nargs='?', default=3, type=int)
    parser.add_argument("--dataset", nargs='?', default="mnist", type=str)
    parser.add_argument("--activation", nargs='?', default="softmax", type=str)
    parser.add_argument("--lr", nargs='?', default=0.002, type=float)
    parser.add_argument("--device", default='cuda', type=str, help='use "cpu" or "cuda".')

    main(args=parser.parse_args())