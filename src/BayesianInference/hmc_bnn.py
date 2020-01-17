import sys
sys.path.append(".")
from directories import *
import pyro
from BayesianInference.bnn import BNN
from pyro.infer.mcmc import MCMC, HMC
import random
from BayesianInference.pyro_utils import data_loaders
import argparse
import torch
from pyro import poutine
import numpy as np
from utils import execution_time
from BayesianInference.adversarial_attacks import expected_loss_gradients


class HMC_BNN(BNN):
    def __init__(self, dataset_name, input_shape, device, n_chains, warmup, n_samples, n_inputs):
        self.dataset_name = dataset_name
        self.input_size = input_shape[0]*input_shape[1]*input_shape[2]
        super(HMC_BNN, self).__init__(input_size=self.input_size, device=device)
        self.n_inputs = n_inputs
        self.n_chains = n_chains
        self.warmup = warmup
        self.n_samples = n_samples
        self.device = device
        self.filename = "hmc_"+str(self.dataset_name)+"_inputs="+str(n_inputs)+"_warmup="+str(warmup)\
                        +"_samples="+str(n_samples)
        self.filepath = "bnn/"

    def run_chains(self, train_loader, num_steps=4, step_size=0.0855):
        pyro.clear_param_store()
        print("\nHMC inference for ", self.filename)

        hmc_kernel = HMC(self.model, step_size=step_size, num_steps=num_steps)

        start = time.time()
        mcmc = MCMC(kernel=hmc_kernel, num_samples=self.n_samples, warmup_steps=self.warmup, num_chains=self.n_chains)
        for images, labels in train_loader:
            mcmc.run(images.to(self.device), labels.to(self.device))
        # mcmc.summary(prob=0.95)
        # self.plot_posterior(mcmc=mcmc,
        #     filename="hmc_bnn_samples={}_chains={}_steps={}_posterior".format(n_samples, num_chains, num_steps))
        sampled_models = self.sample_models(mcmc=mcmc)
        # self.posterior_samples = sampled_models
        execution_time(start=start, end=time.time())
        return sampled_models

    # def plot_posterior(self, mcmc, filename, path=RESULTS+"bnn/"):
    #     plt.figure(figsize=(10, 7))
    #     samples = mcmc.get_samples(num_samples=10)
    #     for param in samples.keys():
    #         print(samples[param].numpy())
    #         sns.distplot(samples[param].numpy(), label=str(param))
    #     os.makedirs(os.path.dirname(path), exist_ok=True)
    #     plt.savefig(path + filename)

    def sample_models(self, mcmc):
        """
        Sample model weights from the posterior distribution.
        :param mcmc:
        :return:
        """
        n_posterior_samples = self.n_samples*self.n_chains

        # todo: old, bug on saving
        # posterior_samples_dict = mcmc.get_samples(num_samples=n_posterior_samples)
        # # print(posterior_samples_dict["module$$$out.weight"].shape)
        # # print(posterior_samples_dict["module$$$out.weight"][:][:10])
        # posterior_samples_list = [{k: v[i] for k, v in posterior_samples_dict.items()}
        #                           for i in range(n_posterior_samples)]

        # todo qua fa un sampling con ripetizione, non va bene
        posterior_samples_list = []
        for i in range(n_posterior_samples):
            posterior_sample = mcmc.get_samples(num_samples=1)
            # print(posterior_sample["module$$$out.weight"].shape)
            # print(posterior_sample["module$$$out.weight"][:10])
            posterior_samples_list.append(posterior_sample)

        return np.array(posterior_samples_list)

    # def forward(self, inputs, n_samples, mcmc):
    #     """
    #     Computes predictions on the given inputs using `n_samples` sampled models and returns one hot predictions.
    #     :param inputs:
    #     :param mcmc:
    #     :param n_samples:
    #     :return:
    #     """
    #     preds = []
    #     for _ in range(n_samples):
    #         sampled_model = mcmc.get_samples(1)
    #         print(sampled_model)
    #         trace = poutine.trace(poutine.condition(self.model, sampled_model)).get_trace(inputs.to(self.device))
    #         preds.append(trace.nodes['_RETURN']['value'])
    #     pred = torch.stack(preds)
    #     exit()
    #     return pred

    def predict(self, inputs, posterior_samples):
        preds = []
        for posterior_sample in posterior_samples:
            model_weights = posterior_sample.items()
            trace = poutine.trace(poutine.condition(self.model, model_weights)).get_trace(inputs.to(self.device))
            preds.append(trace.nodes['_RETURN']['value'])
        # print(preds[0]) # predictions dal primo sample sui 10 input
        pred = torch.stack(preds)
        # print(pred.argmax(dim=2)[0])
        return pred

    def save(self, posterior_samples, relative_path=RESULTS):
        print("\nSaving posterior samples to:", relative_path+self.filepath+self.filename+".npy")
        np.save(file=relative_path+self.filepath+self.filename+".npy",
                arr=posterior_samples)

    def load(self, relative_path=TRAINED_MODELS):
        print("\nLoading posterior samples from:", relative_path+self.filepath+self.filename+".npy")
        sampled_models = np.load(file=relative_path+self.filepath+self.filename+".npy")
        # self.posterior_samples =  sampled_models
        return sampled_models

    def evaluate(self, test_loader, posterior_samples):
        total = 0.0
        correct = 0.0
        for images, labels in test_loader:
            samples_predictions = self.predict(inputs=images.to(self.device).view(-1, self.input_size),
                                  posterior_samples=posterior_samples)
            # print(samples_predictions.shape)
            # print(samples_predictions.mean(0).shape)
            # print(samples_predictions.mean(0))
            prediction = samples_predictions.mean(0).argmax(-1)
            print("\npredictions[:10] =", prediction[:10])
            print("labels[:10]      =", labels.argmax(-1)[:10])
            total += labels.size(0)
            # print("labels.size(0) =", labels.size(0))
            correct += (prediction == labels.argmax(-1).to(self.device)).sum().item()
        accuracy = 100 * correct / total
        print(f"\n === Accuracy on {len(posterior_samples)} sampled models = {accuracy:.2f}")


def test_conjecture(sampled_models, bayesnn):#dataset_name, n_samples, n_inputs, device):
    random.seed(0)

    # # load bayesian model
    # _, test_loader, data_format, input_shape = data_loaders(dataset_name=dataset_name,
    #                                                         batch_size=n_inputs, n_inputs=n_inputs)
    # pyro.clear_param_store()
    #
    # posteriors_list = []
    # if dataset_name == "mnist":
    #     relative_path=TRAINED_MODELS
    #     trained_models = [
    #         # "hidden_vi_mnist_inputs=60000_lr=0.02_epochs=200", # dropout + log softmax
    #         # "hidden_vi_mnist_inputs=60000_lr=0.02_epochs=400", # dropout + log softmax
    #         "hidden_vi_mnist_inputs=60000_lr=0.002_epochs=100", # log softmax
    #         "hidden_vi_mnist_inputs=60000_lr=0.02_epochs=80", # log softmax
    #         "hidden_vi_mnist_inputs=60000_lr=0.0002_epochs=100", # log softmax
    #     ]
    # else:
    #     return AssertionError("wrong dataset name")
    #
    # # relative_path=RESULTS
    # # trained_models = [
    # #         "hidden_vi_mnist_inputs=10_lr=0.002_epochs=10",
    # # ]

    # compute expected loss gradients
    _, test_loader, _, _ = data_loaders(dataset_name=dataset_name, batch_size=1, n_inputs=n_inputs)
    exp_loss_gradients = expected_loss_gradients(posteriors_list=sampled_models,
                                                 n_samples=n_samples,
                                                 data_loader=test_loader,
                                                 device="cuda", mode="hidden")

    filename = "hidden_vi_" + str(dataset_name) + "_inputs=" + str(n_inputs) + "_samples=" + str(n_samples) \
               + "_posteriors=" + str(len(posteriors_list))
    plot_heatmap(columns=exp_loss_gradients, path=RESULTS + "bnn/", filename=filename + "_heatmap.png",
                 xlab="pixel idx", ylab="image idx",
                 title="Expected loss gradients on {} samples from {} posteriors".format(n_samples, len(posteriors_list)))

def main(args):
    random.seed(234)
    train_loader, _, _, input_shape = \
        data_loaders(dataset_name=args.dataset, batch_size=1, n_inputs=args.inputs)

    bayesnn = HMC_BNN(input_shape=input_shape, device=args.device, dataset_name=args.dataset, n_chains=args.chains,
                      warmup=args.warmup, n_samples=args.samples, n_inputs=args.inputs)
    sampled_models = bayesnn.run_chains(train_loader=train_loader)
    # bayesnn.save(posterior_samples=sampled_models)
    # sampled_models = bayesnn.load(relative_path=RESULTS)

    # bayesnn.evaluate(test_loader=train_loader, posterior_samples=sampled_models)
    # _, test_loader, _, _ = data_loaders(dataset_name=args.dataset, batch_size=128, n_inputs=10000)
    # bayesnn.evaluate(test_loader=test_loader, posterior_samples=sampled_models)

    # test_conjecture(dataset_name=args.dataset, n_samples=args.samples, n_inputs=args.inputs,
    #                 device=args.device)

    exp_loss_gradients = expected_loss_gradients(posteriors_list=[bayesnn],
                                                 n_samples=args.samples*args.chains,
                                                 data_loader=train_loader,
                                                 device="cuda", mode="hmc")


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.1.0')
    parser = argparse.ArgumentParser(description="HMC Bayesian Neural Network")
    parser.add_argument("-n", "--inputs", nargs="?", default=100, type=int)
    parser.add_argument("--warmup", nargs='?', default=10, type=int)
    parser.add_argument("--chains", nargs='?', default=2, type=int)
    parser.add_argument("--samples", nargs='?', default=10, type=int)
    parser.add_argument("--steps", nargs='?', default=4, type=int)
    parser.add_argument("--dataset", nargs='?', default="mnist", type=str)
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "gpu".')

    main(parser.parse_args())