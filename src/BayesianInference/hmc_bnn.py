import sys

sys.path.append(".")
from directories import *
import pyro
from BayesianInference.bnn import BNN
# from pyro.infer import MCMC, HMC
from pyro.infer.mcmc import MCMC, HMC
import random
from BayesianInference.pyro_utils import data_loaders
import argparse
import torch
from pyro import poutine
import dill
from utils import save_to_pickle, load_from_pickle
from pyro.infer import TracePosterior, TracePredictive


class HMC_BNN(BNN):
    def __init__(self, input_shape, device):
        self.input_size = input_shape[0]*input_shape[1]*input_shape[2]
        super(HMC_BNN, self).__init__(input_size=self.input_size, device=device)

    def infer_parameters(self, train_loader, n_samples, num_chains, warmup, num_steps, step_size=0.0855):
        print("\nHMC inference.")
        hmc_kernel = HMC(self.model, step_size=step_size, num_steps=num_steps)
        mcmc = MCMC(kernel=hmc_kernel, num_samples=n_samples, warmup_steps=warmup, num_chains=num_chains)
        for images, labels in train_loader:
            mcmc.run(images.to(self.device), labels.to(self.device))
        # mcmc.summary(prob=0.95)
        # self.plot_posterior(mcmc=mcmc,
        #     filename="hmc_bnn_samples={}_chains={}_steps={}_posterior".format(n_samples, num_chains, num_steps))
        sampled_models = self.sample_models(n_samples=n_samples, mcmc=mcmc)
        return sampled_models

    # def plot_posterior(self, mcmc, filename, path=RESULTS+"bnn/"):
    #     plt.figure(figsize=(10, 7))
    #     samples = mcmc.get_samples(num_samples=10)
    #     for param in samples.keys():
    #         print(samples[param].numpy())
    #         sns.distplot(samples[param].numpy(), label=str(param))
    #     os.makedirs(os.path.dirname(path), exist_ok=True)
    #     plt.savefig(path + filename)

    def sample_models(self, n_samples, mcmc):
        """
        Sample model weights from the posterior distribution.
        :param n_samples:
        :param mcmc:
        :return:
        """
        posterior_samples_dict = mcmc.get_samples(num_samples=n_samples)
        posterior_samples_list = [{k: v[i] for k, v in posterior_samples_dict.items()} for i in range(n_samples)]
        return posterior_samples_list

    def forward(self, inputs, n_samples, mcmc):
        """
        Computes predictions on the given inputs using `n_samples` sampled models and returns one hot predictions.
        :param inputs:
        :param mcmc:
        :param n_samples:
        :return:
        """
        preds = []
        for _ in range(n_samples):
            sampled_model = mcmc.get_samples(1)
            trace = poutine.trace(poutine.condition(self.model, sampled_model)).get_trace(inputs.to(self.device))
            preds.append(trace.nodes['_RETURN']['value'])
        pred = torch.stack(preds)
        return pred

    def predict(self, inputs, posterior_samples, n_samples):
        preds = []
        for i in range(n_samples):
            sampled_model = posterior_samples[i]
            trace = poutine.trace(poutine.condition(self.model, sampled_model)).get_trace(inputs.to(self.device))
            preds.append(trace.nodes['_RETURN']['value'])
        pred = torch.stack(preds)
        return pred

    def save(self, posterior_samples, filename, relative_path=RESULTS):
        save_to_pickle(data=posterior_samples, filename=filename+".pkl", relative_path=relative_path+"bnn/")

    def load(self, filename, relative_path=TRAINED_MODELS):
        return load_from_pickle(path=relative_path+"bnn/"+filename+".pkl")

    def evaluate(self, test_loader, posterior_samples, n_samples):
        total = 0.0
        correct = 0.0
        for images, labels in test_loader:
            pred = self.predict(inputs=images.to(self.device).view(-1, self.input_size),
                                posterior_samples=posterior_samples, n_samples=n_samples).mean(0).argmax(-1)
            print("\npredictions =", pred)
            print("labels =", labels.argmax(-1))
            total += labels.size(0)
            correct += (pred == labels.argmax(-1).to(self.device)).sum().item()
        accuracy = 100 * correct / total
        print(f"\n === Accuracy on {n_samples} sampled models = {accuracy:.2f}")


def main(args):
    random.seed(234)
    batch_size = args.inputs
    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=args.dataset_name, batch_size=batch_size, n_inputs=args.inputs)

    pyro.clear_param_store()
    bayesnn = HMC_BNN(input_shape=input_shape, device=args.device)
    # sampled_models = bayesnn.infer_parameters(n_samples=args.samples, train_loader=train_loader,
    #                                 num_chains=args.chains, warmup=args.warmup, num_steps=args.steps)
    filename = "hmc_" + str(args.dataset_name) + "_inputs=" + str(args.inputs) + "_chains=" + str(args.chains) + \
               "_warmup=" + str(args.warmup) + "_steps=" + str(args.steps)
    # bayesnn.save(posterior_samples=sampled_models, filename=filename)
    sampled_models = bayesnn.load(filename=filename, relative_path=RESULTS)

    bayesnn.evaluate(test_loader=test_loader, posterior_samples=sampled_models, n_samples=args.samples)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.1.0')
    parser = argparse.ArgumentParser(description="HMC Bayesian Neural Network")
    parser.add_argument("-n", "--inputs", nargs="?", default=100, type=int)
    parser.add_argument("--warmup", nargs='?', default=5, type=int)
    parser.add_argument("--chains", nargs='?', default=1, type=int)
    parser.add_argument("--samples", nargs='?', default=3, type=int)
    parser.add_argument("--steps", nargs='?', default=10, type=int)
    parser.add_argument("--dataset_name", nargs='?', default="mnist", type=str)
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "gpu".')

    main(parser.parse_args())