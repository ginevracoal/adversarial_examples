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
from utils import save_to_pickle, load_from_pickle
import numpy as np


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
        self.filepath = "bnn/"#+self.filename+"/"

    def run_chains(self, train_loader, num_steps=4, step_size=0.0855):#, n_samples, num_chains, warmup):
        print("\nHMC inference for ", self.filename)

        hmc_kernel = HMC(self.model, step_size=step_size, num_steps=num_steps)
        mcmc = MCMC(kernel=hmc_kernel, num_samples=self.n_samples, warmup_steps=self.warmup, num_chains=self.n_chains)
        for images, labels in train_loader:
            mcmc.run(images.to(self.device), labels.to(self.device))
        # mcmc.summary(prob=0.95)
        # self.plot_posterior(mcmc=mcmc,
        #     filename="hmc_bnn_samples={}_chains={}_steps={}_posterior".format(n_samples, num_chains, num_steps))
        sampled_models = self.sample_models(mcmc=mcmc)
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
        # posterior_samples_list = [{k: v[i] for k, v in posterior_samples_dict.items()}
        #                           for i in range(n_posterior_samples)]
        # print(posterior_samples_dict["module$$$out.weight"])

        posterior_samples_list = []
        for i in range(n_posterior_samples):
            posterior_sample = mcmc.get_samples(num_samples=1)
            # save_to_pickle(data=posterior_sample, filename=self.filename + "_" + str(i) + ".pkl",
            #                relative_path=RESULTS + self.filepath)
            posterior_samples_list.append(posterior_sample)

        return np.array(posterior_samples_list)

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
        # posterior_samples = [{k: v[i] for k, v in posterior_samples.items()} for i in range(n_samples)]
        preds = []
        for i in range(n_samples):
            sampled_model = posterior_samples[i].items()
            trace = poutine.trace(poutine.condition(self.model, sampled_model)).get_trace(inputs.to(self.device))
            preds.append(trace.nodes['_RETURN']['value'])
        pred = torch.stack(preds)
        return pred

    def save(self, posterior_samples, relative_path=RESULTS):
        # print(posterior_samples[0].keys())
        # print(len(posterior_samples))
        print("\nSaving posterior samples to:", relative_path+self.filepath+self.filename+".npy")
        np.save(file=relative_path+self.filepath+self.filename+".npy",
                arr=posterior_samples)
        # # keys = ['module$$$fc1.weight', 'module$$$fc1.bias', 'module$$$out.weight', 'module$$$out.bias']
        # for i, sampled_model in enumerate(posterior_samples):
        #     # print(sys.getsizeof(sampled_model))
        #     # print([sampled_model[key].shape for key in keys])
        #     save_to_pickle(data=sampled_model, filename=self.filename+"_"+str(i)+".pkl",
        #                    relative_path=relative_path+self.filepath)

    def load(self, relative_path=TRAINED_MODELS, n_samples=None):
        print("\nLoading posterior samples from:", relative_path+self.filepath+self.filename+".npy")
        sampled_models = np.load(file=relative_path+self.filepath+self.filename+".npy")
        # sampled_models = []
        # if n_samples is None:
        #     n_samples = self.n_samples*self.n_chains
        # for i in range(n_samples):
        #     sampled_models.append(load_from_pickle(path=relative_path+self.filepath+self.filename+"_"+str(i)+".pkl"))
        return sampled_models

    def evaluate(self, test_loader, posterior_samples, n_samples):
        total = 0.0
        correct = 0.0
        for images, labels in test_loader:
            pred = self.predict(inputs=images.to(self.device).view(-1, self.input_size),
                                posterior_samples=posterior_samples, n_samples=n_samples).mean(0).argmax(-1)
            print("\npredictions[:10] =", pred[:10])
            print("labels[:10]      =", labels.argmax(-1)[:10])
            total += labels.size(0)
            correct += (pred == labels.argmax(-1).to(self.device)).sum().item()
        accuracy = 100 * correct / total
        print(f"\n === Accuracy on {len(posterior_samples)} sampled models = {accuracy:.2f}")


def main(args):
    random.seed(234)
    batch_size = args.inputs
    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=args.dataset, batch_size=batch_size, n_inputs=args.inputs)

    pyro.clear_param_store()
    bayesnn = HMC_BNN(input_shape=input_shape, device=args.device, dataset_name=args.dataset, n_chains=args.chains,
                      warmup=args.warmup, n_samples=args.samples, n_inputs=args.inputs)
    sampled_models = bayesnn.run_chains(train_loader=train_loader)
    bayesnn.save(posterior_samples=sampled_models)
    sampled_models = bayesnn.load(relative_path=RESULTS)
    bayesnn.evaluate(test_loader=test_loader, posterior_samples=sampled_models, n_samples=args.samples)


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