import sys
sys.path.append(".")
import argparse
import random

import pyro
from pyro.infer import TracePredictive, EmpiricalMarginal
from pyro.infer.mcmc import MCMC, HMC
from pyro import poutine
from pyro.infer.mcmc.util import predictive

from utils import execution_time
from utils import save_to_pickle, load_from_pickle
from BayesianInference.bnn import BNN
from BayesianInference.pyro_utils import data_loaders
from BayesianInference.hidden_vi_bnn import test_conjecture
from BayesianInference.plot_utils import *
from BayesianInference.adversarial_attacks import *


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
        self.filepath = "bnn/"
        self.filename = "hmc_" + str(dataset_name) + "_inputs=" + str(n_inputs) + "_warmup=" + str(warmup) \
                    + "_samples=" + str(n_samples) + "_chains=" + str(n_chains)

    def run_chains(self, train_loader, num_steps=1, step_size=0.0855):
        random.seed(0)
        pyro.clear_param_store()
        print("\nHMC BNN:", self.filename)

        hmc_kernel = HMC(self.model, step_size=step_size, num_steps=num_steps)
        mcmc = MCMC(kernel=hmc_kernel, num_samples=self.n_samples, warmup_steps=self.warmup, num_chains=self.n_chains)

        start = time.time()
        for images, labels in train_loader:
            mcmc.run(images.to(self.device), labels.to(self.device))
        execution_time(start=start, end=time.time())

        self.posterior_samples = self.sample_models(mcmc=mcmc)
        return self

    def sample_models(self, mcmc):
        """
        Sample model weights from the posterior distribution.
        :param mcmc:
        :return:
        """
        print("\nSampling weights.")
        # print(mcmc._samples)
        n_posterior_samples = self.n_samples*self.n_chains
        posterior_samples_dict = mcmc.get_samples()#num_samples=n_posterior_samples).to("cuda")
        # posterior_samples_list = [{k: v[i] for k, v in posterior_samples_dict.items()}
        #                           for i in range(n_posterior_samples)]

        # return np.array(posterior_samples_list)
        return posterior_samples_dict

    # def predict(self, inputs, posterior_samples):
    #
    #     preds = []
    #     for posterior in posterior_samples:
    #         posterior_weights = posterior.items() # pesi delle reti salvate
    #         conditioned_model = poutine.condition(self.model, posterior_weights)
    #         guide_trace = poutine.trace(conditioned_model).get_trace(inputs.to(self.device))
    #         # print(guide_trace.nodes['fc1w_prior']['value'][0][:5])
    #         preds.append(guide_trace.nodes['_RETURN']['value'])
    #         # preds.append(guide_trace.nodes['obs']['value'])
    #     preds = torch.stack(preds, dim=0)  # shape = samples x inputs x classes
    #     # print("\npreds.shape =", preds.shape)
    #     # print("\npreds[0] =", preds[0])
    #     return preds

    def forward(self, inputs, n_samples):
        random.seed(0)
        preds = predictive(self.model, self.posterior_samples[:n_samples], inputs, None)["obs"]
        return preds

    def save(self, relative_path=RESULTS):
        # print("\nSaving posterior samples to:", relative_path+self.filepath+self.filename+".npy")
        # os.makedirs(os.path.dirname(relative_path+self.filepath), exist_ok=True)
        # np.save(file=relative_path+self.filepath+self.filename+".npy",
        #         arr=posterior_samples)
        save_to_pickle(data=self.posterior_samples, relative_path=relative_path+self.filepath,
                       filename=self.filename+".pkl")

    def load(self, relative_path, filename):
        # print("\nLoading posterior samples from:", relative_path+self.filepath+filename+".npy")
        # sampled_models = np.load(file=relative_path+self.filepath+filename+".npy")

        sampled_models = load_from_pickle(path=relative_path+self.filepath+filename+".pkl")
        self.posterior_samples = sampled_models
        return self

    def evaluate(self, data_loader, n_samples, device):
        if n_samples > len(self.posterior_samples):
            raise ValueError(f"\nYou can choose max {len(self.posterior_samples)} samples from this posterior.")

        total = 0.0
        correct = 0.0
        for images, labels in data_loader:
            images = images.view(-1, self.input_size).to(device)
            labels = labels.to(device)
            output = self.forward(inputs=images, n_samples=n_samples).to(device)
            output = output.mean(0)

            if DEBUG:
                print("\noutput =", output)
                print("\ncheck prob distributions:", output.sum(dim=-1))

            predictions = output.argmax(dim=-1)
            print("\npredictions[:10] =", predictions[:10])
            print("labels[:10]      =", labels.argmax(-1)[:10])
            total += labels.size(0)
            correct += (predictions == labels.argmax(-1)).sum().item()
        accuracy = 100 * correct / total
        print(f"\n === Accuracy on {n_samples} sampled models = {accuracy:.2f}")


def main(args):
    random.seed(0)
    train_loader, test_loader, _, input_shape = \
        data_loaders(dataset_name=args.dataset, batch_size=args.inputs, n_inputs=args.inputs)
    bayesnn = HMC_BNN(input_shape=input_shape, device=args.device, dataset_name=args.dataset, n_chains=args.chains,
                      warmup=args.warmup, n_samples=args.samples, n_inputs=args.inputs)


    # posterior = bayesnn.run_chains(train_loader=train_loader)
    # posterior.save()

    filename, idx, path = ("hmc_mnist_inputs=10000_warmup=10000_samples=100_chains=1", 0, RESULTS)
    # filename, idx, path = ("hmc_mnist_inputs=60000_warmup=10000_samples=100_chains=1", 1, RESULTS)
    # filename, idx, path = ("hmc_mnist_inputs=10000_warmup=10000_samples=500_chains=1", 2, RESULTS)
    # filename, idx, path = ("hmc_mnist_inputs=60000_warmup=10000_samples=500_chains=1", 3, RESULTS)

    posterior = bayesnn.load(relative_path=path, filename=filename)

    posterior.evaluate(data_loader=train_loader, n_samples=args.samples, device=args.device)

    # === ATTACK ===
    n_samples_list = [1, 10]#500, 1000]
    epsilon_list = [0.1, 0.3, 0.6]
    filename = "catplot_pointwise_softmax_differences_eps=" + str(epsilon_list) \
               + "_inputs=" + str(args.inputs) + "_samples="+str(n_samples_list)+"_mode=vi_model=" + str(idx)

    df = pointwise_attacks(data_loader=test_loader, epsilon_list=epsilon_list, n_samples_list=n_samples_list,
                           posterior=posterior, device=args.device, filename=filename)

    # df = pandas.read_pickle(path=RESULTS+"bnn/"+filename+".pkl")

    catplot_pointwise_softmax_differences(dataframe=df, epsilon_list=epsilon_list,  filename=filename,
                                          n_samples_list=n_samples_list)
    exit()

    # === TEST CONJECTURE ===
    # print(sampled_models)
    # for n_samples in [1, 5, 10]:
    #     test_conjecture(posteriors=sampled_models, data_loader=train_loader, n_samples=n_samples,
    #                     n_inputs=args.inputs, device="cuda", dataset_name=args.dataset, mode="hmc",
    #                     baseclass=bayesnn)

    # # n_samples_list, n_inputs = ([10,30,60,100], 100)
    # n_samples_list, n_inputs = ([5,10,30], 1000)
    # filename="expLossGradients_inputs="+str(n_inputs)+"_samples="+str(n_samples_list)+"_mode=vi_model="+str(idx)
    #
    # # test_loader = data_loaders(dataset_name=args.dataset, batch_size=1, n_inputs=n_inputs, shuffle=True)[1]
    # # exp_loss_gradients = test_conjecture(posterior=posterior, data_loader=test_loader, device=args.device,
    # #                                      n_samples_list=n_samples_list, n_inputs=n_inputs, mode="vi")
    # # save_to_pickle(exp_loss_gradients, relative_path=RESULTS + "bnn/",filename=filename+".pkl")
    #
    # exp_loss_gradients = load_from_pickle(relative_path + "bnn/"+ filename+".pkl")
    #
    # # plot_exp_loss_gradients_norms(exp_loss_gradients=exp_loss_gradients, n_inputs=n_inputs,
    # #                               n_samples_list=n_samples_list, model_idx=idx, filename=filename)
    # plot_gradients_on_images(loss_gradients=exp_loss_gradients, max_n_images=10, n_samples_list=n_samples_list, filename=filename)
    # # catplot_partial_derivatives(filename=filename, n_inputs=n_inputs, n_samples_list=n_samples_list)
    # exit()

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