import sys
sys.path.append(".")
import pyro
from BayesianInference.bnn import BNN
from pyro.infer.mcmc import MCMC, HMC
import random
from BayesianInference.pyro_utils import data_loaders
import argparse
from pyro import poutine
from BayesianInference.plot_utils import *
from utils import execution_time
from BayesianInference.hidden_vi_bnn import test_conjecture


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
                        +"_samples="+str(n_samples)+"_chains="+str(n_chains)
        self.filepath = "bnn/"

    def run_chains(self, train_loader, num_steps=1, step_size=0.0855):
        random.seed(0)

        pyro.clear_param_store()
        print("\nHMC inference for ", self.filename)

        hmc_kernel = HMC(self.model, step_size=step_size, num_steps=num_steps)

        start = time.time()
        mcmc = MCMC(kernel=hmc_kernel, num_samples=self.n_samples, warmup_steps=self.warmup, num_chains=self.n_chains)
        for images, labels in train_loader:
            mcmc.run(images.to(self.device), labels.to(self.device))
        sampled_models = self.sample_models(mcmc=mcmc)
        execution_time(start=start, end=time.time())

        return sampled_models

    def sample_models(self, mcmc):
        """
        Sample model weights from the posterior distribution.
        :param mcmc:
        :return:
        """
        n_posterior_samples = self.n_samples*self.n_chains
        posterior_samples_dict = mcmc.get_samples(num_samples=n_posterior_samples)
        posterior_samples_list = [{k: v[i] for k, v in posterior_samples_dict.items()}
                                  for i in range(n_posterior_samples)]

        return np.array(posterior_samples_list)

        # posterior_samples_dict = mcmc.get_samples(num_samples=500)
        # # posterior_dist = {}
        # # for k in posterior_samples_dict.keys():
        # #     posterior_dist.update({k: posterior_samples_dict[str(k)].mean(0)})
        # posterior_dist = {k: posterior_samples_dict[str(k)].mean(0) for k in posterior_samples_dict.keys()}
        # # posterior_dist = np.mean(posterior_samples_list, axis=0)
        # # print(posterior_dist)
        # return posterior_dist # todo test on a single avg posterior

    def predict(self, inputs, posterior_samples):
        preds = []
        for posterior in posterior_samples:
            posterior_weights = posterior.items()
            guide_trace = poutine.trace(poutine.condition(self.model, posterior_weights)).get_trace(inputs.to(self.device))
            preds.append(guide_trace.nodes['_RETURN']['value'].exp())

        preds = torch.stack(preds, dim=0)  # shape = samples x inputs x classes
        return preds

    def save(self, posterior_samples, relative_path=RESULTS):
        print("\nSaving posterior samples to:", relative_path+self.filepath+self.filename+".npy")
        os.makedirs(os.path.dirname(relative_path+self.filepath), exist_ok=True)
        np.save(file=relative_path+self.filepath+self.filename+".npy",
                arr=posterior_samples)

    def load(self, relative_path=TRAINED_MODELS):
        print("\nLoading posterior samples from:", relative_path+self.filepath+self.filename+".npy")
        sampled_models = np.load(file=relative_path+self.filepath+self.filename+".npy")
        return sampled_models

    def evaluate(self, test_loader, posterior_samples, device):
        total = 0.0
        correct = 0.0
        for images, labels in test_loader:
            images = images.to(device).view(-1, self.input_size)
            labels = labels.to(device)
            # print(images[0][350:400])
            output=self.predict(inputs=images, posterior_samples=posterior_samples).to(device).mean(0)
            # print("\ncheck prob distributions:", output.sum(dim=1))
            predictions = output.argmax(dim=-1)
            print("\npredictions[:10] =", predictions[:10])
            print("labels[:10]      =", labels.argmax(-1)[:10])
            total += labels.size(0)
            correct += (predictions == labels.argmax(-1)).sum().item()
        accuracy = 100 * correct / total
        print(f"\n === Accuracy on {len(posterior_samples)} sampled models = {accuracy:.2f}")


def main(args):
    random.seed(0)
    train_loader, _, _, input_shape = \
        data_loaders(dataset_name=args.dataset, batch_size=100, n_inputs=args.inputs)

    bayesnn = HMC_BNN(input_shape=input_shape, device=args.device, dataset_name=args.dataset, n_chains=args.chains,
                      warmup=args.warmup, n_samples=args.samples, n_inputs=args.inputs)

    sampled_models = bayesnn.run_chains(train_loader=train_loader)
    bayesnn.save(posterior_samples=sampled_models)
    sampled_models = bayesnn.load(relative_path=RESULTS)

    bayesnn.evaluate(test_loader=train_loader, posterior_samples=sampled_models, device="cuda")
    exit()
    # print(sampled_models)
    for n_samples in [1, 5, 10]:
        test_conjecture(posteriors=sampled_models, data_loader=train_loader, n_samples=n_samples,
                        n_inputs=args.inputs, device="cuda", dataset_name=args.dataset, mode="hmc",
                        baseclass=bayesnn)


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