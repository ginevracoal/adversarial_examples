import sys
sys.path.append(".")
from directories import *
import pandas as pd
import matplotlib
from BayesianInference.adversarial_attacks import *
from BayesianInference.loss_gradients import load_loss_gradients

DATA_PATH="../data/random_attacks/"
RESULTS=RESULTS+"random_attacks/"


def plot_random_attack(random_attack_dict, filename):
    # softmax_rob = [1-diff.item() for diff in random_attack_dict["softmax_diff"]]
    dict = {"Attack": random_attack_dict["attack_type"],
                          "samples":random_attack_dict["samples"],
                          "softmax_rob":random_attack_dict["softmax_diff"]}
    df = pd.DataFrame(data=dict)
    print(df.head())
    print(df.describe())

    sns.set()
    plt.subplots(figsize=(8, 8), dpi=100)
    sns.set_palette("YlGnBu_d",2)
    g = sns.lineplot(x="samples", y="softmax_rob", hue="Attack", style="Attack", data=df, ci="sd")
    g.set(xticks=df.samples.values)

    plt.xlabel("Samples involved in expectations ($w_i \sim p(w|D)$)", fontsize=10)
    plt.ylabel('Softmax difference ($l_\infty$)', fontsize=10)
    plt.legend(loc='lower right', title="Attack", labels=["random","FGSM"])

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    plt.savefig(RESULTS+filename)


def attack_model(model, loss_gradients, test_loader, n_samples_list, device, n_pred_samples=None, eps=0.25):

    samples = []
    softmax_diff = []
    attack_type = []

    im_count = 0
    for n_samples in n_samples_list:
        if n_pred_samples is None:
            n_pred_samples = n_samples
        print("\nn_samples = ", n_samples)
        for images, labels in test_loader:
            for idx in tqdm(range(len(images))):
                image = images[idx]
                label = labels[idx]

                input_shape = image.size(0) * image.size(1) * image.size(2)
                label = label.to(device).argmax(-1).view(-1)
                image = image.to(device).view(-1, input_shape)

                # == random attack ==
                attack_dict = random_bayesian_attack(model=model, image=image,
                                                     label=label, epsilon=eps, device=device,
                                                     n_pred_samples=n_pred_samples)


                original_output = attack_dict["original_output"].clone().detach().to(device)
                adversarial_output = attack_dict["adversarial_output"].clone().detach().to(device)
                difference = (original_output - adversarial_output).abs().max(dim=-1)[0].item()
                softmax_diff.append(difference)
                samples.append(n_samples)
                attack_type.append("random")

                # == bayesian fgsm attack ==
                loss_gradient = torch.tensor(loss_gradients[str(n_samples)][im_count])
                attack_dict = fgsm_bayesian_attack(model=model, image=image,
                                                   label=label, epsilon=eps, device=device,
                                                   n_pred_samples=n_samples,
                                                   n_attack_samples=n_pred_samples,
                                                   loss_gradient=loss_gradient)
                original_output = attack_dict["original_output"].clone().detach().to(device)
                adversarial_output = attack_dict["adversarial_output"].clone().detach().to(device)
                difference = (original_output - adversarial_output).abs().max(dim=-1)[0].item()
                softmax_diff.append(difference)
                samples.append(n_samples)
                attack_type.append("fgsm")

                # == bayesian pgd attack ==
                attack_dict = pgd_bayesian_attack(model=model, image=image,
                                                   label=label, epsilon=eps, device=device,
                                                   n_pred_samples=n_samples,
                                                   n_attack_samples=n_pred_samples)
                original_output = attack_dict["original_output"].clone().detach().to(device)
                adversarial_output = attack_dict["adversarial_output"].clone().detach().to(device)
                difference = (original_output - adversarial_output).abs().max(dim=-1)[0].item()
                softmax_diff.append(difference)
                samples.append(n_samples)
                attack_type.append("pgd")

    random_attack_dict = {"samples":samples, "softmax_diff":softmax_diff, "attack_type":attack_type}
    return random_attack_dict


def get_filename(dataset, test_images, pred_samples, n_samples_list):
    return str(dataset)+"_images="+str(test_images)+"_attackSamp="+str(n_samples_list)\
               +"_predSamp="+str(pred_samples)+"_random_attack"


def final_plot(test_images, pred_samples=200, path=DATA_PATH):
    matplotlib.rc('font', **{'weight': 'bold', 'size': 8})
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
    sns.set()
    sns.set_palette("YlGnBu_d",3)

    n_inputs, n_samples_list = 1000, [1,10,50,100]

    plot_attack = []
    plot_samples = []
    plot_diff = []

    for row_idx, pred_samples in enumerate([None, pred_samples]):
        for col_idx, dataset in enumerate( ["mnist","fashion_mnist"]):
            for file in os.listdir(path+str(dataset)+"_predSamp="+str(pred_samples)):
                if file.endswith(".pkl"):
                    random_attack_dict=load_from_pickle(path+str(dataset)+"_predSamp="+str(pred_samples)+"/"+file)
                    plot_attack.extend(random_attack_dict["attack_type"])
                    plot_samples.extend(random_attack_dict["samples"])
                    plot_diff.extend(random_attack_dict["softmax_diff"])

            # softmax_rob = [1-diff for diff in random_attack_dict["softmax_diff"]]
            df = pd.DataFrame(data={"Attack": plot_attack, "samples":plot_samples, "softmax_rob":plot_diff})
            print(df.head())

            g = sns.lineplot(x="samples", y="softmax_rob", hue="Attack", style="Attack", data=df, ci="sd",
                             ax=ax[row_idx, col_idx])
            g.set(xticks=df.samples.values)

            ax[row_idx, col_idx].set_xlabel("")
            ax[row_idx, col_idx].set_ylabel("")
            ax[row_idx, 0].set_ylabel("MNIST", fontsize=9, rotation=270,labelpad=10)
            ax[row_idx, 0].yaxis.set_label_position("right")
            ax[row_idx, 1].set_ylabel(f"Fashion MNIST", fontsize=9, rotation=270,labelpad=10)
            ax[row_idx, 1].yaxis.set_label_position("right")

            if row_idx == 0 and col_idx == 1:
                ax[row_idx, col_idx].legend(loc='upper right', title="Attack", labels=["random", "FGSM","PGD"])
            else:
                ax[row_idx, col_idx].legend().remove()

            ax[0, 0].set_title(f"Predictive samples = attack samples ", fontsize=10)
            ax[1, 0].set_title(f"Predictive samples = {pred_samples}", fontsize=10)
            ax[0, 1].set_title(f"Predictive samples = attack samples ", fontsize=10)
            ax[1, 1].set_title(f"Predictive samples = {pred_samples}", fontsize=10)

    fig.text(0.04, 0.5, r'Softmax difference ($l_\infty$)', fontsize=11, va='center', rotation='vertical')
    fig.text(0.5, 0.04, r"Samples involved in the attacks ($w_i \sim p(w|D)$)", ha='center', fontsize=11)


    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    plt.savefig(RESULTS+"attacks_comparison_VI_testImages="+str(test_images)+".png")

def main(args):

    test_images = 10
    pred_samples = None # None or int

    # == final plot ==
    final_plot(test_images=test_images)
    exit()

    # == produce attacks ==
    n_samples_list = [1,10,50,100]

    if args.dataset == "mnist":
        model = hidden_vi_models[2]
    elif args.dataset == "fashion_mnist":
        model = hidden_vi_models[5]
    else:
        raise AssertionError("wrong dataset name")

    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=args.dataset, batch_size=128, n_inputs=model["n_inputs"], shuffle=True)
    test = slice_data_loader(data_loader=test_loader, slice_size=test_images)

    bayesnn = VI_BNN(input_shape=input_shape, device=args.device, architecture=model["architecture"],
                     activation=model["activation"])
    posterior = bayesnn.load_posterior(posterior_name=model["filename"], relative_path=TRAINED_MODELS,
                                       activation=model["activation"])

    loss_gradients = {}
    for n_samples in n_samples_list:
        gradients = load_loss_gradients(dataset_name=args.dataset, n_inputs=model["n_inputs"], n_samples=n_samples,
                                        model_idx=model["idx"])
        loss_gradients.update({str(n_samples):gradients})
    random_attack_dict = attack_model(model=posterior, loss_gradients=loss_gradients, n_pred_samples=pred_samples,
                                      n_samples_list=n_samples_list, device=args.device, test_loader=test)
    filename = get_filename(args.dataset, test_images, pred_samples, n_samples_list)
    save_to_pickle(data=random_attack_dict, relative_path=RESULTS+"testImages="+str(test_images)+"/",
                   filename=filename+".pkl")

    # random_attack_dict=load_from_pickle(RESULTS+filename+".pkl")
    # random_attack_dict=load_from_pickle(DATA_PATH+filename+".pkl")
    plot_random_attack(random_attack_dict=random_attack_dict, filename=filename+".png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cuda', type=str, help='use "cpu" or "cuda".')
    parser.add_argument("--dataset", default='mnist', type=str, help='use "mnist" or "fashion_mnist".')
    main(args=parser.parse_args())

