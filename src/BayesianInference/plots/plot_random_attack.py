import sys
sys.path.append(".")
from directories import *
import pandas as pd
import matplotlib
from BayesianInference.adversarial_attacks import *
from BayesianInference.loss_gradients import load_loss_gradients

DATA_PATH="../data/random_attacks/"
RESULTS=RESULTS+"random_attacks/"

DEBUG=False


def plot_random_attack(random_attack_dict, filename):
    softmax_rob = [1-diff for diff in random_attack_dict["softmax_diff"]]
    dict = {"Attack": random_attack_dict["attack_type"],
                          "samples":random_attack_dict["samples"],
                          # "softmax_rob":random_attack_dict["softmax_diff"]}
                          "softmax_rob":softmax_rob}
    df = pd.DataFrame(data=dict)
    print(df.head())
    print(df.describe())

    sns.set()
    plt.subplots(figsize=(8, 8), dpi=100)
    sns.set_palette("YlGnBu_d",3)
    g = sns.lineplot(x="samples", y="softmax_rob", hue="Attack", style="Attack", data=df, ci=60)
    g.set(xticks=df.samples.values)

    plt.xlabel("Samples involved in expectations ($w_i \sim p(w|D)$)", fontsize=10)
    plt.ylabel('Softmax robustness ($l_\infty$)', fontsize=10)

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    plt.savefig(RESULTS+filename)


def attack_model(model, test_loader, n_attack_samples_list, device, eps, loss_gradients=None, n_pred_samples=None):

    attack_samples = []
    softmax_diff = []
    attack_type = []

    if DEBUG:
        # check on correct posteriors
        for n_attack_samples in n_attack_samples_list:
            if n_pred_samples is None:
                n_pred_samples = n_attack_samples
            print("\nn_attack_samples = ", n_attack_samples)
            for images, labels in test_loader:
                for idx in range(len(images)):
                    image = images[idx]
                    label = labels[idx]

                    input_shape = image.size(0) * image.size(1) * image.size(2)
                    label = label.to(device).argmax(-1).view(-1)
                    image = image.to(device).view(-1, input_shape)
                    print(model.forward(image, n_samples=5))
                    print(model.forward(image, n_samples=5))
                    print(torch.all(torch.eq(model.forward(image, n_samples=5),model.forward(image, n_samples=5))))
        exit()

    for n_attack_samples in n_attack_samples_list:
        if n_pred_samples is None:
            n_pred_samples = n_attack_samples
        print("\nn_attack_samples = ", n_attack_samples)
        for images, labels in test_loader:
            for idx in tqdm(range(len(images))):
                print("new image")
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
                attack_samples.append(n_attack_samples)
                attack_type.append("random")

                # == bayesian fgsm attack ==
                #
                # if loss_gradients is not None:
                #     loss_gradient = torch.tensor(loss_gradients[str(attack_samples)][im_count])
                attack_dict = fgsm_bayesian_attack(model=model, image=image,
                                                   label=label, epsilon=eps, device=device,
                                                   n_pred_samples=n_pred_samples,
                                                   n_attack_samples=n_attack_samples, #loss_gradient=loss_gradient
                                                   )
                original_output = attack_dict["original_output"].clone().detach().to(device)
                adversarial_output = attack_dict["adversarial_output"].clone().detach().to(device)
                difference = (original_output - adversarial_output).abs().max(dim=-1)[0].item()
                softmax_diff.append(difference)
                attack_samples.append(n_attack_samples)
                attack_type.append("fgsm")

                # == bayesian pgd attack ==
                attack_dict = pgd_bayesian_attack(model=model, image=image,
                                                   label=label, epsilon=eps, device=device,
                                                   n_pred_samples=n_pred_samples,
                                                   n_attack_samples=n_pred_samples)
                original_output = attack_dict["original_output"].clone().detach().to(device)
                adversarial_output = attack_dict["adversarial_output"].clone().detach().to(device)
                difference = (original_output - adversarial_output).abs().max(dim=-1)[0].item()
                softmax_diff.append(difference)
                attack_samples.append(n_attack_samples)
                attack_type.append("pgd")

    random_attack_dict = {"samples":attack_samples, "softmax_diff":softmax_diff, "attack_type":attack_type}
    return random_attack_dict


def get_filename(dataset, pred_samples, epsilon):
    return str(dataset)+"_predSamp="+str(pred_samples)+"_random_attack_eps="+str(epsilon)


# def final_plot(pred_samples, path=DATA_PATH):
#     matplotlib.rc('font', **{'weight': 'bold', 'size': 8})
#     fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
#     sns.set()
#     # sns.set_palette("YlGnBu_d",3)
#     # sns.set_palette("viridis",4)
#     sns.set_palette("gist_heat",3)
#
#     # n_inputs, n_samples_list = 1000, [1,10,50,100]
#
#     for col_idx, pred_samples in enumerate([None, pred_samples]):
#         plot_attack = []
#         plot_samples = []
#         plot_diff = []
#         for row_idx, dataset in enumerate(["mnist","fashion_mnist"]):
#             dir = path+str(dataset)+"_predSamp="+str(pred_samples)+"/"
#             for file in os.listdir(dir):
#                 if file.endswith(".pkl"):
#                     random_attack_dict=load_from_pickle(dir+file)
#                     for idx in range(len(random_attack_dict["softmax_diff"])):
#                         if random_attack_dict["softmax_diff"][idx] > 0.0:
#                             # print("\n",random_attack_dict["softmax_diff"][idx])
#                             plot_attack.append(random_attack_dict["attack_type"][idx])
#                             plot_samples.append(random_attack_dict["samples"][idx])
#                             plot_diff.append(random_attack_dict["softmax_diff"][idx])
#
#             softmax_rob = [1-diff for diff in plot_diff]
#             df = pd.DataFrame(data={"Attack": plot_attack, "samples":plot_samples, "softmax_rob":softmax_rob})
#             print(df.head())
#
#             g = sns.lineplot(x="samples", y="softmax_rob", hue="Attack", style="Attack", data=df, ci=60,
#                              ax=ax[row_idx, col_idx])
#             g.set(xticks=df.samples.values)
#
#             ax[row_idx, col_idx].set_xlabel("")
#             ax[row_idx, col_idx].set_ylabel("")
#             ax[0, 1].set_ylabel("MNIST", fontsize=9, rotation=270,labelpad=10)
#             ax[0, 1].yaxis.set_label_position("right")
#             ax[1, 1].set_ylabel(f"Fashion MNIST", fontsize=9, rotation=270,labelpad=10)
#             ax[1, 1].yaxis.set_label_position("right")
#
#             if row_idx == 0 and col_idx == 1:
#                 ax[row_idx, col_idx].legend(loc='upper right', title="Attack", labels=["random", "FGSM","PGD"])
#             else:
#                 ax[row_idx, col_idx].legend().remove()
#
#             ax[0, 0].set_title(f"Predictive samples = attack samples ", fontsize=10)
#             ax[0, 1].set_title(f"Predictive samples = {pred_samples}", fontsize=10)
#
#     fig.text(0.04, 0.5, r'Softmax robustness ($l_\infty$)', fontsize=11, va='center', rotation='vertical')
#     fig.text(0.5, 0.04, r"Samples involved in the attacks ($w_i \sim p(w|D)$)", ha='center', fontsize=11)
#
#
#     os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
#     plt.savefig(RESULTS+"attacks_comparison_VI.png")

def final_plot_2(pred_samples, eps, path=DATA_PATH):
    matplotlib.rc('font', **{'weight': 'bold', 'size': 8})
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 8), dpi=300, facecolor='w', edgecolor='k')
    sns.set()
    # sns.set_palette("YlGnBu_d",3)
    # sns.set_palette("viridis",4)
    sns.set_palette("gist_heat",5)

    # n_inputs, n_samples_list = 1000, [1,10,50,100]
    plot_attack = []
    plot_samples = []
    plot_rob = []

    for row_idx, dataset in enumerate(["mnist", "fashion_mnist"]):
        for samples in [None, pred_samples]:
            filename = get_filename(dataset=dataset, pred_samples=pred_samples, epsilon=eps)

            # dir = path+str(dataset)+"_predSamp="+str(samples)+"/"
            # for file in os.listdir(dir):
            #     if file.endswith(".pkl"):
            random_attack_dict=load_from_pickle(path+"eps="+str(eps)+filename)
            for idx in range(len(random_attack_dict["softmax_diff"])):
                if random_attack_dict["softmax_diff"][idx] > 0.0:
                    if samples == None:
                        if random_attack_dict["attack_type"][idx] == "fgsm":
                            plot_attack.append("FGSM")
                        elif random_attack_dict["attack_type"][idx] == "pgd":
                            plot_attack.append("PGD")
                        elif random_attack_dict["attack_type"][idx] == "random":
                            plot_attack.append("RAND")
                        plot_samples.append(random_attack_dict["samples"][idx])
                        plot_rob.append(1.0-random_attack_dict["softmax_diff"][idx])
                    else:
                        if random_attack_dict["attack_type"][idx] == "fgsm":
                            plot_attack.append("FFGSM")
                        elif random_attack_dict["attack_type"][idx] == "pgd":
                            plot_attack.append("FPGD")
                        elif random_attack_dict["attack_type"][idx] == "random":
                            plot_attack.append("RAND")
                        plot_samples.append(random_attack_dict["samples"][idx])
                        plot_rob.append(1.0-random_attack_dict["softmax_diff"][idx])

        df = pd.DataFrame(data={"Attack": plot_attack, "samples":plot_samples, "softmax_rob":plot_rob})
        print(df.head(20))
        # print(df[df["Attack"]=="FGSM"])
        # exit()
        g = sns.lineplot(x="samples", y="softmax_rob", hue="Attack", style="Attack", data=df, ci=60,
                         ax=ax[row_idx])
        g.set(xticks=df.samples.values)

        ax[row_idx].set_xlabel("")

    ax[0].set_ylabel("MNIST", fontsize=9, rotation=270,labelpad=10)
    ax[0].yaxis.set_label_position("right")
    ax[1].set_ylabel(f"Fashion MNIST", fontsize=9, rotation=270,labelpad=10)
    ax[1].yaxis.set_label_position("right")

    ax[0].legend(loc='lower right',
                 #, title="Attack",
                 # labels=["Random","FGSM","PGD","Fixed Sample FGSM","Fixed Sample PGD"]
                 )
    ax[1].legend().remove()

    # ax[0].set_title(f"Predictive samples = attack samples ", fontsize=10)
    # ax[0].set_title(f"Predictive samples = {pred_samples}", fontsize=10)

    fig.text(0.04, 0.5, r'Softmax robustness ($l_\infty$)', fontsize=11, va='center', rotation='vertical')
    fig.text(0.5, 0.04, r"Samples involved in the attacks ($w_i \sim p(w|D)$)", ha='center', fontsize=11)


    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    plt.savefig(RESULTS+"attacks_comparison_VI.png")


def final_single_col_plot(pred_samples, dataset):
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from matplotlib.pyplot import figure
    from matplotlib.lines import Line2D

    samples = []
    fgsm = []
    pgd = []
    ffgsm = []
    fpgd = []
    rand = []

    dir = DATA_PATH+str(dataset)+"_predSamp="+str(None)+"/"
    for file in os.listdir(dir):
        if file.endswith(".pkl"):
            random_attack_dict=load_from_pickle(dir+file)
            for idx in range(len(random_attack_dict["softmax_diff"])):
                if random_attack_dict["samples"][idx] != 50:
                    if random_attack_dict["softmax_diff"][idx] > 0.0:
                        # print("\n",random_attack_dict["softmax_diff"][idx])
                        if random_attack_dict["attack_type"][idx] == "fgsm":
                            samples.append(random_attack_dict["samples"][idx])
                            fgsm.append(1 - random_attack_dict["softmax_diff"][idx])
                            pgd.append(np.nan)
                            ffgsm.append(np.nan)
                            fpgd.append(np.nan)
                            rand.append(np.nan)
                        elif random_attack_dict["attack_type"][idx] == "pgd":
                            samples.append(random_attack_dict["samples"][idx])
                            pgd.append(1 - random_attack_dict["softmax_diff"][idx])
                            fgsm.append(np.nan)
                            ffgsm.append(np.nan)
                            fpgd.append(np.nan)
                            rand.append(np.nan)
                        elif random_attack_dict["attack_type"][idx] == "random":
                            samples.append(random_attack_dict["samples"][idx])
                            rand.append(1 - random_attack_dict["softmax_diff"][idx])
                            pgd.append(np.nan)
                            ffgsm.append(np.nan)
                            fpgd.append(np.nan)
                            fgsm.append(np.nan)

    dir = DATA_PATH+str(dataset)+"_predSamp="+str(pred_samples)+"/"
    for file in os.listdir(dir):
        if file.endswith(".pkl"):
            random_attack_dict=load_from_pickle(dir+file)
            for idx in range(len(random_attack_dict["softmax_diff"])):
                if random_attack_dict["samples"][idx] != 50:

                    if random_attack_dict["softmax_diff"][idx] > 0.0:
                        # print("\n",random_attack_dict["softmax_diff"][idx])
                        if random_attack_dict["attack_type"][idx] == "fgsm":
                            samples.append(random_attack_dict["samples"][idx])
                            ffgsm.append(1 - random_attack_dict["softmax_diff"][idx])
                            pgd.append(np.nan)
                            fgsm.append(np.nan)
                            fpgd.append(np.nan)
                            rand.append(np.nan)
                        elif random_attack_dict["attack_type"][idx] == "pgd":
                            samples.append(random_attack_dict["samples"][idx])
                            fpgd.append(1 - random_attack_dict["softmax_diff"][idx])
                            fgsm.append(np.nan)
                            ffgsm.append(np.nan)
                            pgd.append(np.nan)
                            rand.append(np.nan)
                        elif random_attack_dict["attack_type"][idx] == "random":
                            samples.append(random_attack_dict["samples"][idx])
                            rand.append(1 - random_attack_dict["softmax_diff"][idx])
                            pgd.append(np.nan)
                            ffgsm.append(np.nan)
                            fpgd.append(np.nan)
                            fgsm.append(np.nan)

    df = pd.DataFrame({"Samples":samples,"FGSM":fgsm,"PGD":pgd,"FFGSM":ffgsm,"FPGD":fpgd,"RAND":rand})
    print(df.head(20))

    sns.set_style('darkgrid')

    fig = figure(num=None, figsize=(8, 4), dpi=120, facecolor='w', edgecolor='k')

    font = {'weight': 'bold',
            'size': 16}

    matplotlib.rc('font', **font)
    sns.set_palette(sns.color_palette("gist_heat", 3))
    pal = sns.color_palette("gist_heat", 3)
    pal2 = sns.color_palette("ocean_r", 7)

    ax1 = plt.gca()
    plt.title("Change in softmax ($l_\infty$) for attacks")
    plt.xlabel("Samples involved in attack ($w_i \sim p(w|D)$)")
    plt.ylabel(r"Softmax difference ($l_\infty$)")

    sns.catplot(x="Samples", y="FFGSM", markers=["^"], linestyles=["-"], kind="point", data=df, ax=ax1, color=pal[0])
    sns.catplot(x="Samples", y="FGSM", markers=["o"], linestyles=["-"], kind="point", data=df, ax=ax1, color=pal[2])
    sns.catplot(x="Samples", y="FPGD", markers=["^", "o"], linestyles=["-", "--"],
                kind="point", data=df, ax=ax1, color=pal2[2])
    sns.catplot(x="Samples", y="PGD", markers=["^", "o"], linestyles=["-", "--"],
                kind="point", data=df, ax=ax1, color=pal2[0])

    plt.close(2)
    plt.close(3)
    plt.close(4)
    plt.close(5)

    # plt.title("Quantitative Robustness (MNIST)")
    plt.xlabel("Samples involved in attack ($w_i \sim p(w|D)$)")
    plt.ylabel(r"1 - Softmax difference ($l_\infty$)")

    m = np.mean(df['RAND'])
    s = np.std(df['RAND']) / 1.5
    plt.hlines(m, 0, 2, 'r', linewidth=5)
    x = np.asarray([m for i in range(3)])
    plt.fill_between(range(3), x - s, x + s, color='r', alpha=0.1)

    legend_elements = [Line2D([0], [0], color=pal[0], lw=4, label='Fixed Sample FGSM'),
                       Line2D([0], [0], color=pal2[2], lw=4, label='Fixed Sample PGD'),
                       Line2D([0], [0], color=pal[2], lw=4, label='FGSM Attack'),
                       Line2D([0], [0], color=pal2[0], lw=4, label='PGD Attack'),
                       Line2D([0], [0], color='#db4444', lw=4, label='Random Attack')]
    ax1.legend(handles=legend_elements, prop={'size': 10})
    plt.ylim([0, 1.1])

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    plt.savefig(RESULTS+"single_col_attacks_comparison_VI_"+str(dataset)+".png")


def main(args):

    eps = 0.4
    test_images = 10
    pred_samples = None # None or int

    # == final plot ==
    # final_plot_2(pred_samples=100, path=RESULTS, eps=eps)
    # # # final_plot(pred_samples=pred_samples, path=DATA_PATH)
    # # # # final_single_col_plot(pred_samples=pred_samples, dataset=args.dataset)
    # exit()

    # == produce attacks ==
    n_attack_samples_list = [1,10,50]
    # loss_gradients_test_images = 1000

    if args.dataset == "mnist":
        model = hidden_vi_models[2]
    elif args.dataset == "fashion_mnist":
        model = hidden_vi_models[5]
    else:
        raise AssertionError("wrong dataset name")

    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=args.dataset, batch_size=128, n_inputs=test_images)

    bayesnn = VI_BNN(input_shape=input_shape, device=args.device, architecture=model["architecture"],
                     activation=model["activation"])
    posterior = bayesnn.load_posterior(posterior_name=model["filename"], relative_path=TRAINED_MODELS,
                                       activation=model["activation"], dataset_name=args.dataset)

    # loss_gradients = {}
    # for n_attack_samples in n_samples_list:
    #     gradients = load_loss_gradients(dataset_name=args.dataset, n_inputs=loss_gradients_test_images,
    #                                     n_samples=n_attack_samples,  model_idx=model["idx"])
    #     loss_gradients.update({str(n_attack_samples):gradients})

    random_attack_dict = attack_model(model=posterior, n_pred_samples=pred_samples, #loss_gradients=loss_gradients,
                                      n_attack_samples_list=n_attack_samples_list, device=args.device,
                                      test_loader=test_loader, eps=eps)
    filename = get_filename(dataset=args.dataset, pred_samples=pred_samples, epsilon=eps)
    save_to_pickle(data=random_attack_dict, relative_path=RESULTS+"eps="+str(eps)+"/", filename=filename+".pkl")

    # random_attack_dict=load_from_pickle(RESULTS+filename+".pkl")
    # random_attack_dict=load_from_pickle(DATA_PATH+filename+".pkl")
    plot_random_attack(random_attack_dict=random_attack_dict, filename=filename+".png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cuda', type=str, help='use "cpu" or "cuda".')
    parser.add_argument("--dataset", default='mnist', type=str, help='use "mnist" or "fashion_mnist".')
    main(args=parser.parse_args())

