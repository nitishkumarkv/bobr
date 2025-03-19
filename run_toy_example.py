from examples.toy_example.generate_toy_data import generate_toy_data
from utils.plotting import plot_stacked_histograms
from bobr.cat_optimizer import BOBRBinOptimizer
import os
import argparse
import pandas as pd
import hist
import matplotlib.pyplot as plt

def create_hist(df, bin_edges=None):

    # create list of hist and fill them
    if bin_edges is None:
        h = hist.Hist(hist.axis.Regular(50, 0, 1, name="NN_output"), storage=hist.storage.Weight())
    else:
        h = hist.Hist(hist.axis.Variable(bin_edges, name="NN_output"), storage=hist.storage.Weight())
    h.fill(df["NN_output"], weight=df["weight"])

    return h

def main():

    # initialize the parser
    parser = argparse.ArgumentParser(description="Run the toy example")
    parser.add_argument("--output-dir", type=str, default="toy_results", help="Output directory")
    parser.add_argument("--generate-toy-data", action="store_true", help="Generate toy data")
    parser.add_argument("--plot-toy-data", action="store_true", help="Plot toy data")
    args = parser.parse_args()

    # create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    data_path = os.path.join(args.output_dir, "toy_data")

    bkg_list = ["bkg1", "bkg2", "bkg3"]
    signal_list = ["signal"]
    
    # Generate toy data
    if args.generate_toy_data:
        print("INFO: Generating toy data")
        os.makedirs(data_path, exist_ok=True)

        toy_data = generate_toy_data(
            n_signal=100000,
            n_bkg1=200000, n_bkg2=100000, n_bkg3=100000,
            lam_signal=6, lam_bkg1=7, lam_bkg2=7, lam_bkg3=3,
            xs_signal=0.5,  # 500 fb = 0.5 pb
            xs_bkg1=50, xs_bkg2=15, xs_bkg3=1,
            lumi=100,  # in /fb
            seed=42,
            output_dir=data_path
        )
    
    toy_data = {}
    # load toy_data from all files in data_path
    print("INFO: Loading toy data")
    for files in os.listdir(data_path):
        if files.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(data_path, files))
            toy_data[files.split(".")[0]] = df
    

    print("INFO: Plotting toy data")
    # create list of hist
    bkg_hists = [create_hist(toy_data[bkg]) for bkg in bkg_list]
    signal_hists = [create_hist(toy_data[signal]) for signal in signal_list]

    if args.plot_toy_data:
        plot_stacked_histograms(
            bkg_hists,
            bkg_list,
            output_filename=f"{data_path}/plot_log.pdf",
            axis_labels=("NN output", "Events"),
            signal_hists=signal_hists,
            signal_labels=signal_list,
            signal_scale=100,
            normalize=False,
            log=True,
            log_min=1e-5,
        )
        plot_stacked_histograms(
            bkg_hists,
            bkg_list,
            output_filename=f"{data_path}/plot.pdf",
            axis_labels=("NN output", "Events"),
            signal_hists=signal_hists,
            signal_labels=signal_list,
            signal_scale=100,
            normalize=False,
            log=False,
            log_min=1e-5,
        )

    # Run the optimizer
    print("INFO: Running the optimizer")
    nbins_list = [5, 10, 15, 20, 25, 30]
    best_Z_list = []
    for n_bins in nbins_list:
        if n_bins > 10:
            n_trials = 500
        else:
            n_trials = 200
        optimizer = BOBRBinOptimizer(toy_data, bkg_label_lst=bkg_list, signal_label_lst=signal_list,
                                     var_label='NN_output', weight_label='weight', n_bins=n_bins, output_dir=args.output_dir+f"/optimizer_results_nbins_{n_bins}", n_trials=n_trials)
        best_bins, best_hist_dict, best_Z = optimizer.optimize_bins()
        best_Z_list.append(best_Z)
        print("Optimized bin edges:", best_bins)
        print("Best significance:", best_Z)
        optimizer.visualize_optimization()

        # Plot the optimized bins
        print("INFO: Plotting the optimized bins")
        best_signal_hists = [create_hist(toy_data[signal], bin_edges=best_bins) for signal in signal_list]
        best_bkg_hists = [create_hist(toy_data[bkg], bin_edges=best_bins) for bkg in bkg_list]
        plot_stacked_histograms(
            best_bkg_hists,
            bkg_list,
            output_filename=f"{args.output_dir}/optimizer_results_nbins_{n_bins}/plot_log.pdf",
            axis_labels=("NN output", "Events"),
            signal_hists=best_signal_hists,
            signal_labels=signal_list,
            signal_scale=100,
            normalize=False,
            log=True,
            log_min=1e-5,
        )
        plot_stacked_histograms(
            best_bkg_hists,
            bkg_list,
            output_filename=f"{args.output_dir}/optimizer_results_nbins_{n_bins}/plot.pdf",
            axis_labels=("NN output", "Events"),
            signal_hists=best_signal_hists,
            signal_labels=signal_list,
            signal_scale=100,
            normalize=False,
            log=False,
            log_min=1e-5,
        )

    # plot the significance vs n_bins
    plt.figure()
    plt.plot(nbins_list, best_Z_list)
    plt.xlabel("n_bins")
    plt.ylabel("Significance")
    plt.savefig(f"{args.output_dir}/significance_vs_nbins.pdf")



if __name__ == "__main__":
    print("INFO: Running the toy example")
    main()

    


