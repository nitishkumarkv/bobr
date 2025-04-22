import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

from scipy.stats import multivariate_normal

########################################################################################
# functions copied from: https://github.com/FloMau/gato/tree/master/
########################################################################################

def sample_gaussian(n_events, mean, cov, seed=None):
    """
    Draw n_events from N(mean, cov) in ℝ³.
    """
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, cov, size=n_events)


def sample_background(n_events, lam):
    """
    Sample NN outputs for a background using an exponential decay distribution.
    The PDF is proportional to exp(-lam * x) for x in [0,1].
    Uses inverse transform sampling.
    """
    u = np.random.uniform(0, 1, n_events)
    # Normalization factor: integral_0^1 exp(-lam * x) dx = (1 - exp(-lam))/lam
    # Inverse CDF:
    x = -1.0 / lam * np.log(1 - u * (1 - np.exp(-lam)))
    return x

def sample_signal(n_events, lam):
    """
    Sample NN outputs for a signal using a reversed exponential distribution.
    The PDF is proportional to exp(-lam*(1 - x)) for x in [0,1].
    Uses inverse transform sampling.
    """
    u = np.random.uniform(0, 1, n_events)
    # For reversed exponential, let z = 1-x ~ Exponential(lam) in [0,1]
    z = -1.0 / lam * np.log(1 - u * (1 - np.exp(-lam)))
    x = 1 - z
    return x

def generate_toy_data(n_signal=1000, n_bkg1=1000, n_bkg2=1000, n_bkg3=1000,
                      lam_signal = 5, lam_bkg1=10, lam_bkg2=5, lam_bkg3=2, 
                      xs_signal=0.5, xs_bkg1=100, xs_bkg2=10, xs_bkg3=1,
                      lumi=100, seed=None, output_dir="./toy_data"):
    """
    Generate toy NN output data for one signal and three backgrounds.
    
    Parameters:
      - n_signal: number of generated events for the signal.
      - n_bkgX: number of generated events for each background.
      - xs_signal: cross section for the signal in pb (500 fb = 0.5 pb).
      - xs_bkg1, xs_bkg2, xs_bkg3: cross sections for the backgrounds in pb.
      - lumi: integrated luminosity in /fb.
      - seed: (optional) seed for reproducibility.
      
    Returns:
      A dictionary mapping process names to pandas DataFrames with a column "NN_output".
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate NN outputs
    nn_signal = sample_signal(n_signal, lam_signal)
    nn_bkg1 = sample_background(n_bkg1, lam_bkg1)
    nn_bkg2 = sample_background(n_bkg2, lam_bkg2)
    nn_bkg3 = sample_background(n_bkg3, lam_bkg3)

    # Optionally, one could compute per-event weights based on cross section and lumi,
    # e.g., weight = (xs * lumi) / n_generated
    weight_signal = xs_signal * lumi / n_signal
    weight_bkg1   = xs_bkg1   * lumi / n_bkg1
    weight_bkg2   = xs_bkg2   * lumi / n_bkg2
    weight_bkg3   = xs_bkg3   * lumi / n_bkg3

    # Create DataFrames (here we only store the NN output, but you could add weights if needed)
    df_signal = pd.DataFrame({"NN_output": nn_signal, "weight": weight_signal})
    df_bkg1 = pd.DataFrame({"NN_output": nn_bkg1, "weight": weight_bkg1})
    df_bkg2 = pd.DataFrame({"NN_output": nn_bkg2, "weight": weight_bkg2})
    df_bkg3 = pd.DataFrame({"NN_output": nn_bkg3, "weight": weight_bkg3})

    data = {
        "signal": df_signal,
        "bkg1": df_bkg1,
        "bkg2": df_bkg2,
        "bkg3": df_bkg3
    }
    for proc, df in data.items():
        print(f"{proc}: {len(df)} events, NN_output min={df['NN_output'].min():.3f}, max={df['NN_output'].max():.3f}")
        # save the dataframe
        df.to_parquet(f"{output_dir}/{proc}.parquet")

    return data

def generate_toy_data_gauss(
    n_signal=100000, n_bkg1=200000, n_bkg2=100000, n_bkg3=100000,
    xs_signal=0.5, xs_bkg1=50, xs_bkg2=15, xs_bkg3=1,
    lumi=100, seed=None
):
    """
    3D Gaussian toy: sample points for signal & 3 bkgs, then compute
    the likelihood-ratio discriminant (sig vs sum of bkgs), mapped to [0,1].
    Returns dict of DataFrames with columns “NN_output” and “weight”.
    """
    if seed is not None:
        np.random.seed(seed)

    # 1) define means & covariances
    means = {
        "signal": np.array([ 0.5,  0.5,  1.0]),
        "bkg1":   np.array([ -0.5,  1.,  -1]),
        "bkg2":   np.array([ 0.25,  -0.25,  3.0]),
        "bkg3":   np.array([ 0.75,  0.25,  -0.5]),
    }

    covs = {
        "signal": np.array([[1, 0.2, 0.1],
                            [0.2, 1, 0.2],
                            [0.1, 0.2, 1]]),
        "bkg1":   np.array([[0.5, 0.2, 0.0],
                            [0.2, 0.5, 0.2],
                            [0.0, 0.2, 0.5]]),
        "bkg2":   np.array([[0.5, 0.1, 0.3],
                            [0.1, 0.5, 0.1],
                            [0.3, 0.1, 0.5]]),
        "bkg3":   np.array([[0.5, 0.2, 0.4],
                            [0.2, 0.5, 0.2],
                            [0.4, 0.2, 0.5]]),
    }

    # 2) how many to draw and what weights
    counts = {
        "signal": n_signal,
        "bkg1":   n_bkg1,
        "bkg2":   n_bkg2,
        "bkg3":   n_bkg3,
    }
    xs = {
        "signal": xs_signal,
        "bkg1":   xs_bkg1,
        "bkg2":   xs_bkg2,
        "bkg3":   xs_bkg3,
    }

    # 3) sample
    raw = {}
    for proc in means:
        X = sample_gaussian(counts[proc], means[proc], covs[proc], seed=seed)
        w = xs[proc] * lumi / counts[proc]
        raw[proc] = {"X": X, "w": w}

    # 4) build scipy MVN pdfs
    pdfs = {proc: multivariate_normal(means[proc], covs[proc])
            for proc in means}
    
    # total cross section of background
    total_bkg_xs = sum(xs[p] for p in pdfs if p != "signal")

    # 5) compute optimal discriminant for each proc’s points:
    dfs = {}
    for proc, info in raw.items():
        X = info["X"]
        w = info["w"]
        p_sig = pdfs["signal"].pdf(X)
        # sum of background pdfs at X
        p_bkg = sum((xs[p] / total_bkg_xs) * pdfs[p].pdf(X) for p in pdfs if p != "signal")

        # noise 
        p_sig *= np.abs((1 + np.random.normal(scale=0.2, size=p_sig.shape)))
        p_bkg *= np.abs((1 + np.random.normal(scale=0.2, size=p_bkg.shape)))
        # likelihood‐ratio and map to [0,1] via sigmoid‐like:
        lr = p_sig / (p_bkg + 1e-12)

        disc = lr / (1.0 + lr)
        # disc = 2 * tf.nn.sigmoid(lr) - 1
        dfs[proc] = pd.DataFrame({
            "NN_output": disc,
            "weight":    w
        })

    return dfs


def generate_toy_data_multiclass(n_signal=1000, n_bkg1=1000, n_bkg2=1000,
                                 # means chosen so that, after softmax, the highest probability is for the desired class.
                                 mean_signal=[1.5, -1.5, -1.5],
                                 mean_bkg1=[-1.0, 0.0, -2.0],
                                 mean_bkg2=[-2.0, -2.0, 0.0],
                                 # common covariance introduces moderate correlations
                                 cov=[[1.0, 0.3, 0.3],
                                      [0.3, 1.0, 0.3],
                                      [0.3, 0.3, 1.0]],
                                 xs_signal=0.5, xs_bkg1=100, xs_bkg2=10,
                                 lumi=100, seed=None):
    """
    Generate toy NN output data for three processes: signal, bkg1, and bkg2.
    
    Each event is a 3D vector, which is passed through a softmax so that the output
    is a probability vector over the three classes. The distributions are chosen such that:
      - signal events are biased to have high probability in the first component,
      - bkg1 events are biased to have high probability in the second component,
      - bkg2 events are biased to have high probability in the third component.
    
    Parameters:
      - n_signal, n_bkg1, n_bkg2: Number of events for each process.
      - mean_signal, mean_bkg1, mean_bkg2: Mean vectors for the multivariate normal distributions.
      - cov: Covariance matrix (same for all processes here).
      - xs_signal, xs_bkg1, xs_bkg2: Cross sections (in pb) for signal and backgrounds.
      - lumi: Integrated luminosity in /fb.
      - seed: (optional) random seed for reproducibility.
      
    Returns:
      A dictionary mapping process names to pandas DataFrames with columns:
        - "NN_output": a list/array of 3 softmax-activated values,
        - "weight": per-event weight.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate raw 3D outputs for each process
    raw_signal = np.random.multivariate_normal(mean_signal, cov, n_signal)
    raw_bkg1   = np.random.multivariate_normal(mean_bkg1, cov, n_bkg1)
    raw_bkg2   = np.random.multivariate_normal(mean_bkg2, cov, n_bkg2)

    def softmax(x, axis=-1):
        # Shift values for numerical stability
        shifted_x = x - np.max(x, axis=axis, keepdims=True)
        exponent = np.exp(shifted_x)
        return exponent / np.sum(exponent, axis=axis, keepdims=True)

    # Apply softmax row-wise
    signal_softmax = softmax(np.array(raw_signal, dtype=np.float32), axis=1)
    bkg1_softmax   = softmax(np.array(raw_bkg1, dtype=np.float32), axis=1)
    bkg2_softmax   = softmax(np.array(raw_bkg2, dtype=np.float32), axis=1)

    # Compute per-event weights based on cross section and lumi
    weight_signal = xs_signal * lumi / n_signal
    weight_bkg1   = xs_bkg1   * lumi / n_bkg1
    weight_bkg2   = xs_bkg2   * lumi / n_bkg2

    # Create DataFrames; we store the NN_output as a list for each event.
    df_signal = pd.DataFrame({
        "NN_output": list(signal_softmax),
        "weight": weight_signal
    })
    df_bkg1 = pd.DataFrame({
        "NN_output": list(bkg1_softmax),
        "weight": weight_bkg1
    })
    df_bkg2 = pd.DataFrame({
        "NN_output": list(bkg2_softmax),
        "weight": weight_bkg2
    })

    data = {
        "signal": df_signal,
        "bkg1": df_bkg1,
        "bkg2": df_bkg2
    }
    
    return data

# If executed directly, generate toy data and print summary statistics.
#if __name__ == "__main__":



import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

def generate_toy_data_3D(
    n_signal1=100000, n_signal2=100000,
    n_bkg1=100000, n_bkg2=80000, n_bkg3=50000, n_bkg4=20000, n_bkg5=10000,
    xs_signal1=0.5, xs_signal2=0.1,
    xs_bkg1=100, xs_bkg2=80, xs_bkg3=50, xs_bkg4=20, xs_bkg5=10,
    lumi=100.0, noise_scale=0.2, seed=None
):
    """
    Generate 3D Gaussian data for 2 signal and 5 background classes.
    For each point, compute likelihood-ratio-based 3-class scores:
        [score_signal1, score_signal2, score_background]
    Returns dict of DataFrames with columns: 'NN_output' (3-vector) and 'weight'.
    """
    if seed is not None:
        np.random.seed(seed)

    processes = ["bkg1", "bkg2", "bkg3", "bkg4", "bkg5", "signal1", "signal2"]

    means = {
        "signal1": np.array([1.5, -1.0, -1.0]),
        "signal2": np.array([-1.0, 1.5, -1.0]),
        "bkg1":    np.array([-0.5, -0.5, 1.0]),
        "bkg2":    np.array([0.5, -0.5, 0.8]),
        "bkg3":    np.array([0.5, 0.5, -0.6]),
        "bkg4":    np.array([-0.5, 1.0, -0.4]),
        "bkg5":    np.array([-0.5, 0.5, -0.2])
    }

    # Slightly correlated 3D Gaussian
    cov = np.eye(3)*1.0 + 0.2*(np.ones((3,3)) - np.eye(3))

    counts = {
        "signal1": n_signal1, "signal2": n_signal2,
        "bkg1": n_bkg1, "bkg2": n_bkg2, "bkg3": n_bkg3,
        "bkg4": n_bkg4, "bkg5": n_bkg5
    }

    xs = {
        "signal1": xs_signal1, "signal2": xs_signal2,
        "bkg1": xs_bkg1, "bkg2": xs_bkg2, "bkg3": xs_bkg3,
        "bkg4": xs_bkg4, "bkg5": xs_bkg5
    }

    # 1. Sample raw 3D data
    raw = {
        p: np.random.multivariate_normal(mean=means[p], cov=cov, size=counts[p])
        for p in processes
    }

    # 2. Add multiplicative noise
    for p in processes:
        noise = np.random.normal(loc=1.0, scale=noise_scale, size=raw[p].shape)
        raw[p] *= noise

    # 3. Build PDFs
    pdfs = {
        p: multivariate_normal(mean=means[p], cov=cov)
        for p in processes
    }

    # 4. Combined background PDF with proper cross-section weighting
    bkg_processes = [p for p in processes if p.startswith("bkg")]
    total_bkg_xs = sum(xs[p] for p in bkg_processes)

    def combined_bkg_pdf(X):
        return sum(
            (xs[p] / total_bkg_xs) * pdfs[p].pdf(X)
            for p in bkg_processes
        )

    # 5. Compute likelihood-ratio-based scores
    data = {}
    for proc in processes:
        X = raw[proc]
        weight = xs[proc] * lumi / counts[proc]

        p1 = pdfs["signal1"].pdf(X)
        p2 = pdfs["signal2"].pdf(X)
        pb = combined_bkg_pdf(X)

        total = p1 + p2 + pb + 1e-12  # avoid divide-by-zero

        score1 = p1 / total
        score2 = p2 / total
        score_bkg = pb / total

        nn_output = np.stack([score1, score2, score_bkg], axis=1)
        nn_output = [row for row in nn_output]

        data[proc] = pd.DataFrame({
            "NN_output": nn_output,
            "weight": weight
        })

    return data
# If this module is executed directly, generate toy data and print summary stats.
if __name__ == "__main__":
#    toy_data = generate_toy_data(
#    n_signal=100000,
#    n_bkg1=200000, n_bkg2=100000, n_bkg3=100000,
#    lam_signal = 6, lam_bkg1=7, lam_bkg2=7, lam_bkg3=3,
#    xs_signal=0.5,    # 500 fb = 0.5 pb
#    xs_bkg1=50, xs_bkg2=15, xs_bkg3=1,
#    lumi=100,         # in /fb
#    seed=42
#)
#    for proc, df in toy_data.items():
#        print(f"{proc}: {len(df)} events, NN_output min={df['NN_output'].min():.3f}, max={df['NN_output'].max():.3f}")
#        # save the dataframe
#        df.to_parquet(f"toy_data/{proc}.parquet")
    
    toy_data = generate_toy_data_multiclass(seed=42)
    for proc, df in toy_data.items():
        # Extract the NN_output as a 2D array for summary stats.
        outputs = np.stack(df["NN_output"].values)
        print(f"{proc}: {len(df)} events")
        print(f"  NN_output: mean = {outputs.mean(axis=0)}, std = {outputs.std(axis=0)}")
        # save the dataframe
        df.to_parquet(f"toy_data_multiclass/{proc}.parquet")

