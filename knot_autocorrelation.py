import time
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from knot_evolution import *

def analyze_algorithms(array, knot, timesteps, sampler):

    start_time = time.time()
    pivot_array, pivot_metrics = pivot_with_metrics(array, timesteps, knot)
    pivot_time = time.time() - start_time

    start_time = time.time()
    bfacf_array, bfacf_metrics = BFACF_with_metrics(array, timesteps, sampler)
    bfacf_time = time.time() - start_time

    pivot_lags_wr, pivot_autocorr_wr = autocorrelation(pivot_metrics['writhe'], 50)
    bfacf_lags_wr, bfacf_autocorr_wr = autocorrelation(bfacf_metrics['writhe'], 50)

    pivot_lags_rg, pivot_autocorr_rg = autocorrelation(pivot_metrics['radius_of_gyration'], 50)
    bfacf_lags_rg, bfacf_autocorr_rg = autocorrelation(bfacf_metrics['radius_of_gyration'], 50)
    
    pivot_lags_pd, pivot_autocorr_pd = autocorrelation(pivot_metrics['positional_difference'], 50)
    bfacf_lags_pd, bfacf_autocorr_pd = autocorrelation(bfacf_metrics['positional_difference'], 50)

    # exponential decay (e^{-t/tau})
    pivot_uncorrelated_lag_wr = np.argmax(pivot_autocorr_wr < 0.1)
    bfacf_uncorrelated_lag_wr = np.argmax(bfacf_autocorr_wr < 0.1)
    pivot_uncorrelated_lag_rg = np.argmax(pivot_autocorr_rg < 0.1)
    bfacf_uncorrelated_lag_rg = np.argmax(bfacf_autocorr_rg < 0.1)
    pivot_uncorrelated_lag_pd = np.argmax(pivot_autocorr_pd < 0.1)
    bfacf_uncorrelated_lag_pd = np.argmax(bfacf_autocorr_pd < 0.1)

    pivot_time_per_sample_wr = pivot_time/pivot_uncorrelated_lag_wr
    bfacf_time_per_sample_wr = bfacf_time/bfacf_uncorrelated_lag_wr
    pivot_time_per_sample_rg = pivot_time/pivot_uncorrelated_lag_rg
    bfacf_time_per_sample_rg = bfacf_time/bfacf_uncorrelated_lag_rg
    pivot_time_per_sample_pd = pivot_time/pivot_uncorrelated_lag_pd
    bfacf_time_per_sample_pd = bfacf_time/bfacf_uncorrelated_lag_pd

    print('Writhe:')
    print(f"Pivot time: {pivot_time:.2f}s, Uncorrelated Lag: {pivot_uncorrelated_lag_wr}, Time per sample: {pivot_time_per_sample_wr:.2f}s")
    print(f"BFACF time: {bfacf_time:.2f}s, Uncorrelated Lag: {bfacf_uncorrelated_lag_wr}, Time per sample: {bfacf_time_per_sample_wr:.2f}s")

    print('Radius of Gyration:')
    print(f"Pivot time: {pivot_time:.2f}s, Uncorrelated Lag: {pivot_uncorrelated_lag_rg}, Time per sample: {pivot_time_per_sample_rg:.2f}s")
    print(f"BFACF time: {bfacf_time:.2f}s, Uncorrelated Lag: {bfacf_uncorrelated_lag_rg}, Time per sample: {bfacf_time_per_sample_rg:.2f}s")

    print('Embedding Difference:')
    print(f"Pivot time: {pivot_time:.2f}s, Uncorrelated Lag: {pivot_uncorrelated_lag_pd}, Time per sample: {pivot_time_per_sample_pd:.2f}s")
    print(f"BFACF time: {bfacf_time:.2f}s, Uncorrelated Lag: {bfacf_uncorrelated_lag_pd}, Time per sample: {bfacf_time_per_sample_pd:.2f}s")


    plot_autocorrelation(pivot_lags_wr, pivot_autocorr_wr, "Pivot")
    plot_autocorrelation(bfacf_lags_wr, bfacf_autocorr_wr, "BFACF")

    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.legend()
    plt.title("Writhe Autocorrelation Analysis")
    plt.savefig('autocorrelation/writhe.png')
    plt.clf()

    plot_autocorrelation(pivot_lags_wr, pivot_autocorr_rg, "Pivot")
    plot_autocorrelation(bfacf_lags_wr, bfacf_autocorr_rg, "BFACF")

    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.legend()
    plt.title("Radius of Gyration Autocorrelation Analysis")
    plt.savefig('autocorrelation/r_o_g.png')
    plt.clf()

    plot_autocorrelation(pivot_lags_pd, pivot_autocorr_pd, "Pivot")
    plot_autocorrelation(bfacf_lags_pd, bfacf_autocorr_pd, "BFACF")

    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.legend()
    plt.title("Embedding Difference Autocorrelation Analysis")
    plt.savefig('autocorrelation/embedding_diff.png')
    plt.clf()

    return {
        'pivot_writhe': (pivot_time, pivot_uncorrelated_lag_wr, pivot_time_per_sample_wr),
        'BFACF_writhe': (bfacf_time, bfacf_uncorrelated_lag_wr, bfacf_time_per_sample_wr)
    }

def plot_autocorrelation(lags, autocorr, label, color):
    plt.stem(lags, autocorr, label=label, c=color)


def main():

    state_space = np.zeros((discretization, discretization, discretization))
    knot = Knot(knot_type, state_space)
    unknot = knot.initialize()

    # orient knot
    print('Orienting...')
    unknot = orient(unknot)
    analyze_algorithms(unknot, knot_type, it, sampler)


par = ArgumentParser()
'''
    Lets us specify arguements for the code.
'''

par.add_argument("-d", "--discretization", type=int, default=100, help="Discretization of state space y,z axis.")
par.add_argument("-it", "--iterations", type=int, default=1000, help="Iterations of BFACF algorithm.")
par.add_argument("-k", "--knot", type=str, default='0_1', help="Knot type.")
par.add_argument("-s", "--sampler", type=str, default='Metropolis', help="Sampling method.")

args = par.parse_args()

if __name__ == "__main__":
    discretization = args.discretization
    it = args.iterations
    knot_type = args.knot
    sampler = args.sampler

    main()