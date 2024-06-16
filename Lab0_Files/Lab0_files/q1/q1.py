import json
import numpy as np
import matplotlib.pyplot as plt

def inv_cauchy_cdf(u,peak_x, gamma):
    val = gamma*np.tan(np.pi * (u - 0.5)) + peak_x
    return val

def inv_exp_cdf(u,lamb):
    val = - np.log(1 - u)/lamb
    return val

def inv_transform(distribution: str, num_samples: int, **kwargs) -> list:
    """ populate the 'samples' list from the desired distribution """

    samples = []

    # TODO: first generate random numbers from the uniform distribution
    rng = np.random.default_rng()
    uni_samples = [rng.random() for _ in range(num_samples)]
    
    if distribution == "cauchy":
        samples = [round(inv_cauchy_cdf(u,kwargs["peak_x"],kwargs["gamma"]),4) for u in uni_samples]
    else:
        samples = [round(inv_exp_cdf(u,kwargs["lambda"]),4) for u in uni_samples]
    # END TODO
            
    return samples


if __name__ == "__main__":
    np.random.seed(42)

    for distribution in ["cauchy", "exponential"]:
        file_name = "q1_" + distribution + ".json"
        args = json.load(open(file_name, "r"))
        samples = inv_transform(**args)
        
        with open("q1_output_" + distribution + ".json", "w") as file:
            json.dump(samples, file)

        # TODO: plot and save the histogram to "q1_" + distribution + ".png"
        if distribution == 'cauchy':
            plt.hist(samples)
            plt.savefig(f'q1_{distribution}.png')
            plt.close()
        else:
            plt.hist(samples)
            plt.savefig(f'q1_{distribution}.png')
            plt.close()
        # END TODO
