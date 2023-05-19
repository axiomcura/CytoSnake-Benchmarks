import argparse
import numpy as np
import pandas as pd



def main() -> None:
    """
    Generates random dataset with given features, rows.
    """
    parser = argparse.ArgumentParser(description="Generates fake data")
    parser.add_argument("-f", "--n_features", default=1000, type=int, help="number of features")
    parser.add_argument("-r", "--n_rows", default=2000, type=int, help="number of rows")
    parser.add_argument("-o", "--output_dir", default="../data", help="output directory")
    parser.add_argument("-s", "--random_seed", default=42, help="random seed")
    args = parser.parse_args()

    # generate features
    np.random.seed(args.seed)

    # creating test feature names
    features = [f"feature_{n+1}" for n in range(args.n_features)]

    # creating matrices (normal and with noise)
    matrix = np.random.rand(args.n_rows, args.n_features)
    noise = np.random.normal(loc=0, scale=0.1, size=(args, 2000))
    noised_matrix = matrix + noise

    # convert matrix to df
    matrix = pd.DataFrame(matrix, columns=features)
    noised_matrix = pd.DataFrame(noised_matrix, columns=features)

    # save matrix
    matrix.to_csv(f"{args.output_dir}/test_df.csv.gz", index=False, compression="gzip")
    noised_matrix.to_csv(f"{args.output_dir}/noised_test_df.csv.gz", index=False, compression="gzip")

if __name__ == "__main__":
    main()






