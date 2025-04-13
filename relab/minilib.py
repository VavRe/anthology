import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance



def remove_non_compliant_responses(df: pd.DataFrame, question_of_interest: list) -> pd.DataFrame:
    """Remove language model's non-compliant responses from the dataframe

    Args:
        df: DataFrame

    Returns:
        DataFrame

    """
    new_df = df[question_of_interest]
    condition = new_df.apply(lambda x: x < 0, axis=1)
    df[condition] = np.nan

    condition = new_df.apply(
        lambda x: x > 90, axis=1
    )  # refused/non_compliant answer for ATP survey
    df[condition] = np.nan

    return df



def get_human_response_distribution(
    df: pd.DataFrame, weight_data: pd.DataFrame, renormalize: bool = True
) -> tuple[np.array, np.array]:
    """Calculate the weighted distribution of the human responses

    Args:
        df (pd.DataFrame): human data
        weight_data (pd.DataFrame): weight data
        renormalize (bool, optional): renormalize the distribution. Defaults to True.

    Returns:
        tuple[np.array, np.array]: response_ratio_arr, response_std_err
    """
    data = df
    n = weight_data.sum()

    # the user response is in the form of a number
    # response_num = data.value_counts(dropna=False).sort_index().index.astype(int).tolist() # no ignoring nan
    response_num = data.value_counts().sort_index().index.astype(int).tolist()
    response_ratio_list = []
    response_std_err_list = []

    for num in response_num:
        nominal_index = df[data == num].index

        response_ratio = weight_data[nominal_index].sum() / n
        std_err = np.sqrt(response_ratio * (1 - response_ratio) / n)

        response_ratio_list.append(response_ratio)
        response_std_err_list.append(std_err)

    if renormalize:
        total = sum(response_ratio_list)

        response_ratio_list = [ratio / total for ratio in response_ratio_list]
        response_std_err_list = [
            std_err / np.sqrt(total) for std_err in response_std_err_list
        ]

    response_ratio_arr = np.array(response_ratio_list)
    response_std_err = np.array(response_std_err_list)

    return response_ratio_arr, response_std_err


def get_human_response(
    df: pd.DataFrame,
    question_of_interest: list[str],
    weight_data: pd.DataFrame,
) -> tuple[dict[str, np.array], dict[str, np.array]]:
    """Get the weighted human response distribution for each question

    Args:
        df (pd.DataFrame): human data
        question_of_interest (list[str]): list of questions to calculate the response for
        weight_data (pd.DataFrame): weight data

    Returns:
        tuple[dict[str, np.array], dict[str, np.array]]: avg_response_dict, std_err_dict
    """
    avg_response_dict, std_err_dict = {}, {}

    for question in question_of_interest:
        response_dist, response_std_err = get_human_response_distribution(
            df[question], weight_data
        )

        avg_response_dict[question] = response_dist
        std_err_dict[question] = response_std_err

    return avg_response_dict, std_err_dict




def get_weighted_cov(
    X: np.array, w: np.array, corr: bool = False
) -> tuple[np.array, np.array]:
    """Calculate the weighted covariance matrix

    Args:
        X (np.array): data
        w (np.array): weights
        corr (bool, optional): calculate the correlation matrix. Defaults to False.

    Returns:
        mean (np.array): mean of the data
        cov (np.array): covariance matrix of the data (or correlation matrix if corr=True)
    """
    # normalize the weights
    w = w / w.sum()

    # nan values in X will cause the covariance to be nan
    # remove nan values from X and w
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    w = w[mask]

    # calculate the weighted mean
    mean = np.average(X, weights=w, axis=0)

    cov = np.cov(X, aweights=w, rowvar=False)

    if corr:
        v = np.sqrt(np.diag(cov))
        outer_v = np.outer(v, v)
        cov = cov / outer_v

    return mean, cov



def cov_matrix_distance(cov1: np.array, cov2: np.array, dist_type="CBS") -> float:
    """Calculate the distance between two covariance matrices

    Args:
        cov1 (np.array): covariance matrix 1
        cov2 (np.array): covariance matrix 2
        loss_type (str, optional): loss type. Defaults to "CBS". Possible values: CBS, Frobenius, L1.

    Returns:
        float: distance between the two covariance matrices
    """
    if dist_type == "CBS":
        return 1 - np.trace(cov1 @ cov2) / (np.linalg.norm(cov1) * np.linalg.norm(cov2))
    elif dist_type == "Frobenius":
        return np.sqrt(((cov1 - cov2) ** 2).sum())
    elif dist_type == "L1":
        return np.abs(cov1 - cov2).sum()
    
    


def distance_lower_bound(
    human_data: pd.DataFrame,
    question_of_interest: list[str],
    num_iter: int = 20,
    weight_column: str = "WEIGHT_W34",
    dist_type: str = "EMD",
) -> dict[str, float]:
    """Calculate the lower bound of the distance between two distributions
    Divide the human data into two parts randomly, calculate the distance between the two distributions for each question, and estimate the average distance

    Args:
        human_data (pd.DataFrame): human data
        question_of_interest (list[str]): list of questions to calculate the distance for
        num_iter (int, optional): number of samples to estimate the average distance. Defaults to 20.
        weight_column (str, optional): weight column. Defaults to "WEIGHT_W34".
        dist_type (str, optional): distance type. Defaults to "EMD". Possible values: EMD, TV.

    Returns:
        dist_dict_avg (dict): dictionary containing the average distance between the two distributions for each question
    """
    dist_list = []

    for _ in range(num_iter):
        # randomly divide human_data equally
        # Shuffle the DataFrame and reset the index
        df_shuffled = human_data.sample(frac=1).reset_index(drop=True)

        # Calculate the split index
        split_index = len(df_shuffled) // 2

        # Split the DataFrame into two parts
        human_data_part1 = df_shuffled.iloc[:split_index]
        human_data_part2 = df_shuffled.iloc[split_index:]

        part1_dist_dict = get_human_response(
            human_data_part1,
            question_of_interest,
            human_data_part1[weight_column],
        )[0]

        part2_dist_dict = get_human_response(
            human_data_part2,
            question_of_interest,
            human_data_part2[weight_column],
        )[0]

        dist_dict = {}
        for q in question_of_interest:
            dist1 = part1_dist_dict[q]
            dist2 = part2_dist_dict[q]

            distance = get_statistical_distance(dist1, dist2, dist_type)

            dist_dict[q] = distance

        dist_list.append(dist_dict)

    dist_df = pd.DataFrame(dist_list)

    dist_dict_avg = dist_df.mean(axis=0).to_dict()
    dist_dict_avg[f"average {dist_type}"] = dist_df.mean().mean()

    return dist_dict_avg


def get_statistical_distance(data1: np.array, data2: np.array, dist_type: str = "EMD"):
    """Calculate the statistical distance between two distributions

    Args:
        data1 (np.array): distribution 1
        data2 (np.array): distribution 2
        dist_type (str, optional): distance type. Defaults to earth mover's distance (EMD). Possible values: earth mover's distance (EMD), total variation (TV).

    Raises:
        NotImplementedError: Invalid distance type, choose from earth mover's distance (EMD) or total variation (TV)

    Returns:
        float: statistical distance between the two distributions
    """
    if dist_type == "EMD":
        dist1 = np.arange(1, len(data1) + 1)
        dist2 = np.arange(1, len(data2) + 1)

        return wasserstein_distance(dist1, dist2, data1, data2)

    elif dist_type == "TV":
        return 0.5 * np.abs(data1 - data2).sum()

    else:
        raise NotImplementedError



def matrix_dist_lower_bound(
    human_data: pd.DataFrame,
    question_of_interest: list[str],
    wave: int = 34,
    num_iter: int = 20,
    dist_type: str = "CBS",
) -> float:
    """Calculate the lower bound of the distance between two weighted covariance matrices
    Divide the human data into two parts randomly, calculate the weighted covariance matrix for each part, and calculate the distance between the two covariance matrices
    Repeat the process num_iter times and estimate the average distance between the two covariance matrices

    Args:
        human_data (pd.DataFrame): human data
        question_of_interest (list[str]): list of questions to calculate the covariance matrix for
        num_iter (int, optional): number of samples to estimate the average distance. Defaults to 20.
        dist_type (str, optional): distance type. Defaults to "CBS". Possible values: CBS, Frobenius, L1.

    Returns:
        float: average distance between the two covariance matrices
    """
    dist_list = []

    for _ in range(num_iter):
        # divide human data into two parts randomly
        # Shuffle the DataFrame and reset the index
        df_shuffled = human_data.sample(frac=1).reset_index(drop=True)

        # Calculate the split index
        split_index = len(df_shuffled) // 2

        # Split the DataFrame into two parts
        human_data_part1 = df_shuffled.iloc[:split_index]
        human_data_part2 = df_shuffled.iloc[split_index:]

        human_data_part1_X = human_data_part1[question_of_interest].values
        human_data_part1_w = human_data_part1[f"WEIGHT_W{wave}"].values

        human_data_part2_X = human_data_part2[question_of_interest].values
        human_data_part2_w = human_data_part2[f"WEIGHT_W{wave}"].values

        _, human_data_part1_weighted_cov = get_weighted_cov(
            human_data_part1_X, human_data_part1_w, corr=True
        )
        _, human_data_part2_weighted_cov = get_weighted_cov(
            human_data_part2_X, human_data_part2_w, corr=True
        )

        # calculate the distance between the two weighted covariance matrices
        distance = cov_matrix_distance(
            human_data_part1_weighted_cov, human_data_part2_weighted_cov, dist_type
        )

        dist_list.append(distance)

    return np.mean(dist_list)
