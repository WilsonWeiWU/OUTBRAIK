from typing import Any, Dict

import pandas as pd





def separate_data(data: pd.DataFrame, test_data_ratio: float) -> Dict[str, Any]:
    """
    Node that separates the dataset into features and labels and into training
    and test set. The ratio of test and training set is taken from the
    parameters file.
    """
    
    print("\n###########################################")
    print(data.columns)
    
    print("###########################################\n")
    
    
    data.columns = [
            "name",
            "region",
            "deaths",
            "infected",
            "length",
            "pop_density",
            "pop",
            "gdp",
            "infection_rate",
            "temp",
            "humidity",
            "mortality_rate",
            "target",
    ]
    
    data.drop(["region","deaths"], axis=1)
    
    # We want the new coronavirus to be on the test set
    corona = data.iloc[:,0]
    data = data.iloc[:,1:]
    
    # Shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)

    # Split training and testing data
    n = data.shape[0]
    n_test = int(n * test_data_ratio)
    training_data = data.iloc[n_test:, :].reset_index(drop=True)
    test_data = data.iloc[:n_test, :].reset_index(drop=True)
    test_data.append(corona)
    
    # Split data into names, features and targets
    #train_data_x = training_data.loc[:, "infected":"mortality_rate"]
    #train_data_y = training_data["target"]
    #test_data_x = test_data.loc[:, "infected":"mortality_rate"]
    #test_data_y = test_data["target"]

    # When returning many variables, it is a good practice to give them names:
    return dict(
        train=training_data,
        test=test_data,
    )
    