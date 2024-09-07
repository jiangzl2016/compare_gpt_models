# The CNN / DailyMail Dataset is an English-language dataset containing just over 300k unique news articles as written by journalists at CNN and the Daily Mail. 
# Link: https://huggingface.co/datasets/abisee/cnn_dailymail

import pandas as pd

def cnn_news_loader(path):
    """
    Load CNN News Dataset from the given path. The file is in parquet format.
    """
    
    df = pd.read_parquet(path)
    return df


def sample_cnn_news(df, n, seed):
    """
    Sample n news articles from the given dataframe.
    """

    if n > len(df):
        raise ValueError("Sample size is greater than the size of the dataset")
    
    return df.sample(n, random_state=seed, replace=False)


def imdb_review_loader(path):
    """
    Load imdb movie review from the given path. The file is in csv format.
    """
    
    df = pd.read_csv(path)
    classes = list(df['sentiment'].unique())
    return df, classes


def sample_movie_review(df, n, seed):
    """
    Sample n movie reviews from the given dataframe.
    """

    if n > len(df):
        raise ValueError("Sample size is greater than the size of the dataset")
    
    return df.sample(n, random_state=seed, replace=False)