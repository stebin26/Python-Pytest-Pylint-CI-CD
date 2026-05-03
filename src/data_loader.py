""" Importing Pandas """
import pandas as pd


def load_data(path):
    """Load dataset from CSV file"""
    return pd.read_csv(path)

