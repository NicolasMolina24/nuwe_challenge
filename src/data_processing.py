import argparse
import pandas as pd
from pathlib import Path


def load_data(file_path):
    # TODO: Load data from CSV file
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error: {e}")
        print("Loading data from file")

    return df

def clean_data(df):
    # TODO: Handle missing values, outliers, etc.
    path = Path(__file__).parent
    # delete nan values
    df = df.dropna()
    # apply the function to check if the path exists
    valid_df = df.apply(lambda x: path.joinpath('..', x['Path']).exists(), axis=1)
    df_clean = df[valid_df]
    # resolve the path

    return df_clean

def preprocess_data(df):
    # TODO: Generate new features, transform existing features, resampling, etc.
    # this funtion is implemented in the dataset object
    return df

def save_data(df, output_file):
    # TODO: Save processed data to a CSV file
    try:
        df.to_csv(output_file, header=True, index=False)
    except Exception as e:
        print(f"Error: {e}")
        print("Saving data to file")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data processing script for Automated Instrument Sound Recognition Hackathon')
    parser.add_argument(
        '--input_file',
        type=str,
        default='data/labels_paths_train.csv',
        help='Path to the raw data file to process'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='data/processed_data/', 
        help='Folder path to save the processed data'
    )
    return parser.parse_args()

def main(input_file, output_file):
    df = load_data(input_file)
    df_clean = clean_data(df)
    df_processed = preprocess_data(df_clean)
    save_data(df_processed, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.output_file)
