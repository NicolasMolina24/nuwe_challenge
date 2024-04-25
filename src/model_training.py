from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import argparse
import utils

def load_data(file_path):
    # TODO: Load processed data from CSV file
    # catch the data if error and load the data from the file
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error: {e}")
        print("Loading data from file")

    return df

def split_data(df):
    # TODO: Split data into training and validation sets (the test set is already provided in data/test_data.csv)
    # split the df into training and validation sets

    # Assuming the last column of df is the target variable
    # select the columns except the label 'Last column'
    y = df['Label']
    X = df.drop('Label', axis=1)

    X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=utils.TEST_SIZE, 
            random_state=utils.RANDOM_STATE, stratify=df["Label"]
        )

    return X_train, X_val, y_train, y_val

def train_model(X_train, X_val, y_train, y_val):
    # TODO: Initialize your model and train it
    train_dataset = utils.SoundDataset(X_train, y_train)
    val_dataset = utils.SoundDataset(X_val, y_val)
    train_loader = utils.create_dataloader(train_dataset, batch_size=utils.BATCH_SIZE, shuffle=True)
    val_loader = utils.create_dataloader(val_dataset, batch_size=utils.BATCH_SIZE)
    # create the model
    model = utils.CNNModel(num_classes=train_dataset.get_classes())
    model.to(utils.DEVICE)
    # define the loss function and the optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    epochs = utils.EPOCHS
    device = utils.DEVICE
    model = utils.train_model(model, train_loader, val_loader, loss_fn, optimizer, device, epochs)
    return model

def save_model(model, model_path):
    # TODO: Save your trained model
    torch.save(model, model_path)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model training script for Automated Instrument Sound Recognition Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/processed_data/', 
        help='Path to the processed data file to train the model'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model.pkl', 
        help='Path to save the trained model'
    )
    return parser.parse_args()

def main(input_file, model_file):
    df = load_data(input_file)
    X_train, X_val, y_train, y_val = split_data(df)
    model = train_model(X_train, X_val, y_train, y_val)
    save_model(model, model_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file)