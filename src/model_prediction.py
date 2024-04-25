import pandas as pd
import argparse
import torch
import json
import utils

def load_data(file_path):
    # TODO: Load test data from CSV file
    # catch the data if error and load the data from the file
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error: {e}")
        print("Loading data from file")

    return df

def load_model(model_path):
    # TODO: Load the trained model
    # load model from the file ppt from pytorch
    try:
        model = torch.load(model_path)
    except Exception as e:
        print(f"Error: {e}")
        print("Loading the model from file")
    return model

def make_predictions(df, model):

    _ = df['Idx']
    make_pred_dataset = utils.SoundDataset(df, _)
    pred_data_loader = utils.create_dataloader(make_pred_dataset, batch_size=utils.BATCH_SIZE)
    predictions = utils.predict(model, pred_data_loader, utils.DEVICE)
    df['preds'] = predictions
    predictions = df[['Idx', 'preds']].to_dict(orient='records')
    return predictions

def save_predictions(predictions, predictions_file):
    # TODO: Save predictions to a JSON file
    data = {"target": predictions}
    # Write the data to the output file as JSON
    with open(predictions_file, 'w') as f:
        json.dump(data, f, cls=NpEncoder)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Prediction script for Automated Instrument Sound Recognition Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/processed_data/train/', 
        help='Path to the test data file to make predictions'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model.pkl',
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='predictions/predictions.json', 
        help='Path to save the predictions'
    )
    return parser.parse_args()

def main(input_file, model_file, output_file):
    df = load_data(input_file)
    model = load_model(model_file)
    predictions = make_predictions(df, model)
    save_predictions(predictions, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file, args.output_file)
