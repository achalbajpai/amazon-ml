import pandas as pd
from tqdm import tqdm
from model import setup_model, extract_entities, post_process
from prepare_data import test_dataset
import torch  # Import torch to load model weights


# Function to generate predictions from the test dataset
def generate_predictions():
    # Load the model, processor, and device
    model, processor, device = setup_model()

    # Load the fine-tuned model weights
    model.load_state_dict(
        torch.load("./fine_tuned_model/pytorch_model.bin", map_location=device)
    )

    predictions = []
    # Iterate through the test dataset (assuming it's a list of tuples: (image_path, label))
    for img_path, _ in tqdm(test_dataset):
        # Extract entities from the image
        sequence = extract_entities(img_path, model, processor, device)
        # Post-process the sequence to extract the number and unit
        prediction = post_process(sequence)
        # Append the processed prediction to the list
        predictions.append(prediction)

    return predictions


# Function to create the output file in the required format
def create_output_file(predictions):
    # Load the test CSV file
    test_df = pd.read_csv("dataset/test.csv")

    # Create a DataFrame with 'index' from the CSV and 'prediction' from the model
    output_df = pd.DataFrame(
        {
            "index": test_df[
                "index"
            ],  # Ensure you're using the 'index' column from the CSV
            "prediction": predictions,
        }
    )

    # Save the DataFrame to a CSV file in the required format
    output_df.to_csv("test_out.csv", index=False)


if __name__ == "__main__":
    # Generate predictions for the test dataset
    predictions = generate_predictions()

    # Create the output CSV file with predictions
    create_output_file(predictions)
