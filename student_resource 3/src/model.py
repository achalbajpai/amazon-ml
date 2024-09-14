import re
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
from src.utils import parse_string  # Assuming this is part of your project structure
import constants  # Assuming this contains allowed units


# Function to extract entities from images using the Donut model
def extract_entities(image_path, model, processor, device):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    # Set up the task-specific prompt for entity extraction
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    # Generate output using the model
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=512,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # Decode the generated sequence and clean up unwanted tokens
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(
        processor.tokenizer.pad_token, ""
    )
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

    # Post-process the extracted sequence
    return post_process(sequence)


# Function to post-process the extracted sequence
def post_process(sequence):
    try:
        # Parse the sequence into number and unit
        number, unit = parse_string(sequence)
        # Check if the unit is allowed
        if unit in constants.allowed_units:
            return f"{number} {unit}"
    except ValueError:
        # Return an empty string if parsing fails or unit is not allowed
        pass
    return ""


# Function to set up the model, processor, and device
def setup_model():
    processor = DonutProcessor.from_pretrained(
        "naver-clova-ix/donut-base-finetuned-cord-v2"
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        "naver-clova-ix/donut-base-finetuned-cord-v2"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, processor, device


# Example usage
if __name__ == "__main__":
    # Set up the model, processor, and device
    model, processor, device = setup_model()

    # Example image path
    image_path = "path_to_your_image.jpg"

    # Extract entities from the image and post-process
    extracted_entities = extract_entities(image_path, model, processor, device)

    # Print the extracted and post-processed entities
    print("Extracted Entities:", extracted_entities)
