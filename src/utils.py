import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

def download_and_load_model(model_name="t5-small", device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Downloads and loads the T5-small model and tokenizer. 
    The model is moved to the appropriate device.
    
    Args:
        model_name (str): The name of the pre-trained model to load.
        device (str): The device to load the model on, 'cuda' or 'cpu'.
    
    Returns:
        model (PreTrainedModel): The loaded T5-small model.
        tokenizer (PreTrainedTokenizer): The corresponding tokenizer.
    """
    print(f"Downloading and loading the {model_name} model...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    
    return model, tokenizer, device
