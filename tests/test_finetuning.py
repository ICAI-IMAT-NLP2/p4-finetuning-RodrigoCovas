import pytest
import torch
import torch.nn as nn
import platform
from src.utils import download_and_load_model
from src.finetuning import LoRA, inject_lora_into_model, SoftPromptEmbedding


def set_seed(seed):
    """
    Set the seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@pytest.mark.order(2)
def test_lora_initialization():
    original_layer = nn.Linear(10, 10)  # Create a simple linear layer
    r = 4
    alpha = 32

    # Create the LoRA layer 
    lora_layer = LoRA(original_layer, r=r, alpha=alpha)

    # Check that A and B have the correct dimensions
    assert lora_layer.A.shape == (original_layer.weight.size(0), r), "Matrix A is not the correct size."
    assert lora_layer.B.shape == (r, original_layer.weight.size(1)), "Matrix B is not the correct size."

    # Check that the scaling factor is computed correctly
    expected_scaling = alpha / r
    assert lora_layer.scaling == expected_scaling, f"Scaling factor is incorrect. Expected {expected_scaling}, got {lora_layer.scaling}."

    # Check that matrix B is initialized to zeros
    assert torch.equal(lora_layer.B, torch.zeros_like(lora_layer.B)), "Matrix B is not initialized to zeros."

    # Check that matrix A is not zero (kaiming uniform initialization)
    assert not torch.equal(lora_layer.A, torch.zeros_like(lora_layer.A)), "Matrix A is incorrectly initialized to zeros."

    # Check if the original layer's parameters are frozen (i.e., requires_grad is False)
    for param in lora_layer.original_layer.parameters():
        assert not param.requires_grad, "Original layer parameters are not frozen."

@pytest.mark.order(3)
def test_lora_forward_pass():

    # Set seed for reproducibility
    set_seed(42)
    x = torch.randn(2, 10)  # Batch size 2, input dimension 10

    original_layer = nn.Linear(10, 10)  # Create a simple linear layer
    r = 4
    alpha = 32

    # Create the LoRA layer
    lora_layer = LoRA(original_layer, r=r, alpha=alpha)

    # Let's change the B matrix to avoid zeros
    lora_layer.B = nn.Parameter(torch.ones(lora_layer.B.size()))

    # Perform the forward pass
    lora_output = lora_layer(x)
    original_output = original_layer(x)

    # Check that the output shape matches the original layer output
    assert lora_output.shape == original_output.shape, "LoRA output shape does not match original layer output shape."

    # Check that the output is different from the original output (LoRA applied)
    assert not torch.equal(lora_output, original_output), "LoRA output should differ from the original layer output."

    if "Microsoft" in platform.uname().release or platform.system() == "Windows":	
        expected_output = torch.tensor([[-7.2452, -7.3211, -7.5850, -8.1368, -8.8998, -8.4971, -7.1514, -9.7109, -8.8360, -8.3201],
                                        [-2.1295, -2.3751, -1.9628, -2.1550, -1.5780, -2.5529, -1.4401, -2.0061, -2.0690, -1.8637]])
    elif platform.system() == "Darwin" or platform.system() == "Linux":
        expected_output = torch.tensor([[-7.2452, -7.3211, -7.5850, -8.1368, -8.8998, -8.4971, -7.1514, -9.7109, -8.8360, -8.3201],
                                        [-2.1295, -2.3751, -1.9628, -2.1550, -1.5780, -2.5529, -1.4401, -2.0061, -2.0690, -1.8637]])

    assert torch.allclose(lora_output, expected_output, atol=1e-4), "LoRA output doesn't match the expected output."


@pytest.mark.order(4)
def test_inject_lora():
    """
    Test the inject_lora_into_model function.
    """
    def count_lora_layers(model):
        """
        Counts the number of LoRA layers present in the model.
        
        Args:
            model (torch.nn.Module): The model in which to count LoRA layers.
        
        Returns:
            int: The number of LoRA layers found.
        """
        lora_count = 0
        for module in model.modules():
            if isinstance(module, LoRA):
                lora_count += 1
        return lora_count

    # Load the model
    model_name = "t5-small"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Download model
    model, tokenizer, device = download_and_load_model(model_name, device)

    # Check that the model is returned correctly
    assert model is not None, "Model has not been downloaded correctly."
    assert tokenizer is not None, "Tokenizer has not been downloaded correctly."

    # Inject LoRA into the model's attention layers
    model = inject_lora_into_model(model, device=device)
    
    # Check that the model is returned correctly
    assert model is not None, "LoRA fine-tuning did not return a model."
    
    # Count how many LoRA layers were injected
    num_lora_layers = count_lora_layers(model)
    
    # Check that LoRA layers were successfully added
    assert num_lora_layers > 0, "No LoRA layers were injected into the model."
    assert num_lora_layers == 72, "Not all linear attention layers were substituted by LoRA layers"

@pytest.mark.order(5)
def test_soft_prompt_initialization():
    """
    Test that the soft prompt embeddings are initialized with the correct dimensions.
    """
    prompt_length = 10
    model_hidden_size = 512  # Example hidden size (T5-small uses 512)

    # Initialize the soft prompt embeddings
    soft_prompt = SoftPromptEmbedding(prompt_length, model_hidden_size)

    # Check the dimensions of the soft prompt
    assert soft_prompt.soft_prompt.shape == (prompt_length, model_hidden_size), \
        f"Soft prompt shape mismatch. Expected {(prompt_length, model_hidden_size)}, but got {soft_prompt.soft_prompt.shape}"

    # Check that the soft prompt is not initialized to all zeros
    assert not torch.equal(soft_prompt.soft_prompt, torch.zeros_like(soft_prompt.soft_prompt)), \
        "Soft prompt embeddings are incorrectly initialized to zeros."

    print("Soft prompt initialization test passed.")

@pytest.mark.order(6)
def test_soft_prompt_forward_pass():
    """
    Test that the soft prompt embeddings are correctly prepended to the input embeddings.
    """
    # Load the model
    model_name = "t5-small"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Download model
    model, tokenizer, device = download_and_load_model(model_name, device)
    
    prompt_length = 10
    model_hidden_size = model.config.d_model  # Should be 512 for T5-small

    # Initialize the soft prompt embeddings
    soft_prompt = SoftPromptEmbedding(prompt_length, model_hidden_size).to(device)

    # Simulate token embeddings from input
    input_text = "Translate English to French: The house is beautiful."
    inputs = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    input_embeddings = model.encoder.embed_tokens(inputs)

    # Perform forward pass with soft prompt embeddings
    combined_embeddings = soft_prompt(input_embeddings)

    # Check that the combined embeddings have the correct shape
    assert combined_embeddings.shape[1] == input_embeddings.shape[1] + prompt_length, \
        f"Combined embedding shape mismatch. Expected {input_embeddings.shape[1] + prompt_length}, but got {combined_embeddings.shape[1]}."


@pytest.mark.order(7)
def test_soft_prompt_backpropagation():
    """
    Test that gradients are backpropagated through the soft prompt embeddings.
    """
    # Load the model
    model_name = "t5-small"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Download model
    model, tokenizer, device = download_and_load_model(model_name, device)
    
    prompt_length = 10
    model_hidden_size = model.config.d_model  # Should be 512 for T5-small

    # Initialize the soft prompt embeddings
    soft_prompt = SoftPromptEmbedding(prompt_length, model_hidden_size).to(device)
    soft_prompt.train()

    # Simulate token embeddings from input
    input_text = "Translate English to French: I love programming."
    inputs = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    input_embeddings = model.encoder.embed_tokens(inputs)

    # Forward pass with soft prompt embeddings
    combined_embeddings = soft_prompt(input_embeddings)

    # Simulate a forward pass through the model
    labels = tokenizer("J'adore programmer.", return_tensors="pt").input_ids.to(device)
    outputs = model(inputs_embeds=combined_embeddings, labels=labels)
    loss = outputs.loss

    # Backward pass to compute gradients
    loss.backward()

    # Check that gradients exist for soft prompt parameters
    assert soft_prompt.soft_prompt.grad is not None, "Gradients were not backpropagated through the soft prompts."
    assert torch.any(soft_prompt.soft_prompt.grad != 0), "Gradients for soft prompt parameters are all zero."

