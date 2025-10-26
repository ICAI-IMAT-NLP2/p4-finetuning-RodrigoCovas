import torch
import torch.nn as nn
import math

try:
    from utils import download_and_load_model
except:
    from src.utils import download_and_load_model

class LoRA(nn.Module):
    def __init__(self, original_layer, r=4, alpha=32):
        """
        Low-Rank Adaptation (LoRA) module.
        
        Args:
            original_layer (nn.Module): The original layer to which LoRA is applied.
            r (int): Rank of the low-rank approximation.
            alpha (int): Scaling factor for the LoRA module.
        """
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.original_layer = original_layer
        
        w = self.original_layer.weight
        device, dtype = w.device, w.dtype
        in_features = self.original_layer.in_features
        out_features = self.original_layer.out_features

        self.A = nn.Parameter(torch.empty((in_features, r), device=device, dtype=dtype))
        self.B = nn.Parameter(torch.zeros((r, out_features), device=device, dtype=dtype))

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        
        self.scaling = self.alpha / r

        for param in self.original_layer.parameters():
            param.requires_grad = False
                
    def forward(self, x):
        out = self.original_layer(x)
        lora = x @ self.A @ self.B * self.scaling
        return out + lora


def inject_lora_into_model(model, r=4, alpha=32, device='cpu'):
    """
    Inject LoRA layers into the linear layers of the attention modules of the model.
    
    Args:
        model (PreTrainedModel): The pre-trained model.
        r (int): Rank of the low-rank approximation.
        alpha (int): Scaling factor for LoRA.
        device (torch.device): The device to run the model on ('cuda' or 'cpu').
    
    Returns:
        model (PreTrainedModel): The model with LoRA injected into attention layers.
    """
    for child_name, child_module in model.named_children():
        print(child_name)
        if child_name.lower() in ['q', 'k', 'v', 'o']:
            lora_layer = LoRA(child_module, r=r, alpha=alpha)
            setattr(model, child_name, lora_layer)
        else:
            inject_lora_into_model(child_module, r=r, alpha=alpha, device=device)
    return model.to(device)


class SoftPromptEmbedding(nn.Module):
    def __init__(self, prompt_length, model_hidden_size):
        """
        Creates trainable soft prompts to prepend to input embeddings.

        Args:
            prompt_length (int): Number of virtual tokens in the soft prompt.
            model_hidden_size (int): The hidden size of the pre-trained model.
        """
        super().__init__()
        self.soft_prompt = nn.Parameter(torch.randn(prompt_length, model_hidden_size))

    def forward(self, input_embeddings):
        """
        Forward pass to prepend soft prompts to input embeddings.

        Args:
            input_embeddings (torch.Tensor): The original input embeddings from the tokenizer.

        Returns:
            torch.Tensor: The concatenated soft prompts and original embeddings.
        """
        batch_size = input_embeddings.size(0)
        soft_prompt_expanded = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)

        return torch.cat([soft_prompt_expanded, input_embeddings], dim=1)