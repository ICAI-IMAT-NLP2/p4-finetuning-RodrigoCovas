import pytest
from src.utils import download_and_load_model

@pytest.mark.order(1)
def test_download_and_load_model():
    model, tokenizer, device = download_and_load_model()
    assert model is not None, "Model loading failed"
    assert tokenizer is not None, "Tokenizer loading failed"
