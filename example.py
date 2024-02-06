import torch
from audio_flamingo.model import AudioFlamingo

# Generate a random input sequence
text = torch.randint(0, 256, (1, 1024))
audio = torch.randn(1, 16000)

# Initialize GPT-3 model
model = AudioFlamingo(
    dim=512,
    num_tokens=256,
    max_seq_len=1024,
    heads=8,
    depth=6,
    dim_head=64,
    dropout=0.1,
    context_dim=512,
)

# Pass the input sequence through the model
output = model(text, audio)  # (1, 1024, 256)

# Print the output shape
print(output.shape)
# Path: audio_flamingo/model.py
