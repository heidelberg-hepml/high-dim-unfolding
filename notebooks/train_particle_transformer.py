import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from tqdm import tqdm
import energyflow as ef

# Constants
MAX_SEQ_LENGTH = 100
NUM_EVENTS = 1000  # Small number for testing
BATCH_SIZE = 32
NUM_EPOCHS = 10  # Just enough epochs to check functionality


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0)]


class ParticleTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_length=MAX_SEQ_LENGTH,
    ):
        super().__init__()

        # Regular feature dimension (pT, eta, phi, pid) + stop token
        self.feature_dim = 4
        self.stop_token = torch.zeros(self.feature_dim)
        self.stop_token[-1] = -1.0  # Special value for stop token

        # Embeddings and transformer layers
        self.particle_embedding = nn.Linear(self.feature_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        self.final_head = nn.Linear(d_model, self.feature_dim)

    def forward(self, sim_seq, gen_seq, tgt_mask):
        sim_embedded = self.particle_embedding(sim_seq)
        gen_embedded = self.particle_embedding(gen_seq)

        sim_embedded = self.pos_encoding(sim_embedded)
        gen_embedded = self.pos_encoding(gen_embedded)

        decoded = self.transformer_decoder(
            gen_embedded.transpose(0, 1),
            sim_embedded.transpose(0, 1),
            tgt_mask=tgt_mask,
        ).transpose(0, 1)

        return self.final_head(decoded)

    @torch.no_grad()
    def generate(self, sim_seq, max_length=None):
        """Generate sequence autoregressively until stop token or max length"""
        if max_length is None:
            max_length = MAX_SEQ_LENGTH

        batch_size = sim_seq.size(0)
        device = sim_seq.device

        # Start with empty sequence
        current_seq = torch.zeros(batch_size, 1, self.feature_dim).to(device)

        # Generate particles one at a time
        for _ in range(max_length - 1):
            tgt_mask = generate_square_subsequent_mask(current_seq.size(1)).to(device)

            # Get next particle prediction
            sim_embedded = self.particle_embedding(sim_seq)
            current_embedded = self.particle_embedding(current_seq)

            sim_embedded = self.pos_encoding(sim_embedded)
            current_embedded = self.pos_encoding(current_embedded)

            decoded = self.transformer_decoder(
                current_embedded.transpose(0, 1),
                sim_embedded.transpose(0, 1),
                tgt_mask=tgt_mask,
            ).transpose(0, 1)

            next_particle = self.final_head(decoded[:, -1:, :])

            # Check if stop token is predicted
            if (next_particle[:, :, -1] < -0.5).any():
                break

            current_seq = torch.cat([current_seq, next_particle], dim=1)

        # Add stop token
        stop_token = self.stop_token.to(device).view(1, 1, -1).repeat(batch_size, 1, 1)
        current_seq = torch.cat([current_seq, stop_token], dim=1)

        return current_seq


class ParticleDataset(Dataset):
    def __init__(self, sim_particles, gen_particles, max_length=MAX_SEQ_LENGTH):
        self.sim_particles = sim_particles
        self.gen_particles = gen_particles
        self.max_length = max_length
        self.stop_token = torch.zeros(4)
        self.stop_token[-1] = -1.0

    def pad_sequence(self, sequence):
        """Pad sequence with zeros after adding stop token"""
        length = len(sequence)
        if length >= self.max_length:
            # Truncate and add stop token
            padded = np.vstack(
                (sequence[: self.max_length - 1], self.stop_token.numpy())
            )
        else:
            # Add stop token and pad
            padded = np.vstack((sequence, self.stop_token.numpy()))
            padding = np.zeros((self.max_length - length - 1, 4))
            padded = np.vstack((padded, padding))
        return padded

    def __getitem__(self, idx):
        sim = self.sim_particles[idx]
        gen = self.gen_particles[idx]

        return {
            "sim_features": torch.FloatTensor(self.pad_sequence(sim)),
            "gen_features": torch.FloatTensor(self.pad_sequence(gen)),
            "gen_length": min(len(gen) + 1, self.max_length),  # +1 for stop token
        }

    def __len__(self):
        return len(self.sim_particles)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def compute_loss(pred, target, target_lengths):
    """Compute MSE loss between predicted and target sequences up to target length"""
    batch_size = pred.size(0)
    total_loss = 0

    for i in range(batch_size):
        target_length = target_lengths[i].item()

        # Get sequences up to target length
        pred_seq = pred[i, :target_length]
        target_seq = target[i, :target_length]

        # Compute MSE loss on the sequence
        seq_loss = F.mse_loss(pred_seq, target_seq, reduction="mean")

        # Add penalty for predictions beyond target length
        if pred.size(1) > target_length:
            extra_pred = pred[i, target_length:]
            extra_loss = torch.mean(extra_pred.pow(2))
            seq_loss = seq_loss + 0.1 * extra_loss  # Reduced weight for length penalty

        total_loss += seq_loss

    return total_loss / batch_size


def main():
    # Load data
    print("Loading data...")
    data = ef.zjets_delphes.load(
        "Herwig",
        num_data=NUM_EVENTS,
        pad=False,
        cache_dir="data",
        source="zenodo",
        which="all",
    )

    # Split data
    print("Preparing datasets...")
    train_size = int(0.8 * NUM_EVENTS)
    indices = np.random.permutation(NUM_EVENTS)

    train_gen = [data["gen_particles"][i] for i in indices[:train_size]]
    train_sim = [data["sim_particles"][i] for i in indices[:train_size]]
    val_gen = [data["gen_particles"][i] for i in indices[train_size:]]
    val_sim = [data["sim_particles"][i] for i in indices[train_size:]]

    # Create dataloaders
    train_dataset = ParticleDataset(train_sim, train_gen)
    val_dataset = ParticleDataset(val_sim, val_gen)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ParticleTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            sim_features = batch["sim_features"].to(device)
            gen_features = batch["gen_features"].to(device)
            gen_lengths = batch["gen_length"].to(device)

            tgt_mask = generate_square_subsequent_mask(gen_features.size(1)).to(device)
            pred = model(sim_features, gen_features, tgt_mask)

            loss = compute_loss(pred, gen_features, gen_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Validation and generation test
        model.eval()
        with torch.no_grad():
            # Test on a single batch
            batch = next(iter(val_loader))
            sim_features = batch["sim_features"].to(device)
            gen_features = batch["gen_features"].to(device)
            gen_lengths = batch["gen_length"].to(device)

            # Generate sequences
            generated_seqs = model.generate(sim_features)

            # Compare lengths
            gen_lengths = gen_lengths.cpu().numpy()
            pred_lengths = [
                (
                    (seq[:, -1] < -0.5).nonzero()[0].item()
                    if len((seq[:, -1] < -0.5).nonzero()) > 0
                    else len(seq)
                )
                for seq in generated_seqs
            ]

            avg_target_len = gen_lengths.mean()
            avg_pred_len = np.mean(pred_lengths)

            # Compute generation loss
            gen_loss = compute_loss(generated_seqs, gen_features, gen_lengths)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {np.mean(train_losses):.4f}")
        print(f"Generation Loss: {gen_loss.item():.4f}")
        print(f"Average Target Length: {avg_target_len:.1f}")
        print(f"Average Generated Length: {avg_pred_len:.1f}")


if __name__ == "__main__":
    main()
