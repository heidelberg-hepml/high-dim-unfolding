import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import math
from tqdm import tqdm
import energyflow as ef
from length_predictor import LengthPredictor, get_device
import random
import matplotlib.pyplot as plt

# Constants
MAX_SEQ_LENGTH = 100
ITERATIONS_PER_EPOCH = 1000  # Fixed number of iterations per epoch
BATCH_SIZE = 64
NUM_EPOCHS = 100


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
        num_layers=3,
        dropout=0.1,
        max_seq_length=MAX_SEQ_LENGTH,
        length_predictor_path="best_length_predictor.pt",
    ):
        super().__init__()

        # Load length predictor
        self.length_predictor = LengthPredictor()
        checkpoint = torch.load(length_predictor_path, map_location="cpu")
        self.length_predictor.load_state_dict(checkpoint["model_state_dict"])
        self.length_stats = checkpoint["length_stats"]

        # Verify required keys are present
        if "diff_mean" not in self.length_stats or "diff_std" not in self.length_stats:
            raise KeyError(
                f"Length stats must contain 'diff_mean' and 'diff_std'. Found keys: {self.length_stats.keys()}"
            )

        print(f"Loaded length stats: {self.length_stats}")
        self.length_predictor.eval()

        # Model parameters
        self.feature_dim = 4

        # Network layers (similar to LengthPredictor)
        self.particle_embedding = nn.Linear(self.feature_dim, d_model, bias=False)

        # Sequence processing layers
        self.sequence_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model, bias=False),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )

        # Attention mechanism
        self.self_attention = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.ReLU(),
            nn.Linear(d_model, 1, bias=False),
            nn.Softmax(dim=1),
        )

        # Output head
        self.final_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, self.feature_dim, bias=False),
        )

    def forward(self, sim_seq, gen_seq, tgt_mask=None):
        batch_size = sim_seq.size(0)

        # Embed sequences
        sim_embedded = self.particle_embedding(sim_seq)
        gen_embedded = self.particle_embedding(gen_seq)

        # Process simulator sequence
        x = sim_embedded
        for layer in self.sequence_layers:
            x = x + layer(x)

        # Apply attention
        attn_weights = self.self_attention(x)
        sim_context = (x * attn_weights).sum(dim=1, keepdim=True)

        # Process generator sequence with simulator context
        x = gen_embedded
        for layer in self.sequence_layers:
            x = x + layer(x) + sim_context

        # Generate output
        return self.final_head(x)

    @torch.no_grad()
    def predict_sequence_length(self, sim_seq, sim_lengths):
        """Predict the length of the sequence to generate"""
        # Get normalized length difference prediction
        norm_diff_pred = self.length_predictor(sim_seq, sim_lengths)

        # Unnormalize the prediction
        diff_pred = (
            norm_diff_pred * self.length_stats["diff_std"]
            + self.length_stats["diff_mean"]
        )

        # Add to sim length and round to nearest integer
        pred_lengths = sim_lengths + torch.round(diff_pred.flatten())

        # Ensure lengths are within reasonable bounds
        pred_lengths = torch.clamp(pred_lengths, 1, MAX_SEQ_LENGTH - 1)

        return pred_lengths.long()

    @torch.no_grad()
    def generate(self, sim_seq):
        batch_size = sim_seq.size(0)
        device = sim_seq.device

        # Get sequence lengths
        sim_lengths = torch.sum(torch.abs(sim_seq).sum(dim=-1) > 0, dim=1)
        target_lengths = self.predict_sequence_length(sim_seq, sim_lengths)

        # Start with empty sequence
        current_seq = torch.zeros(
            batch_size, 1, self.feature_dim, device=device, dtype=torch.float32
        )

        # Generate particles one at a time
        for _ in range(MAX_SEQ_LENGTH - 1):
            pred = self(sim_seq, current_seq)
            next_particle = pred[:, -1:, :]

            # Check if we've reached target lengths
            reached_target = torch.arange(
                current_seq.size(1), device=device
            ) >= target_lengths.unsqueeze(1)
            if reached_target.all():
                break

            current_seq = torch.cat([current_seq, next_particle], dim=1)

        # Trim and pad sequences
        final_seqs = []
        for i in range(batch_size):
            length = target_lengths[i]
            seq = current_seq[i, :length]
            final_seqs.append(seq)

        # Pad sequences to max length in batch
        max_len = max(len(seq) for seq in final_seqs)
        padded_seqs = []
        for seq in final_seqs:
            if len(seq) < max_len:
                padding = torch.zeros(
                    (max_len - len(seq), self.feature_dim),
                    device=device,
                    dtype=torch.float32,
                )
                seq = torch.cat([seq, padding])
            padded_seqs.append(seq)

        return torch.stack(padded_seqs)


class ParticleDataset(Dataset):
    def __init__(self, sim_particles, gen_particles, max_length=MAX_SEQ_LENGTH):
        self.sim_particles = sim_particles
        self.gen_particles = gen_particles
        self.max_length = max_length

    def pad_sequence(self, sequence):
        """Pad sequence with zeros"""
        sequence = np.asarray(sequence)
        length = len(sequence)
        if length >= self.max_length:
            return sequence[: self.max_length]
        else:
            padding = np.zeros(
                (self.max_length - length, sequence.shape[1]), dtype=sequence.dtype
            )
            return np.vstack((sequence, padding))

    def __getitem__(self, idx):
        sim = self.sim_particles[idx]
        gen = self.gen_particles[idx]

        return {
            "sim_features": torch.FloatTensor(self.pad_sequence(sim)),
            "gen_features": torch.FloatTensor(self.pad_sequence(gen)),
            "gen_length": len(gen),
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
    """Compute MSE loss between predicted and target sequences"""
    batch_size = pred.size(0)
    total_loss = 0

    # Find maximum length between predicted and target sequences
    max_len = max(pred.size(1), target.size(1))

    # Pad both sequences to max_len
    if pred.size(1) < max_len:
        padding = torch.zeros(
            (batch_size, max_len - pred.size(1), pred.size(2)),
            device=pred.device,
            dtype=pred.dtype,
        )
        pred = torch.cat([pred, padding], dim=1)

    if target.size(1) < max_len:
        padding = torch.zeros(
            (batch_size, max_len - target.size(1), target.size(2)),
            device=target.device,
            dtype=target.dtype,
        )
        target = torch.cat([target, padding], dim=1)

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


def create_epoch_sampler(dataset, iterations_per_epoch, batch_size):
    """Create a subset of indices for one epoch"""
    total_samples = iterations_per_epoch * batch_size
    indices = random.sample(range(len(dataset)), min(total_samples, len(dataset)))
    return indices


def plot_feature_distributions(
    true_features, pred_features, epoch, save_path="feature_dist.png"
):
    """Plot distributions of each feature"""
    feature_names = ["pT", "eta", "phi", "pid"]

    plt.figure(figsize=(15, 12))

    for i, name in enumerate(feature_names):
        plt.subplot(2, 2, i + 1)

        # Flatten arrays for histograms
        true_feat = true_features[:, :, i].flatten()
        pred_feat = pred_features[:, :, i].flatten()

        # Remove padding (zeros)
        true_feat = true_feat[true_feat != 0]
        pred_feat = pred_feat[pred_feat != 0]

        # Plot histograms
        plt.hist(
            true_feat,
            bins=50,
            alpha=0.5,
            label="True",
            density=True,
            color="blue",
        )
        plt.hist(
            pred_feat,
            bins=50,
            alpha=0.5,
            label="Predicted",
            density=True,
            color="red",
        )

        plt.xlabel(name)
        plt.ylabel("Density")
        plt.title(f"{name} Distribution")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    # Load data
    print("Loading data...")
    data = ef.zjets_delphes.load(
        "Herwig",
        pad=False,
        cache_dir="../data",
        source="zenodo",
        which="all",
    )

    # Split data
    print("Preparing datasets...")
    num_events = len(data["gen_particles"])
    train_size = int(0.8 * num_events)
    indices = np.random.permutation(num_events)

    train_gen = [data["gen_particles"][i] for i in indices[:train_size]]
    train_sim = [data["sim_particles"][i] for i in indices[:train_size]]
    val_gen = [data["gen_particles"][i] for i in indices[train_size:]]
    val_sim = [data["sim_particles"][i] for i in indices[train_size:]]

    # Create datasets
    train_dataset = ParticleDataset(train_sim, train_gen)
    val_dataset = ParticleDataset(val_sim, val_gen)

    # Initialize model and optimizer
    device = get_device()
    print(f"Using device: {device}")
    model = ParticleTransformer().to(device).to(torch.float32)
    model.length_predictor = model.length_predictor.to(device).to(torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    print("Starting training...")
    best_val_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        # Create new subset for this epoch
        train_indices = create_epoch_sampler(
            train_dataset, ITERATIONS_PER_EPOCH, BATCH_SIZE
        )
        epoch_dataset = Subset(train_dataset, train_indices)

        train_loader = DataLoader(
            epoch_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues with MPS
            pin_memory=True,  # Faster data transfer to GPU
        )

        # Training phase
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            sim_features = batch["sim_features"].to(device, dtype=torch.float32)
            gen_features = batch["gen_features"].to(device, dtype=torch.float32)
            gen_lengths = batch["gen_length"].to(device)

            tgt_mask = generate_square_subsequent_mask(gen_features.size(1)).to(device)
            pred = model(sim_features, gen_features, tgt_mask)

            loss = compute_loss(pred, gen_features, gen_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Validation phase
        model.eval()
        val_losses = []
        all_true_features = []
        all_pred_features = []

        with torch.no_grad():
            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            # First pass to find maximum sequence length
            max_seq_len = 0
            for batch in val_loader:
                gen_features = batch["gen_features"]
                max_seq_len = max(max_seq_len, gen_features.size(1))

            # Second pass for evaluation and plotting
            for batch in val_loader:
                sim_features = batch["sim_features"].to(device, dtype=torch.float32)
                gen_features = batch["gen_features"].to(device, dtype=torch.float32)
                gen_lengths = batch["gen_length"].to(device)

                # Generate sequences
                generated_seqs = model.generate(sim_features)

                # Pad sequences to max_seq_len if needed
                if generated_seqs.size(1) < max_seq_len:
                    padding = torch.zeros(
                        (
                            generated_seqs.size(0),
                            max_seq_len - generated_seqs.size(1),
                            generated_seqs.size(2),
                        ),
                        device=generated_seqs.device,
                        dtype=generated_seqs.dtype,
                    )
                    generated_seqs = torch.cat([generated_seqs, padding], dim=1)

                if gen_features.size(1) < max_seq_len:
                    padding = torch.zeros(
                        (
                            gen_features.size(0),
                            max_seq_len - gen_features.size(1),
                            gen_features.size(2),
                        ),
                        device=gen_features.device,
                        dtype=gen_features.dtype,
                    )
                    gen_features = torch.cat([gen_features, padding], dim=1)

                # Compute reconstruction loss
                val_loss = compute_loss(generated_seqs, gen_features, gen_lengths)
                val_losses.append(val_loss.item())

                # Store features for plotting
                all_true_features.append(gen_features.cpu().numpy())
                all_pred_features.append(generated_seqs.cpu().numpy())

        # Calculate mean validation loss
        mean_val_loss = np.mean(val_losses)

        # Concatenate all features
        all_true_features = np.concatenate(all_true_features, axis=0)
        all_pred_features = np.concatenate(all_pred_features, axis=0)

        # Plot feature distributions
        plot_feature_distributions(
            all_true_features,
            all_pred_features,
            epoch,
            save_path=f"feature_dist.png",
        )

        # Save model if validation loss improves
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            print(f"New best validation loss: {best_val_loss:.4f}, saving model...")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                },
                "best_particle_transformer.pt",
            )

        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {np.mean(train_losses):.4f}")
        print(f"Validation Loss: {mean_val_loss:.4f}")

    print(f"Training finished. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
