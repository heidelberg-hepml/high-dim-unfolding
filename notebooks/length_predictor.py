import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from tqdm import tqdm
import energyflow as ef
import matplotlib.pyplot as plt
import random

# Constants
MAX_SEQ_LENGTH = 100
ITERATIONS_PER_EPOCH = 1000  # Fixed number of iterations per epoch
BATCH_SIZE = 64
NUM_EPOCHS = 100


def get_device():
    """Get the best available device (MPS, CUDA, or CPU)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def plot_length_distributions(
    true_lengths, pred_lengths, sim_lengths, epoch, save_path="length_dist.png"
):
    """Plot and save the length distributions"""
    plt.figure(figsize=(10, 18))

    # First subplot - Length distributions
    plt.subplot(3, 1, 1)
    bins = np.arange(0, 101, 1) - 0.5  # 0 to 100 inclusive, one bin per integer
    plt.hist(
        true_lengths,
        bins=bins,
        alpha=0.5,
        label="True Gen Lengths",
        density=True,
        color="blue",
        rwidth=0.8,
    )
    plt.hist(
        pred_lengths,
        bins=bins,
        alpha=0.5,
        label="Predicted Gen Lengths",
        density=True,
        color="red",
        rwidth=0.8,
    )
    plt.hist(
        sim_lengths,
        bins=bins,
        alpha=0.5,
        label="Sim Lengths",
        density=True,
        color="green",
        rwidth=0.8,
    )

    plt.xlabel("Number of Particles")
    plt.ylabel("Density")
    plt.title(f"Length Distribution (Epoch {epoch+1})")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 100)

    # Second subplot - Pred vs True length differences
    plt.subplot(3, 1, 2)
    length_diffs = np.array(pred_lengths) - np.array(true_lengths)
    # Create integer bins centered on each difference value
    min_diff = int(np.floor(min(length_diffs)))
    max_diff = int(np.ceil(max(length_diffs)))
    diff_bins = (
        np.arange(min_diff, max_diff + 2) - 0.5
    )  # +2 to include last value, -0.5 for bin edges

    plt.hist(
        length_diffs,
        bins=diff_bins,
        alpha=0.7,
        label="Prediction - True",
        density=True,
        color="purple",
        rwidth=0.8,
    )

    plt.xlabel("Length Difference (Predicted - True)")
    plt.ylabel("Density")
    plt.title("Distribution of Length Prediction Errors")
    plt.legend()
    plt.grid(True)

    # Third subplot - True/Pred vs Sim length differences
    plt.subplot(3, 1, 3)
    true_sim_diffs = np.array(true_lengths) - np.array(sim_lengths)
    pred_sim_diffs = np.array(pred_lengths) - np.array(sim_lengths)

    min_diff = int(np.floor(min(min(true_sim_diffs), min(pred_sim_diffs))))
    max_diff = int(np.ceil(max(max(true_sim_diffs), max(pred_sim_diffs))))
    diff_bins = np.arange(min_diff, max_diff + 2) - 0.5

    plt.hist(
        true_sim_diffs,
        bins=diff_bins,
        alpha=0.7,
        label="True - Sim",
        density=True,
        color="orange",
        rwidth=0.8,
    )
    plt.hist(
        pred_sim_diffs,
        bins=diff_bins,
        alpha=0.7,
        label="Prediction - Sim",
        density=True,
        color="cyan",
        rwidth=0.8,
    )

    plt.xlabel("Length Difference")
    plt.ylabel("Density")
    plt.title("Distribution of True/Predicted vs Sim Length Differences")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def create_epoch_sampler(dataset, iterations_per_epoch, batch_size):
    """Create a subset of indices for one epoch"""
    total_samples = iterations_per_epoch * batch_size
    indices = random.sample(range(len(dataset)), min(total_samples, len(dataset)))
    return indices


class LengthPredictor(nn.Module):
    def __init__(
        self,
        d_model=256,
        num_layers=3,
        dropout=0.1,
    ):
        super().__init__()

        # Input feature dimension (pT, eta, phi, pid)
        self.feature_dim = 4

        # Embeddings
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

        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(d_model, 1, bias=False), nn.Softmax(dim=1)
        )

        # Final prediction layers
        self.length_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1, bias=False),
        )

    def forward(self, sim_seq, sim_lengths):
        # Create attention mask for padding
        mask = (
            torch.arange(sim_seq.size(1), device=sim_seq.device)[None, :]
            < sim_lengths[:, None]
        )
        mask = mask.float().unsqueeze(-1)  # [batch, seq_len, 1]

        # Embed sequence
        x = self.particle_embedding(sim_seq)  # [batch, seq_len, d_model]

        # Apply sequence layers with residual connections
        for layer in self.sequence_layers:
            x = x + layer(x) * mask

        # Compute attention weights and apply mask
        attn_weights = self.attention(x)  # [batch, seq_len, 1]
        attn_weights = attn_weights * mask
        attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-9)

        # Weighted pooling
        x = (x * attn_weights).sum(dim=1)  # [batch, d_model]

        # Predict length
        length_pred = self.length_head(x)
        return length_pred


class ParticleDataset(Dataset):
    def __init__(
        self, sim_particles, gen_particles, max_length=MAX_SEQ_LENGTH, length_stats=None
    ):
        self.sim_particles = sim_particles
        self.gen_particles = gen_particles
        self.max_length = max_length

        # Store normalization parameters for length differences
        if length_stats is None:
            # Compute statistics from this dataset
            length_diffs = np.array(
                [len(gen) - len(sim) for gen, sim in zip(gen_particles, sim_particles)]
            )
            self.diff_mean = float(np.mean(length_diffs))
            self.diff_std = float(np.std(length_diffs))
        else:
            self.diff_mean = length_stats["diff_mean"]
            self.diff_std = length_stats["diff_std"]

    def pad_sequence(self, sequence):
        """Pad sequence with zeros"""
        sequence = np.asarray(sequence)  # Convert to numpy array first
        length = len(sequence)
        if length >= self.max_length:
            return sequence[: self.max_length]
        else:
            padding = np.zeros(
                (self.max_length - length, sequence.shape[1]), dtype=sequence.dtype
            )
            return np.vstack((sequence, padding))

    def normalize_diff(self, gen_length, sim_length):
        """Normalize length difference using stored statistics"""
        diff = gen_length - sim_length
        return (diff - self.diff_mean) / self.diff_std

    def __getitem__(self, idx):
        sim = self.sim_particles[idx]
        gen = self.gen_particles[idx]

        sim_length = len(sim)
        gen_length = len(gen)
        norm_diff = self.normalize_diff(gen_length, sim_length)

        return {
            "sim_features": torch.FloatTensor(self.pad_sequence(sim)),
            "sim_length": sim_length,
            "gen_length": gen_length,
            "norm_length_diff": norm_diff,
        }

    def __len__(self):
        return len(self.sim_particles)


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

    # Create datasets with length normalization
    train_dataset = ParticleDataset(train_sim, train_gen)
    length_stats = {
        "diff_mean": train_dataset.diff_mean,
        "diff_std": train_dataset.diff_std,
    }
    print(
        f"Length difference statistics - Mean: {length_stats['diff_mean']:.2f}, Std: {length_stats['diff_std']:.2f}"
    )

    # Use same normalization parameters for validation set
    val_dataset = ParticleDataset(val_sim, val_gen, length_stats=length_stats)

    # Initialize model and optimizer
    device = get_device()
    print(f"Using device: {device}")
    model = LengthPredictor().to(device).to(torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

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
            num_workers=0,
            pin_memory=True,
        )

        # Training phase
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            sim_features = batch["sim_features"].to(device, dtype=torch.float32)
            sim_lengths = batch["sim_length"].to(device)
            norm_target_diffs = (
                batch["norm_length_diff"].to(device, dtype=torch.float32).unsqueeze(1)
            )

            diff_pred = model(sim_features, sim_lengths)
            loss = criterion(diff_pred, norm_target_diffs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Validation phase
        model.eval()
        val_losses = []
        true_lengths_all = []
        pred_lengths_all = []
        sim_lengths_all = []

        with torch.no_grad():
            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            for batch in val_loader:
                sim_features = batch["sim_features"].to(device, dtype=torch.float32)
                sim_lengths = batch["sim_length"].to(device)
                norm_target_diffs = (
                    batch["norm_length_diff"]
                    .to(device, dtype=torch.float32)
                    .unsqueeze(1)
                )
                true_lengths = batch["gen_length"].cpu().numpy()
                sim_lengths_batch = batch["sim_length"].cpu().numpy()

                # Predict difference and unnormalize
                norm_diff_pred = model(sim_features, sim_lengths)
                val_loss = criterion(norm_diff_pred, norm_target_diffs)

                # Convert predicted difference to predicted length
                diff_pred = (
                    (
                        norm_diff_pred * length_stats["diff_std"]
                        + length_stats["diff_mean"]
                    )
                    .cpu()
                    .numpy()
                )
                length_pred = sim_lengths_batch + np.round(diff_pred.flatten())

                # Store lengths for plotting
                true_lengths_all.extend(true_lengths)
                pred_lengths_all.extend(length_pred)
                sim_lengths_all.extend(sim_lengths_batch)

                val_losses.append(val_loss.item())

        # Calculate mean validation loss
        mean_val_loss = np.mean(val_losses)

        # Save model if validation loss improves
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            print(f"New best validation loss: {best_val_loss:.4f}, saving model...")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "length_stats": {
                        "diff_mean": train_dataset.diff_mean,
                        "diff_std": train_dataset.diff_std,
                    },
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                },
                "best_length_predictor.pt",
            )

        # Plot length distributions
        plot_length_distributions(
            true_lengths_all,
            pred_lengths_all,
            sim_lengths_all,
            epoch,
            save_path=f"length_dist.png",
        )

        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss (normalized): {np.mean(train_losses):.4f}")
        print(f"Validation Loss (normalized): {mean_val_loss:.4f}")
        print(
            f"Mean Absolute Length Error: {np.mean(np.abs(np.array(true_lengths_all) - np.array(pred_lengths_all))):.2f}"
        )
        print(
            f"Median Absolute Length Error: {np.median(np.abs(np.array(true_lengths_all) - np.array(pred_lengths_all))):.2f}"
        )

    print(f"Training finished. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
