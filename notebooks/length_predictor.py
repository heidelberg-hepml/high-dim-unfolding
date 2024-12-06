import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from tqdm import tqdm
import energyflow as ef
import matplotlib.pyplot as plt
import random
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from itertools import cycle
import gc
from tqdm.auto import tqdm

# Constants
MAX_SEQ_LENGTH = 116
ITERATIONS_PER_EPOCH = 1000  # Number of iterations per epoch
BATCH_SIZE = 256  # Smaller batch size for better generalization
NUM_EPOCHS = 100
NUM_WORKERS = 8
SEED = 42  # Fixed random seed
LEARNING_RATE = 3e-4  # Standard transformer learning rate
WEIGHT_DECAY = 0.01  # L2 regularization


def set_seeds(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


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
    """Create indices for one epoch by cycling through the dataset"""
    total_samples = iterations_per_epoch * batch_size
    
    # Create a list of all indices
    all_indices = list(range(len(dataset)))
    
    # Calculate how many complete cycles we need
    num_complete_cycles = total_samples // len(dataset)
    remaining_samples = total_samples % len(dataset)
    
    # Create the final list of indices
    indices = all_indices * num_complete_cycles
    if remaining_samples > 0:
        indices.extend(all_indices[:remaining_samples])
    
    return indices


class LengthPredictor(nn.Module):
    def __init__(
        self,
        d_model=256,  # Reduced size for faster training
        num_layers=3,
        dropout=0.1,  # Reduced dropout
    ):
        super().__init__()
        
        self.feature_dim = 4
        # Use Layer Normalization instead of Batch Normalization
        self.input_norm = nn.LayerNorm(self.feature_dim)
        
        # Simpler but effective embedding
        self.particle_embedding = nn.Sequential(
            nn.Linear(self.feature_dim, d_model),
            nn.GELU(),  # Changed to GELU
            nn.LayerNorm(d_model),
        )
        
        # Transformer-style processing
        self.sequence_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        # Global context
        self.global_context = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        
        # Prediction head with multiple branches
        self.length_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1)
            )
            for _ in range(3)  # Multiple prediction heads
        ])

    def forward(self, sim_seq, sim_lengths):
        # Create attention mask
        mask = torch.arange(sim_seq.size(1), device=sim_seq.device)[None, :] < sim_lengths[:, None]
        mask = mask.float().unsqueeze(-1)
        
        # Process input features
        x = self.input_norm(sim_seq)
        x = self.particle_embedding(x)
        
        # Apply sequence layers with residual connections and masked mean pooling
        sequence_features = []
        for layer in self.sequence_layers:
            # Compute masked mean for global context
            mean_pool = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            global_context = self.global_context(mean_pool).unsqueeze(1)
            
            # Apply layer with residual connection
            x = x + layer(x + global_context) * mask
            sequence_features.append(x)
        
        # Combine features from different layers
        final_features = torch.stack([
            (feat * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            for feat in sequence_features
        ], dim=-1).mean(dim=-1)
        
        # Multiple prediction heads
        predictions = torch.stack([head(final_features) for head in self.length_head], dim=-1)
        
        # Average predictions
        return predictions.mean(dim=-1)


class ParticleDataset(Dataset):
    def __init__(
        self, sim_particles, sim_lengths, gen_lengths, max_length=MAX_SEQ_LENGTH, length_stats=None
    ):
        self.sim_particles = sim_particles
        self.sim_lengths = sim_lengths
        self.gen_lengths = gen_lengths
        self.max_length = max_length

        # Store normalization parameters for length differences
        if length_stats is None:
            # Compute statistics from this dataset
            length_diffs = np.array(
                [gen - sim for gen, sim in zip(self.gen_lengths, self.sim_lengths)]
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
        sim_length = self.sim_lengths[idx]
        gen_length = self.gen_lengths[idx]
        norm_diff = self.normalize_diff(gen_length, sim_length)

        return {
            "sim_features": torch.FloatTensor(self.pad_sequence(self.sim_particles[idx])),
            "sim_length": sim_length,
            "gen_length": gen_length,
            "norm_length_diff": norm_diff,
        }

    def __len__(self):
        return len(self.sim_particles)


def custom_loss(pred, target, sim_lengths):
    # MSE loss
    mse_loss = F.mse_loss(pred, target)
    
    # Add L1 loss for better stability
    l1_loss = F.l1_loss(pred, target)
    
    # Variance loss to prevent constant predictions
    batch_variance = torch.var(pred)
    variance_loss = torch.exp(-batch_variance)  # Penalize low variance
    
    # Combine losses
    total_loss = mse_loss + 0.5 * l1_loss + 0.1 * variance_loss
    
    return total_loss


def main():
    # Set random seeds for reproducibility
    set_seeds(SEED)
    
    # Load data
    print("Loading data...")
    data = ef.zjets_delphes.load(
        "Herwig",
        num_data=1000000,
        pad=True,
        cache_dir="../data",
        source="zenodo",
        which="all",
    )
    print(f"Dataset shapes - Sim: {data['sim_particles'].shape}, Gen: {data['gen_particles'].shape}")
    
    # Split data
    print("Preparing datasets...")
    num_events = len(data["gen_particles"])
    train_size = int(0.8 * num_events)
    indices = np.random.permutation(num_events)

    # Convert lists to numpy arrays for better memory efficiency
    train_sim = data["sim_particles"][indices[:train_size]]
    train_gen_lengths = data["gen_mults"][indices[:train_size]]
    train_sim_lengths = data["sim_mults"][indices[:train_size]]
    val_sim = data["sim_particles"][indices[train_size:]]
    val_gen_lengths = data["gen_mults"][indices[train_size:]]
    val_sim_lengths = data["sim_mults"][indices[train_size:]]
    
    # Clear original data from memory
    del data
    gc.collect()

    print(f"Training samples: {len(train_sim)}, Validation samples: {len(val_sim)}")
    print(f"Iterations per epoch: {ITERATIONS_PER_EPOCH}, Batch size: {BATCH_SIZE}")
    print(f"Samples per epoch: {ITERATIONS_PER_EPOCH * BATCH_SIZE}")

    # Create datasets with length normalization
    train_dataset = ParticleDataset(train_sim, train_sim_lengths, train_gen_lengths)
    length_stats = {
        "diff_mean": train_dataset.diff_mean,
        "diff_std": train_dataset.diff_std,
    }
    print(
        f"Length difference statistics - Mean: {length_stats['diff_mean']:.2f}, Std: {length_stats['diff_std']:.2f}"
    )

    # Use same normalization parameters for validation set
    val_dataset = ParticleDataset(val_sim, val_sim_lengths, val_gen_lengths, length_stats=length_stats)

    # Initialize model and optimizer
    device = get_device()
    print(f"Using device: {device}")
    model = LengthPredictor().to(device).to(torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Use OneCycleLR scheduler instead
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=NUM_EPOCHS,
        steps_per_epoch=ITERATIONS_PER_EPOCH,
        pct_start=0.1,  # Warm-up for 10% of training
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    # Initialize mixed precision training
    scaler = amp.GradScaler(growth_interval=100)
    
    criterion = custom_loss  # Replace nn.MSELoss()

    # Training loop
    print("Starting training...")
    best_val_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        # Clear memory before each epoch
        print(f"Epoch {epoch+1}")
        gc.collect()
        torch.cuda.empty_cache()

        # Create new subset for this epoch
        train_indices = create_epoch_sampler(
            train_dataset, ITERATIONS_PER_EPOCH, BATCH_SIZE
        )
        epoch_dataset = Subset(train_dataset, train_indices)

        train_loader = DataLoader(
            epoch_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )

        # Training phase
        model.train()
        train_losses = []
        
        # Use gradient accumulation for effective larger batch size
        ACCUMULATION_STEPS = 2
        optimizer.zero_grad(set_to_none=True)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]', 
                   leave=True, position=0)
        for i, batch in enumerate(pbar):
            sim_features = batch["sim_features"].to(device, dtype=torch.float32, non_blocking=True)
            sim_lengths = batch["sim_length"].to(device, non_blocking=True)
            norm_target_diffs = batch["norm_length_diff"].to(device, dtype=torch.float32, non_blocking=True).unsqueeze(1)

            with amp.autocast():
                diff_pred = model(sim_features, sim_lengths)
                loss = criterion(diff_pred, norm_target_diffs, sim_lengths)
                loss = loss / ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{np.mean(train_losses[-100:]):.4f}'})
            
            # Clear memory for batch variables
            del sim_features, sim_lengths, norm_target_diffs, diff_pred, loss

        # Clear memory after training phase
        gc.collect()
        torch.cuda.empty_cache()

        # Validation phase
        model.eval()
        val_losses = []
        true_lengths_all = []
        pred_lengths_all = []
        sim_lengths_all = []
        running_val_loss = 0.0
        num_val_samples = 0

        with torch.no_grad():
            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True
            )

            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Val]', 
                       leave=True, position=0)
            for batch in pbar:
                sim_features = batch["sim_features"].to(device, dtype=torch.float32, non_blocking=True)
                sim_lengths = batch["sim_length"].to(device, non_blocking=True)
                norm_target_diffs = batch["norm_length_diff"].to(device, dtype=torch.float32, non_blocking=True).unsqueeze(1)
                true_lengths = batch["gen_length"].cpu().numpy()
                sim_lengths_batch = batch["sim_length"].cpu().numpy()

                with amp.autocast():
                    norm_diff_pred = model(sim_features, sim_lengths)
                    val_loss = criterion(norm_diff_pred, norm_target_diffs, sim_lengths)
                    
                # Accumulate loss properly
                batch_size = sim_features.size(0)
                running_val_loss += val_loss.item() * batch_size
                num_val_samples += batch_size

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

                # Show current running average
                current_val_loss = running_val_loss / num_val_samples
                pbar.set_postfix({'val_loss': f'{current_val_loss:.4f}'})
                
                # Clear memory for batch variables
                del sim_features, sim_lengths, norm_target_diffs, norm_diff_pred, val_loss

        # Calculate proper mean validation loss
        mean_val_loss = running_val_loss / num_val_samples

        # Also update training loss computation for consistency
        mean_train_loss = np.mean([loss * ACCUMULATION_STEPS for loss in train_losses])  # Undo accumulation scaling

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
        print(f"Train Loss (normalized): {mean_train_loss:.4f}")
        print(f"Validation Loss (normalized): {mean_val_loss:.4f}")
        print(
            f"Mean Absolute Length Error: {np.mean(np.abs(np.array(true_lengths_all) - np.array(pred_lengths_all))):.2f}"
        )
        print(
            f"Median Absolute Length Error: {np.median(np.abs(np.array(true_lengths_all) - np.array(pred_lengths_all))):.2f}"
        )

        del true_lengths_all, pred_lengths_all, sim_lengths_all

        scheduler.step(mean_val_loss)

    print(f"Training finished. Best validation loss: {best_val_loss:.4f}")
    
    del model, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
