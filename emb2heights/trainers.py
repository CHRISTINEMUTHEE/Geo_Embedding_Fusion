"""
Plain PyTorch training loop for embedding -> (landcover, height) models.

Entry point: train(config) — builds loaders/model/loss/optimizer from an
ExperimentConfig, trains with validation each epoch, saves best/last weights,
a loss curve, and a config snapshot under outputs/<experiment_name>/.
"""
import random

import matplotlib
matplotlib.use("Agg")  # save figures without a display
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


from emb2heights.datasets import build_dataloaders
from emb2heights.losses import build_loss
from emb2heights.models import build_model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_optimizer(config, model):
    opts = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
    }
    if config.optimizer not in opts:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    kwargs = dict(lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == "sgd":
        kwargs["momentum"] = 0.9
    return opts[config.optimizer](model.parameters(), **kwargs)


def build_scheduler(config, optimizer):
    if config.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=config.patience, factor=config.factor)
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=1e-6)
    if config.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.step_size, gamma=config.gamma)
    if config.scheduler in ("none", None):
        return None
    raise ValueError(f"Unsupported scheduler: {config.scheduler}")


# ---------------------------------------------------------
# Evaluation (challenge-style metrics)
# ---------------------------------------------------------
def binary_iou_from_channel(pred, target, threshold=0.1, eps=1e-6):
    """pred, target: [B, H, W] fraction maps; IoU of the thresholded masks."""
    pred_mask = pred > threshold
    target_mask = target > threshold
    intersection = (pred_mask & target_mask).sum().float()
    union = (pred_mask | target_mask).sum().float()
    if union == 0:
        return torch.tensor(float("nan"), device=pred.device)
    return (intersection + eps) / (union + eps)


def masked_rmse(pred_height, true_height, mask):
    """RMSE in meters over pixels where mask is True."""
    if mask.sum() == 0:
        return torch.tensor(float("nan"), device=pred_height.device)
    return torch.sqrt(torch.mean((pred_height[mask] - true_height[mask]) ** 2))


# REVIEW REQUIRED
def evaluate_metrics(model, val_loader, device, height_norm, threshold=0.1):
    model.eval()
    scores = {k: [] for k in ["iou_building", "iou_vegetation", "iou_water",
                              "rmse_building", "rmse_vegetation"]}
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)

            pred = torch.clamp(outputs[:, :3], 0, 1)
            true = torch.clamp(targets[:, :3], 0, 1)
            ## channel 3 is height, stored normalized -> back to meters
            pred_height = outputs[:, 3] * height_norm
            true_height = targets[:, 3] * height_norm

            scores["iou_building"].append(binary_iou_from_channel(pred[:, 0], true[:, 0], threshold))
            scores["iou_vegetation"].append(binary_iou_from_channel(pred[:, 1], true[:, 1], threshold))
            scores["iou_water"].append(binary_iou_from_channel(pred[:, 2], true[:, 2], threshold))
            scores["rmse_building"].append(masked_rmse(pred_height, true_height, true[:, 0] > threshold))
            scores["rmse_vegetation"].append(masked_rmse(pred_height, true_height, true[:, 1] > threshold))

    return {k: torch.nanmean(torch.stack(v)).item() for k, v in scores.items()}


def visualize_results(model, dataset, config, device, num_samples=10):
    """Side-by-side true vs predicted maps for a few random samples."""
    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    target_names = ["% Building", "% Vegetation", "% Water", "nDSM Height (m)"]
    h_norm = config.height_normalization_constant

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_tensor, target_tensor = dataset[idx]
            output = model(img_tensor.unsqueeze(0).to(device))
            pred = output.squeeze(0).cpu().numpy()
            true = target_tensor.numpy()

            pred[3] *= h_norm
            true[3] *= h_norm

            fig, axs = plt.subplots(2, 4, figsize=(20, 10))
            for c in range(4):
                vmin, vmax = (0, 1) if c < 3 else (0, h_norm)
                axs[0, c].imshow(true[c], vmin=vmin, vmax=vmax)
                axs[0, c].set_title(f"True {target_names[c]}")
                axs[0, c].axis("off")
                axs[1, c].imshow(pred[c], vmin=vmin, vmax=vmax)
                axs[1, c].set_title(f"Predicted {target_names[c]}")
                axs[1, c].axis("off")
            plt.suptitle(f"Sample {i + 1} - True vs Predicted")
            plt.savefig(config.viz_output_dir / f"visualization_{i}.png")
            plt.close(fig)


def plot_height_rmse_vs_labels(labels_seen, rmse_building, rmse_vegetation, path):
    """Val height RMSE (m) vs cumulative training labels (tiles) seen so far."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(labels_seen, rmse_building, marker="o", label="Building height RMSE")
    ax.plot(labels_seen, rmse_vegetation, marker="s", label="Vegetation height RMSE")
    ax.set_xlabel("Training labels seen (tiles)")
    ax.set_ylabel("Height RMSE (m) — lower is better")
    ax.set_title("Height accuracy vs training labels")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path)
    plt.close(fig)


def _run_epoch(model, loader, criterion, device, optimizer=None, desc=""):
    """One pass over loader. Trains if optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, samples_seen = 0.0, 0

    progress = tqdm(loader, desc=desc, leave=False)
    with torch.enable_grad() if training else torch.no_grad():
        for images, targets in progress:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            samples_seen += images.size(0)
            progress.set_postfix(loss=f"{total_loss / samples_seen:.4f}")

    return total_loss / max(1, samples_seen)


# REVIEW REQUIRED
## Main entry point: config -> trained model + artifacts in outputs/<experiment_name>/
def train(config):
    seed_everything(config.random_seed)
    device = get_device()
    config.make_dirs()
    config.save()

    train_loader, val_loader = build_dataloaders(config)

    ## Infer input channels from the data (AlphaEarth=64, Tessera=128, ...)
    sample_img, _ = train_loader.dataset[0]
    n_channels = sample_img.shape[0]
    model = build_model(config, n_channels).to(device)
    print(f"Device: {device} | Model: {config.model_name} "
          f"({n_channels} in-channels, {config.n_classes} out-channels)")

    criterion = build_loss(config)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)

    train_losses, val_losses = [], []
    labels_seen, height_rmse_building, height_rmse_vegetation = [], [], []
    best_val_loss = float("inf")
    n_train = len(train_loader.dataset)

    for epoch in range(1, config.epochs + 1):
        desc = f"Epoch {epoch}/{config.epochs}"
        train_loss = _run_epoch(model, train_loader, criterion, device, optimizer, desc + " [train]")
        val_loss = _run_epoch(model, val_loader, criterion, device, desc=desc + " [val]")
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        print(f"{desc} - train MAE: {train_loss:.4f} - val MAE: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.best_model_path)
            print(f"  New best (val {val_loss:.4f}) -> {config.best_model_path}")

        metrics = evaluate_metrics(model, val_loader, device,
                                   config.height_normalization_constant)
        labels_seen.append(epoch * n_train)
        height_rmse_building.append(metrics["rmse_building"])
        height_rmse_vegetation.append(metrics["rmse_vegetation"])
        print(f"  Height RMSE (m): building={metrics['rmse_building']:.3f}, "
              f"vegetation={metrics['rmse_vegetation']:.3f}")

        if epoch % 10 == 0:
            print("  Challenge-style evaluation:")
            for name, value in metrics.items():
                print(f"    {name}: {value:.4f}")

    torch.save(model.state_dict(), config.last_model_path)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.savefig(config.loss_curve_path)
    plt.close()

    plot_height_rmse_vs_labels(labels_seen, height_rmse_building, height_rmse_vegetation,
                               config.height_curve_path)

    visualize_results(model, val_loader.dataset, config, device)
    print(f"Artifacts saved under {config.experiment_dir}")
    return model, {"train_losses": train_losses, "val_losses": val_losses,
                   "best_val_loss": best_val_loss,
                   "labels_seen": labels_seen,
                   "height_rmse_building": height_rmse_building,
                   "height_rmse_vegetation": height_rmse_vegetation}
