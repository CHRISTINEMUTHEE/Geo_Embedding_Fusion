"""
Plain PyTorch training loop for embedding -> (landcover, height) models.

Entry point: train(config) — builds loaders/model/loss/optimizer from an
ExperimentConfig, trains with validation each epoch, saves best/last weights,
a loss curve, and a config snapshot under outputs/<experiment_name>/.
"""
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures without a display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from tqdm import tqdm


from emb2heights.datamodule import Embed2HeightsDataModule, HEIGHT_BIN_EDGES, HEIGHT_BIN_NAMES
from emb2heights.datasets import build_dataloaders
from emb2heights.losses import build_loss
from emb2heights.models import build_model


## data_root set -> region-grouped split, dequantized embeddings (datamodule.py).
## Otherwise -> legacy random tile split (datasets.py), kept for 01/02 baselines.
def build_train_val_loaders(config):
    if config.data_root:
        dm = Embed2HeightsDataModule(
            root=config.data_root,
            source=config.embedding_source,
            patch_size=config.patch_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            val_frac=config.val_split,
            seed=config.random_seed,
            height_norm=config.height_normalization_constant,
            max_train_tiles=config.max_train_tiles,
            stratify_threshold=config.stratify_threshold,
            class_balance_boost=config.class_balance_boost,
            standardize_bands=config.standardize_bands,
        ).setup()
        print(f"Region-grouped split -> {len(dm.train_regions)} train regions "
              f"({len(dm.train_ds)} tiles) / {len(dm.val_regions)} val regions "
              f"({len(dm.val_ds)} tiles)")
        for split in ("train", "val"):
            stats = dm.class_distribution[split]
            summary = "  ".join(f"{cls}: mean={s['mean_frac']:.3f} "
                                 f"present={s['pct_tiles_present']:.0f}%"
                                 for cls, s in stats.items())
            heights = "  ".join(f"{name}={pct:.1f}%" for name, pct in dm.height_distribution[split].items())
            print(f"  {split} class distribution -> {summary}")
            print(f"  {split} height distribution -> {heights}")
        distributions = {"class": dm.class_distribution, "height_bins": dm.height_distribution}
        return dm.train_dataloader(), dm.val_dataloader(), distributions
    train_loader, val_loader = build_dataloaders(config)
    return train_loader, val_loader, None


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
def _mask_counts(pred, target, threshold):
    """Raw (intersection, union) pixel counts for pred/target masks thresholded at
    `threshold`. Split out from binary_iou_from_channel so evaluate_metrics can
    accumulate counts across an entire validation set and divide once, instead of
    computing a ratio per batch and averaging ratios (see evaluate_metrics docstring
    for why that's wrong)."""
    pred_mask = pred > threshold
    target_mask = target > threshold
    intersection = (pred_mask & target_mask).sum()
    union = (pred_mask | target_mask).sum()
    return intersection, union


def binary_iou_from_channel(pred, target, threshold=0.1, eps=1e-6):
    """pred, target: [B, H, W] fraction maps; single-call IoU of the thresholded masks
    -- pools everything passed into it into one ratio. Fine for a one-off comparison
    of two tensors. NOT what evaluate_metrics uses for the validation-set-wide IoU:
    calling this once per batch and averaging the per-batch ratios is batch-size-
    sensitive (verified: same val tiles, same predictions, batch_size 4 vs 23 alone
    swung the result from 0.28 to 0.36 -- a ~26% relative range) because each batch's
    ratio gets equal weight regardless of how many positive pixels it actually
    contained. evaluate_metrics instead accumulates raw counts via _mask_counts()
    across the whole loader and divides once."""
    intersection, union = _mask_counts(pred, target, threshold)
    if union == 0:
        return torch.tensor(float("nan"), device=pred.device)
    return (intersection.float() + eps) / (union.float() + eps)


def masked_rmse(pred_height, true_height, mask=None):
    """RMSE in meters. mask=None -> every pixel; otherwise only where mask is True."""
    if mask is None:
        mask = torch.ones_like(true_height, dtype=torch.bool)
    if mask.sum() == 0:
        return torch.tensor(float("nan"), device=pred_height.device)
    return torch.sqrt(torch.mean((pred_height[mask] - true_height[mask]) ** 2))


def masked_mae(pred_height, true_height, mask=None):
    """MAE in meters. mask=None -> every pixel; otherwise only where mask is True."""
    if mask is None:
        mask = torch.ones_like(true_height, dtype=torch.bool)
    if mask.sum() == 0:
        return torch.tensor(float("nan"), device=pred_height.device)
    return torch.mean((pred_height[mask] - true_height[mask]).abs())


## Mean/median need the actual pooled values, not a per-batch average (a mean of
## per-batch means isn't the overall mean unless every batch is the same size, and a
## median can't be computed from per-batch medians at all) -- so this collects every
## nonzero-height pixel across the whole validation set instead of reducing per batch.
## model=None -> baseline only (ground truth; same for every source sharing a split).
## model given -> also returns the model's own predicted mean/median, at the exact
## same pixel locations, so "is this source's average prediction close to the true
## average, or just its RMSE small" can be read directly off the evaluation table.
def height_distribution(val_loader, height_norm, device, model=None):
    if model is not None:
        model.eval()
    true_vals, pred_vals = [], []
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            true_height = targets[:, 3] * height_norm
            mask = true_height > 0
            true_vals.append(true_height[mask])
            if model is not None:
                pred_height = model(imgs)[:, 3] * height_norm
                pred_vals.append(pred_height[mask])

    true_vals = torch.cat(true_vals)
    if true_vals.numel() == 0:
        result = {"true_mean_height": float("nan"), "true_median_height": float("nan")}
    else:
        result = {"true_mean_height": true_vals.mean().item(),
                  "true_median_height": true_vals.median().item()}
    if model is not None:
        pred_vals = torch.cat(pred_vals)
        if pred_vals.numel() == 0:
            result["pred_mean_height"] = result["pred_median_height"] = float("nan")
        else:
            result["pred_mean_height"] = pred_vals.mean().item()
            result["pred_median_height"] = pred_vals.median().item()
    return result


# REVIEW REQUIRED
## iou_threshold and height_mask_threshold are decoupled on purpose -- they used to be
## one shared `threshold` parameter, so changing it for one silently changed the other.
def evaluate_metrics(model, val_loader, device, height_norm, iou_threshold=0.3, height_mask_threshold=0.3):
    model.eval()
    ## IoU: intersection/union accumulated as raw pixel counts across the WHOLE
    ## validation set, divided once at the end -- not computed per-batch and averaged.
    ## A per-batch ratio, mean-of-batches, is sensitive to batch_size and batch
    ## composition (verified: same val tiles, same predictions, batch_size 4 vs 23
    ## alone swung iou_building from 0.28 to 0.36). Pooling counts first removes that
    ## -- the result no longer depends on how validation happens to be batched.
    iou_counts = {c: [0, 0] for c in ("building", "vegetation", "water")}  # [intersection, union]
    ## mae_height/rmse_height: unmasked, every pixel -- the primary RQ1/RQ2 metric.
    ## rmse_building/rmse_vegetation: masked to where that class is present in the
    ## target -- diagnostic (where does height error concentrate), not the headline number.
    height_scores = {k: [] for k in ["mae_height", "rmse_height", "rmse_building", "rmse_vegetation"]}
    ## Height-as-classification (low/medium/high, see datamodule.HEIGHT_BIN_EDGES):
    ## [true_bin, pred_bin] confusion counts, pooled across the whole val set for the
    ## same reason IoU is pooled above -- a per-batch accuracy average is batch-size
    ## sensitive. Only defined where true_height > 0 (bare ground isn't a height class).
    n_bins = len(HEIGHT_BIN_NAMES)
    height_bin_confusion = torch.zeros(n_bins, n_bins, dtype=torch.long)
    edges = torch.tensor(HEIGHT_BIN_EDGES)

    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)

            pred = torch.clamp(outputs[:, :3], 0, 1)
            true = torch.clamp(targets[:, :3], 0, 1)
            ## channel 3 is height, stored normalized -> back to meters
            pred_height = outputs[:, 3] * height_norm
            true_height = targets[:, 3] * height_norm

            for i, c in enumerate(("building", "vegetation", "water")):
                intersection, union = _mask_counts(pred[:, i], true[:, i], iou_threshold)
                iou_counts[c][0] += intersection.item()
                iou_counts[c][1] += union.item()

            height_scores["mae_height"].append(masked_mae(pred_height, true_height))
            height_scores["rmse_height"].append(masked_rmse(pred_height, true_height))
            height_scores["rmse_building"].append(
                masked_rmse(pred_height, true_height, true[:, 0] > height_mask_threshold))
            height_scores["rmse_vegetation"].append(
                masked_rmse(pred_height, true_height, true[:, 1] > height_mask_threshold))

            hmask = true_height > 0
            if hmask.any():
                true_bin = torch.bucketize(true_height[hmask].cpu(), edges)
                pred_bin = torch.bucketize(pred_height[hmask].cpu(), edges)
                idx = true_bin * n_bins + pred_bin
                height_bin_confusion += torch.bincount(idx, minlength=n_bins * n_bins).reshape(n_bins, n_bins)

    results = {}
    for c, (intersection, union) in iou_counts.items():
        results[f"iou_{c}"] = (intersection / union) if union > 0 else float("nan")
    for name, values in height_scores.items():
        results[name] = torch.nanmean(torch.stack(values)).item()
    total_bin_pixels = height_bin_confusion.sum().item()
    results["height_bin_accuracy"] = (
        (torch.trace(height_bin_confusion).item() / total_bin_pixels) if total_bin_pixels > 0 else float("nan"))
    ## Per-class F1 from the pooled confusion matrix directly (not sklearn, which needs
    ## raw y_true/y_pred arrays -- all we keep is the already-pooled counts, same
    ## pooled-not-per-batch reasoning as IoU/accuracy above). TP/FP/FN read off row c
    ## (true=c) and column c (pred=c) of the [true, pred] matrix.
    f1_per_class = {}
    for i, name in enumerate(HEIGHT_BIN_NAMES):
        tp = height_bin_confusion[i, i].item()
        fp = height_bin_confusion[:, i].sum().item() - tp
        fn = height_bin_confusion[i, :].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_per_class[name] = (2 * precision * recall / (precision + recall)
                              if (precision + recall) > 0 else 0.0)
        results[f"height_bin_f1_{name}"] = f1_per_class[name]
    results["height_bin_f1_macro"] = sum(f1_per_class.values()) / len(f1_per_class)
    results["height_bin_confusion"] = height_bin_confusion.tolist()  # rows=true, cols=pred, order=HEIGHT_BIN_NAMES
    return results


def visualize_results(model, dataset, config, device, num_samples=5):
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
            ## Per-sample overall (unmasked) height MAE, for a quick sanity read next to the maps
            sample_mae = float(np.abs(pred[3] - true[3]).mean())

            fig, axs = plt.subplots(2, 4, figsize=(20, 10))
            for c in range(4):
                vmin, vmax = (0, 1) if c < 3 else (0, h_norm)
                axs[0, c].imshow(true[c], vmin=vmin, vmax=vmax)
                axs[0, c].set_title(f"True {target_names[c]}")
                axs[0, c].axis("off")
                axs[1, c].imshow(pred[c], vmin=vmin, vmax=vmax)
                axs[1, c].set_title(f"Predicted {target_names[c]}")
                axs[1, c].axis("off")
            plt.suptitle(f"Sample {i + 1} - True vs Predicted (height MAE: {sample_mae:.3f}m)")
            plt.savefig(config.viz_output_dir / f"visualization_{i}.png")
            plt.close(fig)


def plot_height_rmse_vs_epoch(epochs, rmse_overall, rmse_building, rmse_vegetation, path):
    """Val height RMSE (m) vs training epoch, within this one run.
    rmse_overall (unmasked, every pixel) is the headline curve; building/vegetation
    (masked to where that class is present) are diagnostic. This is a training-progress
    curve, NOT a label-efficiency curve -- it shows one run's epochs, not different
    amounts of training data. For the RQ2 label-efficiency sweep (accuracy vs number of
    labeled tiles, across separate runs), see scripts/label_efficiency_sweep.py."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, rmse_overall, marker="^", linewidth=2.5, label="Overall height RMSE")
    ax.plot(epochs, rmse_building, marker="o", linestyle="--", alpha=0.7, label="Building height RMSE")
    ax.plot(epochs, rmse_vegetation, marker="s", linestyle="--", alpha=0.7, label="Vegetation height RMSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Height RMSE (m) — lower is better")
    ax.set_title("Height accuracy vs training epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path)
    plt.close(fig)


## wandb.Table (already logged in train()) renders as a plain numeric grid in the
## WandB UI, not a heatmap -- this is the actual visual confusion matrix. Normalized
## by true-row (each row sums to 1) so color reads as per-class recall, annotated
## with the raw pixel counts so the absolute scale isn't lost.
def plot_height_bin_confusion(confusion, path):
    confusion = np.asarray(confusion, dtype=np.float64)
    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, row_sums, out=np.zeros_like(confusion), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(normalized, annot=confusion.astype(np.int64), fmt="d", cmap="Blues",
                vmin=0, vmax=1, xticklabels=HEIGHT_BIN_NAMES, yticklabels=HEIGHT_BIN_NAMES,
                cbar_kws={"label": "row-normalized (recall)"}, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Height-bin confusion matrix")
    fig.tight_layout()
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


## use_wandb=True needs credentials already resolvable, non-interactively, before
## wandb.init() runs -- fail fast with a clear error instead of wandb's own
## interactive login prompt, which hangs a non-interactive Slurm job forever. wandb
## itself resolves credentials from either WANDB_API_KEY or a cached `wandb login`
## (~/.netrc) -- check both, the same two places it checks.
def _wandb_credentials_available():
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc_path = Path.home() / ".netrc"
    return netrc_path.exists() and "api.wandb.ai" in netrc_path.read_text()


def _init_wandb(config):
    if not _wandb_credentials_available():
        raise RuntimeError(
            "config.use_wandb=True but no WandB credentials found -- set WANDB_API_KEY "
            "in the environment, or run `wandb login` on this machine, before starting this job."
        )
    return wandb.init(entity=config.wandb_entity, project=config.wandb_project,
                       name=config.experiment_name, config=config.model_dump(mode="json"))


# REVIEW REQUIRED
## Main entry point: config -> trained model + artifacts in outputs/<experiment_name>/.
## save_artifacts=False skips every disk write (checkpoints, config.yaml, loss/height
## curves, sample visualizations) and the visualize_results() forward passes -- metrics
## are still computed and returned. For label_efficiency_sweep.py, which calls train()
## once per tile budget and only needs the returned metrics dict, not N full per-run
## artifact sets it would otherwise never look at.
def train(config, save_artifacts=True):
    seed_everything(config.random_seed)
    device = get_device()
    if save_artifacts:
        config.make_dirs()
        config.save()

    train_loader, val_loader, distributions = build_train_val_loaders(config)
    if save_artifacts and distributions is not None:
        with open(config.class_distribution_path, "w") as f:
            json.dump(distributions, f, indent=2)

    run = _init_wandb(config) if config.use_wandb else None
    if run is not None and distributions is not None:
        ## Static (computed once at setup, not per epoch) -- logged at step 0.
        flat = {}
        for split, stats in distributions["class"].items():
            for cls, s in stats.items():
                for k, v in s.items():
                    flat[f"data/class_{split}_{cls}_{k}"] = v
        for split, stats in distributions["height_bins"].items():
            for name, pct in stats.items():
                flat[f"data/height_{split}_{name}"] = pct
        wandb.log(flat, step=0)
        if save_artifacts:
            ## Raw file, not just the flattened scalars above -- visible/downloadable
            ## from the run's Files tab.
            wandb.save(str(config.class_distribution_path))

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
    epochs_seen = []
    height_mae_overall, height_rmse_overall = [], []
    height_rmse_building, height_rmse_vegetation = [], []
    best_height_rmse = float("inf")

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

        metrics = evaluate_metrics(model, val_loader, device,
                                   config.height_normalization_constant,
                                   iou_threshold=config.iou_threshold,
                                   height_mask_threshold=config.height_mask_threshold)
        epochs_seen.append(epoch)
        height_mae_overall.append(metrics["mae_height"])
        height_rmse_overall.append(metrics["rmse_height"])
        height_rmse_building.append(metrics["rmse_building"])
        height_rmse_vegetation.append(metrics["rmse_vegetation"])
        print(f"  Height MAE/RMSE (m) overall: {metrics['mae_height']:.3f}/{metrics['rmse_height']:.3f} | "
              f"RMSE building={metrics['rmse_building']:.3f} vegetation={metrics['rmse_vegetation']:.3f}")
        print(f"  Height-as-classification: accuracy={metrics['height_bin_accuracy']:.3f} "
              f"macro-F1={metrics['height_bin_f1_macro']:.3f} "
              f"(F1 low/medium/high={metrics['height_bin_f1_low']:.3f}/"
              f"{metrics['height_bin_f1_medium']:.3f}/{metrics['height_bin_f1_high']:.3f}) | "
              f"confusion[true][pred]={metrics['height_bin_confusion']}")

        if run is not None:
            wandb.log({"train/loss": train_loss, "val/loss": val_loss,
                       **{f"val/{k}": v for k, v in metrics.items() if k != "height_bin_confusion"}},
                      step=epoch)

        ## "Best" checkpoint = lowest overall (unmasked) height RMSE, not lowest val loss --
        ## ties model selection to the actual research metric (RQ1/RQ2) instead of a
        ## training-loss number that, under "weighted" loss, blends height with the
        ## auxiliary landcover term and isn't in meters.
        if metrics["rmse_height"] < best_height_rmse:
            best_height_rmse = metrics["rmse_height"]
            if save_artifacts:
                torch.save(model.state_dict(), config.best_model_path)
                print(f"  New best (val height RMSE {metrics['rmse_height']:.3f}m) -> {config.best_model_path}")
            else:
                print(f"  New best (val height RMSE {metrics['rmse_height']:.3f}m)")

        if epoch % 10 == 0:
            print("  Challenge-style evaluation:")
            for name, value in metrics.items():
                if name != "height_bin_confusion":
                    print(f"    {name}: {value:.4f}")

    if save_artifacts:
        torch.save(model.state_dict(), config.last_model_path)

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.legend()
        plt.savefig(config.loss_curve_path)
        plt.close()

        plot_height_rmse_vs_epoch(epochs_seen, height_rmse_overall, height_rmse_building,
                                  height_rmse_vegetation, config.height_curve_path)

        plot_height_bin_confusion(metrics["height_bin_confusion"], config.height_confusion_path)

        visualize_results(model, val_loader.dataset, config, device)

        if run is not None:
            ## Re-logs the already-saved local PNGs -- generated once, uploaded once.
            wandb.log({
                "train/loss_curve": wandb.Image(str(config.loss_curve_path)),
                "val/height_rmse_curve": wandb.Image(str(config.height_curve_path)),
                "val/height_bin_confusion_plot": wandb.Image(str(config.height_confusion_path)),
                "val/sample_visualizations": [
                    wandb.Image(str(p)) for p in sorted(config.viz_output_dir.glob("visualization_*.png"))],
            })

        print(f"Artifacts saved under {config.experiment_dir}")
    else:
        print("Training complete (save_artifacts=False -- no checkpoints, plots, or visualizations written)")

    if run is not None:
        ## Final epoch's confusion matrix only (per-epoch would be 100 tables for the
        ## full-data configs) -- rows/cols both ordered HEIGHT_BIN_NAMES.
        wandb.log({"val/height_bin_confusion": wandb.Table(
            columns=[f"pred_{n}" for n in HEIGHT_BIN_NAMES],
            data=metrics["height_bin_confusion"],
        )})
        wandb.finish()

    return model, {"train_losses": train_losses, "val_losses": val_losses,
                   "best_val_loss": min(val_losses),
                   "n_train_tiles": len(train_loader.dataset),
                   "epochs_seen": epochs_seen,
                   "height_mae_overall": height_mae_overall,
                   "height_rmse_overall": height_rmse_overall,
                   "best_height_rmse_overall": min(height_rmse_overall),
                   "height_rmse_building": height_rmse_building,
                   "height_rmse_vegetation": height_rmse_vegetation,
                   ## last epoch's IoU (metrics is already computed once per epoch above --
                   ## no need for a second evaluate_metrics() pass over val_loader here)
                   "iou_building": metrics["iou_building"],
                   "iou_vegetation": metrics["iou_vegetation"],
                   "iou_water": metrics["iou_water"],
                   "height_bin_accuracy": metrics["height_bin_accuracy"],
                   "height_bin_f1_macro": metrics["height_bin_f1_macro"],
                   "height_bin_confusion": metrics["height_bin_confusion"]}
