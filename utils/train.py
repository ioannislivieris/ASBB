import time
import torch
import torch.nn as nn
import numpy as np
from optimizer.StepTunedSGD import AdaptiveBBTuner
from utils.classification_evaluation import performance_evaluation

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

def train_ASBB(
    model: nn.Module,
    train_loader,
    test_loader,
    optimizer=None,
    criterion=None,
    epochs: int = 10,
    device: torch.device = torch.device("cpu"),
    algorithm: str = None,
    scheduler=None,
    accumulation_steps: int = 1,
    grad_clip: float = 1.0,
    adaptive_tuner: AdaptiveBBTuner = None,
    use_train_loss_for_tuning: bool = False,
):
    """
    Trains a model using Step-Tuned SGD with adaptive tau and beta tuning.

    Args:
        model: Neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer: Step-Tuned SGD optimizer
        criterion: Loss function (nn.BCELoss, nn.CrossEntropyLoss, nn.BCEWithLogitsLoss)
        epochs: Number of training epochs
        device: Device to train on
        algorithm: Algorithm name for logging
        scheduler: Learning rate scheduler (optional)
        accumulation_steps: Number of mini-batches to accumulate
        grad_clip: Maximum gradient norm for clipping
        adaptive_tuner: AdaptiveBBTuner instance (if None, no adaptation)
        use_train_loss_for_tuning: Use train loss instead of test loss for adaptation
    """

    if not optimizer:
        raise ValueError("Optimizer is not defined.")
    if not criterion:
        raise ValueError("Criterion (loss function) is not defined.")

    # Detect loss type
    is_bce = isinstance(criterion, (nn.BCELoss, nn.BCEWithLogitsLoss))

    history = {
        "train_loss": [],
        "test_loss": [],
        "test_metrics": [],
        "iterations": [],
        "time": [],
        "tau_history": [],
        "beta_history": [],
        "bb_stats": [],
    }

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_losses = []
        accumulated_loss = 0.0
        batch_count = 0
        accumulated_grads = None

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Compute gradients for current mini-batch
            model.zero_grad()
            outputs = model(batch_x)

            if is_bce:
                outputs = outputs.squeeze()
                loss = criterion(outputs, batch_y.float())
            else:
                loss = criterion(outputs, batch_y.long())

            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            # Accumulate gradients
            if accumulated_grads is None:
                accumulated_grads = [
                    p.grad.clone() if p.grad is not None else None
                    for p in model.parameters()
                ]
            else:
                for i, p in enumerate(model.parameters()):
                    if p.grad is not None:
                        if accumulated_grads[i] is None:
                            accumulated_grads[i] = p.grad.clone()
                        else:
                            accumulated_grads[i] += p.grad

            accumulated_loss += loss.item()
            batch_count += 1

            # Update when accumulated enough
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(
                train_loader
            ):
                # Average accumulated gradients
                for i, p in enumerate(model.parameters()):
                    if accumulated_grads[i] is not None:
                        p.grad = accumulated_grads[i] / batch_count

                def closure():
                    return torch.tensor(accumulated_loss / batch_count)

                # First half step
                loss1 = optimizer.step(closure)

                # Second half step
                def closure_second():
                    model.zero_grad()
                    outputs = model(batch_x)

                    if is_bce:
                        outputs = outputs.squeeze()
                        loss = criterion(outputs, batch_y.float())
                    else:
                        loss = criterion(outputs, batch_y.long())

                    loss.backward()
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=grad_clip
                        )
                    return loss

                loss2 = optimizer.step(closure_second)

                train_losses.append(accumulated_loss / batch_count)

                # Reset accumulation
                accumulated_grads = None
                accumulated_loss = 0.0
                batch_count = 0

        avg_train_loss = np.mean(train_losses)

        # Evaluation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            all_predictions, all_labels = [], []
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)

                if is_bce:
                    outputs = outputs.squeeze()
                    test_loss += criterion(outputs, batch_y.float()).item()
                    # Predictions: raw probabilities for BCE
                    predicted = outputs
                else:
                    test_loss += criterion(outputs, batch_y.long()).item()
                    # Predictions: argmax for multi-class classification
                    predicted = outputs.argmax(dim=1)

                all_predictions.append(predicted.cpu())
                all_labels.append(batch_y.cpu())

        test_loss /= len(test_loader)
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        avg_metrics = performance_evaluation(all_labels, all_predictions)

        # Get BB statistics
        bb_stats = optimizer.get_bb_statistics()

        # Adaptive parameter tuning
        if adaptive_tuner is not None:
            loss_for_tuning = avg_train_loss if use_train_loss_for_tuning else test_loss
            new_tau, new_beta, adjustment_info = adaptive_tuner.update(
                loss_for_tuning, bb_stats, epoch
            )

            # Update optimizer parameters
            for param_group in optimizer.param_groups:
                param_group["abb_tau"] = new_tau
                param_group["beta"] = new_beta

            # Reset BB statistics after adjustment (optional)
            # optimizer.reset_bb_statistics()

        elapsed = time.time() - start_time

        # Record history
        history["train_loss"].append(float(avg_train_loss))
        history["test_loss"].append(float(test_loss))
        history["test_metrics"].append(avg_metrics)
        history["iterations"].append(epoch + 1)
        history["time"].append(elapsed)
        history["bb_stats"].append(bb_stats)

        if adaptive_tuner is not None:
            history["tau_history"].append(adaptive_tuner.get_tau())
            history["beta_history"].append(adaptive_tuner.get_beta())

        # Logging
        algo_str = f"[{algorithm}] | " if algorithm else ""
        accum_str = f"[Accum={accumulation_steps}] " if accumulation_steps > 1 else ""
        adaptive_str = ""
        if adaptive_tuner is not None:
            adaptive_str = (
                f"[τ={adaptive_tuner.get_tau():.3f} β={adaptive_tuner.get_beta():.3f}] "
            )

        print(
            f"[{epoch + 1:2.0f}/{epochs}] {algo_str}{accum_str}{adaptive_str}"
            f"Train Loss: {avg_train_loss:.4f}   "
            f"Test Loss: {test_loss:.4f} | "
            f"Accuracy: {avg_metrics['Accuracy']:.2f}%   "
            f"AUC: {avg_metrics['AUC']:.4f}   "
            f"GM: {avg_metrics['GM']:.4f} | "
            f"Time: {elapsed:.2f}s"
        )

        if scheduler is not None:
            scheduler.step()

    return history
