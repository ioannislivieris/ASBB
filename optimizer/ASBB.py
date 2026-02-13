import torch
import numpy as np
from torch.optim.optimizer import Optimizer

class StepTunedSGD(Optimizer):
    """Step-Tuned SGD using Barzilai-Borwein inspired step-size tuning."""

    def __init__(
        self,
        params,
        alpha=1.0,
        nu=1.5,
        beta=0.95,
        m_tilde=0.1,
        M_tilde=10.0,
        delta=0.0001,
        weight_decay=0,
        verbose=False,
        epoch_based_decay=False,
        bb_type="ABB",  # 'BB1', 'BB2', or 'ABB'
        abb_tau=0.5,  # Threshold for ABB switching (cos²θ threshold)
    ):
        if alpha <= 0.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if not (0 <= beta < 1):
            raise ValueError(f"Invalid beta: {beta}")
        if not (0 < delta < 0.5):
            raise ValueError(f"Invalid delta: {delta}")
        if bb_type not in ["BB1", "BB2", "ABB"]:
            raise ValueError(
                f"Invalid bb_type: {bb_type}. Must be 'BB1', 'BB2', or 'ABB'"
            )
        if not (0 < abb_tau < 1):
            raise ValueError(f"Invalid abb_tau: {abb_tau}. Must be in (0, 1)")

        defaults = dict(
            lr=alpha,  # For PyTorch scheduler compatibility
            alpha=alpha,
            nu=nu,
            beta=beta,
            m_tilde=m_tilde,
            M_tilde=M_tilde,
            delta=delta,
            weight_decay=weight_decay,
            verbose=verbose,
            epoch_based_decay=epoch_based_decay,
            bb_type=bb_type,
            abb_tau=abb_tau,
        )
        super(StepTunedSGD, self).__init__(params, defaults)

        self.state["step"] = 0
        self.state["half_step"] = False
        self.state["epoch"] = 0
        self.state["last_bb_choice"] = "BB1"  # For ABB tracking

        # Statistics tracking
        self.state["bb1_count"] = 0
        self.state["bb2_count"] = 0
        self.state["negative_curvature_count"] = 0

    def set_epoch(self, epoch):
        """Call this at the start of each epoch"""
        self.state["epoch"] = epoch

    def get_bb_statistics(self):
        """
        Get statistics about BB method usage.

        Returns:
            dict: Dictionary containing:
                - bb1_count: Number of times BB1 was selected
                - bb2_count: Number of times BB2 was selected
                - total_steps: Total number of steps
                - bb1_percentage: Percentage of BB1 usage
                - bb2_percentage: Percentage of BB2 usage
                - negative_curvature_count: Number of times negative curvature was detected
        """
        bb1_count = self.state["bb1_count"]
        bb2_count = self.state["bb2_count"]
        total_steps = bb1_count + bb2_count

        return {
            "bb1_count": bb1_count,
            "bb2_count": bb2_count,
            "total_steps": total_steps,
            "bb1_percentage": 100.0 * bb1_count / total_steps
            if total_steps > 0
            else 0.0,
            "bb2_percentage": 100.0 * bb2_count / total_steps
            if total_steps > 0
            else 0.0,
            "negative_curvature_count": self.state["negative_curvature_count"],
        }

    def reset_bb_statistics(self):
        """Reset the BB statistics counters."""
        self.state["bb1_count"] = 0
        self.state["bb2_count"] = 0
        self.state["negative_curvature_count"] = 0

    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step."""
        if closure is None:
            raise RuntimeError("Step-Tuned SGD requires closure")

        k = self.state["step"]
        half_step = self.state["half_step"]

        # Get current alpha (potentially modified by scheduler)
        current_alpha = self.param_groups[0]["lr"]

        # Compute step size decay
        if self.param_groups[0]["epoch_based_decay"]:
            epoch = self.state["epoch"]
            eta_k = current_alpha / (
                (epoch + 1) ** (0.5 + self.param_groups[0]["delta"])
            )
        else:
            eta_k = current_alpha / ((k + 1) ** (0.5 + self.param_groups[0]["delta"]))

        # Initialize gamma
        if "gamma" not in self.state:
            self.state["gamma"] = 1.0

        gamma_k = self.state["gamma"]

        if not half_step:
            # First half step
            with torch.enable_grad():
                loss = closure()

            for group in self.param_groups:
                weight_decay = group["weight_decay"]

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]

                    # Save θ_k
                    state["theta_k"] = p.data.clone()

                    # Save ∇J_{B_k}(θ_k)
                    grad = p.grad.data.clone()
                    if weight_decay != 0:
                        grad = grad.add(p.data, alpha=weight_decay)

                    state["grad_theta_k"] = grad

                    # Update: θ_{k+1/2} = θ_k - η_k * γ_k * ∇J_{B_k}(θ_k)
                    p.data.add_(grad, alpha=-eta_k * gamma_k)

            self.state["half_step"] = True
            return loss.item()

        else:
            # Second half step
            with torch.enable_grad():
                loss = closure()

            delta_theta_norm_sq_total = 0.0
            delta_g_norm_sq_total = 0.0
            G_hat_dot_delta_theta_total = 0.0

            for group in self.param_groups:
                weight_decay = group["weight_decay"]
                beta = group["beta"]

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]

                    # Get ∇J_{B_k}(θ_{k+1/2})
                    grad_theta_k_half = p.grad.data.clone()
                    if weight_decay != 0:
                        grad_theta_k_half = grad_theta_k_half.add(
                            p.data, alpha=weight_decay
                        )

                    # Compute Δθ_{B_k} = θ_{k+1/2} - θ_k
                    delta_theta = p.data - state["theta_k"]

                    # Compute Δg_{B_k} = ∇J_{B_k}(θ_{k+1/2}) - ∇J_{B_k}(θ_k)
                    delta_g = grad_theta_k_half - state["grad_theta_k"]

                    # Update exponential moving average G_k
                    if "G" not in state:
                        state["G"] = delta_g.clone()
                    else:
                        state["G"] = beta * state["G"] + (1 - beta) * delta_g

                    # Debias: Ĝ_k = G_k / (1 - β^{k+1})
                    bias_correction = 1 - beta ** (k + 1)
                    if bias_correction > 1e-8:
                        G_hat = state["G"] / bias_correction
                    else:
                        G_hat = state["G"]

                    # Accumulate for gamma computation
                    delta_theta_norm_sq_total += (delta_theta.norm() ** 2).item()
                    delta_g_norm_sq_total += (G_hat.norm() ** 2).item()
                    G_hat_dot_delta_theta_total += torch.sum(G_hat * delta_theta).item()

                    # Update: θ_{k+1} = θ_{k+1/2} - η_k * γ_k * ∇J_{B_k}(θ_{k+1/2})
                    p.data.add_(grad_theta_k_half, alpha=-eta_k * gamma_k)

            # Compute γ_{k+1} using selected BB method
            bb_type = self.param_groups[0]["bb_type"]
            verbose = self.param_groups[0]["verbose"]

            # Compute cos²θ for ABB method
            # cos²θ = (<s, y>)² / (||s||² * ||y||²)
            if bb_type == "ABB":
                if delta_theta_norm_sq_total > 1e-10 and delta_g_norm_sq_total > 1e-10:
                    cos_sq_theta = (G_hat_dot_delta_theta_total**2) / (
                        delta_theta_norm_sq_total * delta_g_norm_sq_total
                    )
                else:
                    cos_sq_theta = 0.0
                # Adaptive selection based on cos²θ
                tau = self.param_groups[0]["abb_tau"]
                
                if cos_sq_theta > tau:
                    # Use BB1 (short step) when cos²θ → 1 (nearly parallel)
                    selected_bb = "BB1"
                    denominator = G_hat_dot_delta_theta_total
                    numerator = delta_theta_norm_sq_total
                    self.state["bb1_count"] += 1
                else:
                    # Use BB2 (long step) when cos²θ → 0 (nearly orthogonal)
                    selected_bb = "BB2"
                    denominator = delta_g_norm_sq_total
                    numerator = G_hat_dot_delta_theta_total
                    self.state["bb2_count"] += 1

                self.state["last_bb_choice"] = selected_bb

            elif bb_type == "BB1":
                # BB1: γ = ||Δθ||² / <Ĝ, Δθ>
                denominator = G_hat_dot_delta_theta_total
                numerator = delta_theta_norm_sq_total
                selected_bb = "BB1"
                cos_sq_theta = None
                self.state["bb1_count"] += 1

            else:  # BB2
                # BB2: γ = <Ĝ, Δθ> / ||Ĝ||²
                denominator = delta_g_norm_sq_total
                numerator = G_hat_dot_delta_theta_total
                selected_bb = "BB2"
                cos_sq_theta = None
                self.state["bb2_count"] += 1

            # Compute gamma
            negative_curvature = False
            if abs(denominator) > 1e-10 and abs(numerator) > 1e-10:
                gamma_new = numerator / denominator

                # Check for positive curvature
                if G_hat_dot_delta_theta_total > 0:
                    gamma_new = abs(gamma_new)
                else:
                    # Negative curvature - use large step
                    gamma_new = self.param_groups[0]["nu"]
                    negative_curvature = True
                    self.state["negative_curvature_count"] += 1
                    if verbose and k % 10 == 0:
                        print(f"Step {k}: Negative curvature detected")
            else:
                # Denominator too small - use nu
                gamma_new = self.param_groups[0]["nu"]
                if verbose and k % 10 == 0:
                    print(f"Step {k}: Small denominator, using nu")

            # Clip gamma to reasonable bounds
            gamma_new = max(
                self.param_groups[0]["m_tilde"],
                min(gamma_new, self.param_groups[0]["M_tilde"]),
            )

            self.state["gamma"] = gamma_new
            self.state["step"] += 1
            self.state["half_step"] = False

            if verbose and k % 10 == 0:
                if bb_type == "ABB":
                    print(
                        f"Step {k}: [ABB->{selected_bb}] gamma = {gamma_new:.4f}, "
                        f"cos²θ = {cos_sq_theta:.4f}, "
                        f"||Δθ||² = {delta_theta_norm_sq_total:.4e}, "
                        f"||Ĝ||² = {delta_g_norm_sq_total:.4e}, "
                        f"<Ĝ,Δθ> = {G_hat_dot_delta_theta_total:.4e}"
                    )
                else:
                    print(
                        f"Step {k}: [{selected_bb}] gamma = {gamma_new:.4f}, "
                        f"||Δθ||² = {delta_theta_norm_sq_total:.4e}, "
                        f"||Ĝ||² = {delta_g_norm_sq_total:.4e}, "
                        f"<Ĝ,Δθ> = {G_hat_dot_delta_theta_total:.4e}"
                    )

            return loss.item()


class AdaptiveBBTuner:
    """
    Adaptive tuner for ABB tau and beta parameters based on training dynamics.
    Adjusts parameters based on loss reduction, gradient variance, and convergence behavior.
    """

    def __init__(
        self,
        initial_tau=0.5,
        initial_beta=0.95,
        tau_range=(0.2, 0.8),
        beta_range=(0.85, 0.99),
        window_size=5,
        adjustment_rate=0.05,
        verbose=True,
    ):
        """
        Args:
            initial_tau: Starting tau value
            initial_beta: Starting beta value
            tau_range: (min, max) allowable tau values
            beta_range: (min, max) allowable beta values
            window_size: Number of epochs to consider for loss reduction
            adjustment_rate: How aggressively to adjust parameters (0-1)
            verbose: Print adjustment decisions
        """
        self.tau = initial_tau
        self.beta = initial_beta
        self.tau_min, self.tau_max = tau_range
        self.beta_min, self.beta_max = beta_range
        self.window_size = window_size
        self.adjustment_rate = adjustment_rate
        self.verbose = verbose

        # History tracking
        self.loss_history = []
        self.tau_history = [initial_tau]
        self.beta_history = [initial_beta]
        self.bb_stats_history = []

        # State tracking
        self.stagnation_count = 0
        self.oscillation_count = 0

    def update(self, current_loss, bb_stats, epoch):
        """
        Update tau and beta based on training dynamics.

        Args:
            current_loss: Current epoch's training or validation loss
            bb_stats: Dictionary from optimizer.get_bb_statistics()
            epoch: Current epoch number

        Returns:
            tuple: (new_tau, new_beta, adjustment_info)
        """
        self.loss_history.append(current_loss)
        self.bb_stats_history.append(bb_stats)

        # Need at least 2 epochs to compute trends
        if len(self.loss_history) < 2:
            return self.tau, self.beta, "Initializing"

        # Compute loss reduction metrics
        loss_reduction = self._compute_loss_reduction()
        loss_variance = self._compute_loss_variance()
        is_stagnating = self._detect_stagnation()
        is_oscillating = self._detect_oscillation()
        bb_balance = self._compute_bb_balance(bb_stats)

        # Decision logic
        adjustment_info = []
        tau_adjustment = 0.0
        beta_adjustment = 0.0

        # === TAU ADJUSTMENTS ===

        # 1. Loss reduction rate
        if len(self.loss_history) >= self.window_size:
            prev_loss = self.loss_history[-self.window_size]
            loss_reduction_pct = ((prev_loss - current_loss) / (abs(prev_loss) + 1e-8)) * 100
            if loss_reduction_pct < 1 and not is_stagnating:
                # Less than 1% reduction - try more exploration (lower tau for longer steps)
                tau_adjustment += -self.adjustment_rate * 0.3
                adjustment_info.append(f"Low progress ({loss_reduction_pct:.1f}% < 1%) → explore with lower tau")
            elif loss_reduction_pct > 10:
                # Fast progress (>10% reduction) - maintain or be slightly more conservative
                tau_adjustment += self.adjustment_rate * 0.1
                adjustment_info.append(f"Fast progress ({loss_reduction_pct:.1f}% > 10%) → maintain/increase tau")

        # 3. Oscillation detection
        if is_oscillating:
            # Loss is oscillating - need more stability (higher tau for shorter steps)
            tau_adjustment += self.adjustment_rate * 0.4
            adjustment_info.append("Oscillation detected → increase tau for stability")

        # 4. Stagnation detection
        
        if is_stagnating:
            self.stagnation_count += 1
            if self.stagnation_count >= 3:
                # Persistent stagnation - try opposite strategy
                current_bb1_pct = bb_stats["bb1_percentage"]
                if current_bb1_pct > 50:
                    # Currently conservative, try aggressive
                    tau_adjustment = -self.adjustment_rate * 0.6
                    adjustment_info.append(
                        "Stagnation → switch to aggressive (low tau)"
                    )
                else:
                    # Currently aggressive, try conservative
                    tau_adjustment = self.adjustment_rate * 0.6
                    adjustment_info.append(
                        "Stagnation → switch to conservative (high tau)"
                    )
                self.stagnation_count = 0
        else:
            self.stagnation_count = 0

        # === BETA ADJUSTMENTS ===

        # 1. Loss variance (noise level)
        if loss_variance > 0.1:
            # High variance - increase beta for more smoothing
            beta_adjustment = self.adjustment_rate * 0.3
            adjustment_info.append("High loss variance → increase beta")
        elif loss_variance < 0.01:
            # Low variance - can decrease beta for more responsiveness
            beta_adjustment = -self.adjustment_rate * 0.2
            adjustment_info.append("Low loss variance → decrease beta")

        # 2. Oscillation
        if is_oscillating:
            # Oscillating - need more gradient smoothing
            beta_adjustment += self.adjustment_rate * 0.4
            adjustment_info.append("Oscillation → increase beta for smoothing")

        # 3. Negative curvature frequency
        if bb_stats["negative_curvature_count"] > bb_stats["total_steps"] * 0.1:
            # Frequent negative curvature - increase smoothing
            beta_adjustment += self.adjustment_rate * 0.3
            adjustment_info.append("Frequent neg. curvature → increase beta")

        # 4. Coupling with tau adjustments
        if abs(tau_adjustment) > 0.3:
            # Large tau changes need complementary beta adjustments
            if tau_adjustment < 0:
                # Decreasing tau (more aggressive) - increase beta for stability
                beta_adjustment += self.adjustment_rate * 0.2
                adjustment_info.append(
                    "Large tau decrease → compensate with beta increase"
                )
            else:
                # Increasing tau (more conservative) - can decrease beta slightly
                beta_adjustment -= self.adjustment_rate * 0.1
                adjustment_info.append("Large tau increase → slight beta decrease")

        # Apply adjustments with bounds
        new_tau = np.clip(self.tau + tau_adjustment, self.tau_min, self.tau_max)
        new_beta = np.clip(self.beta + beta_adjustment, self.beta_min, self.beta_max)

        # Update state
        self.tau = new_tau
        self.beta = new_beta
        self.tau_history.append(new_tau)
        self.beta_history.append(new_beta)

        if self.verbose and adjustment_info:
            print(f"\n[Epoch {epoch}] Adaptive Tuning:")
            print(
                f"  Loss: {current_loss:.4f} | Reduction: {loss_reduction:.4f} | Variance: {loss_variance:.4f}"
            )
            print(
                f"  BB1: {bb_stats['bb1_percentage']:.1f}% | BB2: {bb_stats['bb2_percentage']:.1f}%"
            )
            print(f"  Adjustments: {', '.join(adjustment_info)}")
            print(
                f"  New tau: {self.tau:.3f} (Δ{tau_adjustment:+.3f}) | New beta: {self.beta:.3f} (Δ{beta_adjustment:+.3f})"
            )

        return new_tau, new_beta, adjustment_info

    def _compute_loss_reduction(self):
        """Compute average loss reduction over window"""
        if len(self.loss_history) < 2:
            return 0.0

        window = min(self.window_size, len(self.loss_history))
        recent_losses = self.loss_history[-window:]

        # Average reduction per step
        reductions = [
            recent_losses[i] - recent_losses[i + 1]
            for i in range(len(recent_losses) - 1)
        ]
        return np.mean(reductions) if reductions else 0.0

    def _compute_loss_variance(self):
        """Compute variance of recent losses"""
        if len(self.loss_history) < 3:
            return 0.0

        window = min(self.window_size, len(self.loss_history))
        recent_losses = self.loss_history[-window:]
        return np.var(recent_losses)

    def _detect_stagnation(self):
        """Detect if loss has stagnated based on relative error"""
        if len(self.loss_history) < self.window_size:
            return False

        recent_losses = self.loss_history[-self.window_size:]
        
        # Compute relative errors between consecutive epochs
        relative_errors = []
        for i in range(len(recent_losses) - 1):
            prev_loss = recent_losses[i]
            curr_loss = recent_losses[i + 1]
            
            # Relative error: |current - previous| / |previous|
            rel_error = abs(curr_loss - prev_loss) / (abs(prev_loss) + 1e-8)
            relative_errors.append(rel_error)
        
        # Stagnation: average relative error is very small
        avg_relative_error = np.mean(relative_errors) if relative_errors else 0.0
        
        # Threshold: less than 1% average relative change
        return avg_relative_error < 0.01

    def _detect_oscillation(self):
        """Detect if loss is oscillating"""
        if len(self.loss_history) < 4:
            return False

        recent_losses = self.loss_history[-4:]

        # Check for zigzag pattern
        diff1 = recent_losses[1] - recent_losses[0]
        diff2 = recent_losses[2] - recent_losses[1]
        diff3 = recent_losses[3] - recent_losses[2]

        # Oscillation: alternating signs of differences
        oscillating = (diff1 * diff2 < 0) and (diff2 * diff3 < 0)

        if oscillating:
            self.oscillation_count += 1
        else:
            self.oscillation_count = max(0, self.oscillation_count - 1)

        return self.oscillation_count >= 2

    def _compute_bb_balance(self, bb_stats):
        """Compute how balanced BB1/BB2 usage is (0=perfect, 1=completely imbalanced)"""
        if bb_stats["total_steps"] == 0:
            return 0.5

        bb1_pct = bb_stats["bb1_percentage"] / 100.0
        # Perfect balance is 0.5, compute distance from it
        return abs(bb1_pct - 0.5) * 2

    def get_tau(self):
        """Get current tau value"""
        return self.tau

    def get_beta(self):
        """Get current beta value"""
        return self.beta

    def get_history(self):
        """Get full history of adjustments"""
        return {
            "tau_history": self.tau_history,
            "beta_history": self.beta_history,
            "loss_history": self.loss_history,
            "bb_stats_history": self.bb_stats_history,
        }