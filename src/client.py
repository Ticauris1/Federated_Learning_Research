import torch
from contextlib import nullcontext
from tqdm.auto import tqdm
from typing import Dict, List      # You added this
from torch.optim import Optimizer  # You added this
from models import ImageClassifier # <-- ADD THIS LINE

# ===========================================================
# Single, correct epoch_train that handles CUDA AMP vs MPS/CPU
# ============================================================
def epoch_train(model, loader, optimizer, criterion, device, use_amp: bool = True):
    """
    One training epoch.
    - Uses torch.amp.autocast('cuda') + torch.amp.GradScaler('cuda') on NVIDIA GPUs.
    - Falls back to full precision on MPS (Mac) and CPU.
    Returns: (mean_loss, accuracy)
    """
    model.train()

    is_cuda = (device.type == "cuda")
    is_mps  = (device.type == "mps")
    amp_enabled = (use_amp and is_cuda)

    # GradScaler only for CUDA
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if is_cuda else None

    running_loss, running_correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Training", leave=False)

    for inputs, labels in pbar:
        inputs = inputs.to(device, non_blocking=is_cuda)   # non_blocking helps on CUDA with pinned memory
        labels = labels.to(device, non_blocking=is_cuda)

        optimizer.zero_grad(set_to_none=True)

        if amp_enabled:
            # CUDA path with AMP
            # This is the corrected line:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(inputs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # MPS/CPU (or CUDA without AMP)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_correct += (logits.argmax(1) == labels).sum().item()
        total += bs

        pbar.set_postfix(
            loss=f"{running_loss / max(total, 1):.4f}",
            acc=f"{running_correct / max(total, 1):.4f}"
        )

    return running_loss / max(total, 1), running_correct / max(total, 1)

class GradualUnfreezer:
    """Unfreeze the backbone gradually by blocks according to a user schedule."""
    def __init__(self, model: ImageClassifier, unfreeze_schedule: Dict[int, int]):
        self.model = model
        self.schedule = unfreeze_schedule
        self.backbone_layers: List[nn.Module] = list(model.backbone.children())
        self.current_block_to_unfreeze = len(self.backbone_layers) - 1
        # Start with the backbone completely frozen
        for param in self.model.backbone.parameters():
            param.requires_grad = False

    def step(self, epoch: int, optimizer: Optimizer, base_learning_rate: float):
        if epoch not in self.schedule:
            return
        blocks_to_unfreeze = self.schedule[epoch]
        print(f"\n🔥 Epoch {epoch}: Unfreezing {blocks_to_unfreeze} layer block(s)...")
        for _ in range(blocks_to_unfreeze):
            if self.current_block_to_unfreeze < 0:
                print("  - All blocks already unfrozen.")
                break
            block = self.backbone_layers[self.current_block_to_unfreeze]
            for param in block.parameters():
                param.requires_grad = True
            # Add the newly unfrozen parameters to the optimizer with a lower learning rate
            optimizer.add_param_group({'params': list(block.parameters()), 'lr': base_learning_rate / 10})
            print(f"  - Unfroze block {self.current_block_to_unfreeze} and added to optimizer.")
            self.current_block_to_unfreeze -= 1

def _local_train_fedprox(model, train_loader, criterion, optimizer, device,
                         global_params, mu: float, local_epochs: int = 1,
                         use_amp: bool = True):
    is_cuda = (device.type == "cuda")
    amp_ctx = torch.amp.autocast("cuda") if (use_amp and is_cuda) else nullcontext()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and is_cuda))

    model.train()
    total, correct, running_loss = 0, 0, 0.0

    for _ in range(local_epochs):
        pbar = tqdm(train_loader, leave=False, desc="FedProx local")
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=is_cuda)
            yb = yb.to(device, non_blocking=is_cuda)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                logits = model(xb)
                base_loss = criterion(logits, yb)

                # ---- proximal term: sum_k ||w_k - w_global_k||^2 over float tensors ----
                prox = 0.0
                for name, param in model.state_dict().items():
                    gparam = global_params[name]
                    if torch.is_floating_point(param):
                        diff = (param - gparam.to(param.dtype).to(param.device))
                        prox = prox + (diff * diff).sum()
                loss = base_loss + (mu / 2.0) * prox

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            bs = yb.size(0)
            running_loss += loss.item() * bs
            correct += (logits.argmax(1) == yb).sum().item()
            total += bs
            pbar.set_postfix(loss=f"{running_loss/max(total,1):.4f}", acc=f"{correct/max(total,1):.4f}")

    return running_loss / max(total, 1), correct / max(total, 1)