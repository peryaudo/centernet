import hydra
from omegaconf import OmegaConf
import albumentations as A
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from model import CenterNet
from dataset import COCODatasetWrapper
from loss import CenterNetLoss

@hydra.main(version_base=None, config_path=".", config_name="config")
def train_main(cfg):
    wandb.init(project="centernet", name=cfg.get("run_name", None), config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    # Load COCO dataset
    hf_dataset = load_dataset("detection-datasets/coco")
    
    # Create train/val transforms
    train_transforms = A.Compose([
        A.RandomScale(scale_limit=0.5, p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.LongestMaxSize(max_size=512),  # Resize keeping aspect ratio
        A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, value=0),  # Pad to square
        A.Normalize()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    val_transforms = A.Compose([
        A.LongestMaxSize(max_size=512),  # Resize keeping aspect ratio
        A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, value=0),  # Pad to square
        A.Normalize()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    # Create datasets
    train_dataset = COCODatasetWrapper(
        hf_dataset["train"], 
        transforms=train_transforms,
        output_size=(cfg["output_size"], cfg["output_size"]),
        num_classes=cfg["num_classes"]
    )
    val_dataset = COCODatasetWrapper(
        hf_dataset["val"],
        transforms=val_transforms,
        output_size=(cfg["output_size"], cfg["output_size"]),
        num_classes=cfg["num_classes"]
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg["batch_size"], 
        shuffle=True, 
        num_workers=cfg["num_workers"],
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg["eval_batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"]
    )

    # Initialize model
    model = CenterNet(
        backbone=cfg["backbone"],
        num_classes=cfg["num_classes"]
    )
    model = model.to(cfg["device"])
    if cfg["compile"]:
        model = torch.compile(model)

    # Initialize loss function
    criterion = CenterNetLoss(
        heatmap_weight=cfg["heatmap_weight"],
        wh_weight=cfg["wh_weight"],
        offset_weight=cfg["offset_weight"]
    )

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["lr"],
        epochs=cfg["epochs"],
        steps_per_epoch=len(train_dataloader)
    )

    global_steps = 0
    best_val_loss = float('inf')
    
    for epoch in range(cfg["epochs"]):
        # Training loop
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{cfg['epochs']}"):
            optimizer.zero_grad()
            
            # Move data to device
            images = batch['image'].to(cfg["device"], non_blocking=True)
            targets = {
                'heatmap': batch['heatmap'].to(cfg["device"], non_blocking=True),
                'wh': batch['wh'].to(cfg["device"], non_blocking=True),
                'offset': batch['offset'].to(cfg["device"], non_blocking=True),
                'reg_mask': batch['reg_mask'].to(cfg["device"], non_blocking=True)
            }
            
            # Forward pass
            predictions = model(images)
            
            # Calculate loss
            loss_dict = criterion(predictions, targets)
            total_loss = loss_dict['loss']
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if cfg["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_value"])
            
            optimizer.step()
            lr_scheduler.step()
            
            # Log training metrics
            if global_steps % cfg["log_interval"] == 0:
                wandb.log({
                    "train/total_loss": loss_dict['loss'].item(),
                    "train/heatmap_loss": loss_dict['heatmap_loss'].item(),
                    "train/wh_loss": loss_dict['wh_loss'].item(),
                    "train/offset_loss": loss_dict['offset_loss'].item(),
                    "train/learning_rate": lr_scheduler.get_last_lr()[0]
                }, step=global_steps)
            
            global_steps += 1

        # Validation loop
        model.eval()
        val_loss_dict = {
            'loss': 0.0,
            'heatmap_loss': 0.0,
            'wh_loss': 0.0,
            'offset_loss': 0.0
        }
        val_count = 0
        
        for batch in tqdm(val_dataloader, desc="Validation"):
            with torch.no_grad():
                images = batch['image'].to(cfg["device"])
                targets = {
                    'heatmap': batch['heatmap'].to(cfg["device"]),
                    'wh': batch['wh'].to(cfg["device"]),
                    'offset': batch['offset'].to(cfg["device"]),
                    'reg_mask': batch['reg_mask'].to(cfg["device"])
                }
                
                predictions = model(images)
                loss_dict = criterion(predictions, targets)
                
                batch_size = images.size(0)
                for k in val_loss_dict.keys():
                    val_loss_dict[k] += loss_dict[k].item() * batch_size
                val_count += batch_size

        # Calculate average validation losses
        for k in val_loss_dict.keys():
            val_loss_dict[k] /= val_count

        # Log validation metrics
        wandb.log({
            "val/total_loss": val_loss_dict['loss'],
            "val/heatmap_loss": val_loss_dict['heatmap_loss'],
            "val/wh_loss": val_loss_dict['wh_loss'],
            "val/offset_loss": val_loss_dict['offset_loss']
        }, step=global_steps)

        # Save checkpoint if validation loss improved
        if val_loss_dict['loss'] < best_val_loss:
            best_val_loss = val_loss_dict['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, "best_model.pt")

        print(f"Epoch {epoch+1} - Val Loss: {val_loss_dict['loss']:.4f}")

    # Save final checkpoint
    torch.save({
        'epoch': cfg["epochs"],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'final_val_loss': val_loss_dict['loss'],
    }, "final_model.pt")

    wandb.finish()

if __name__ == "__main__":
    train_main()
