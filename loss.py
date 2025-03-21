import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterNetLoss(nn.Module):
    def __init__(self, heatmap_weight=1.0, wh_weight=0.1, offset_weight=1.0, focal_alpha=2, focal_beta=4):
        """
        CenterNet Loss combining focal loss for heatmap and L1 losses for size and offset.
        
        Args:
            heatmap_weight (float): Weight for the focal loss on heatmap
            wh_weight (float): Weight for the L1 loss on width/height prediction
            offset_weight (float): Weight for the L1 loss on offset prediction
            focal_alpha (int): Alpha parameter for focal loss
            focal_beta (int): Beta parameter for focal loss
        """
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.wh_weight = wh_weight
        self.offset_weight = offset_weight
        self.focal_alpha = focal_alpha
        self.focal_beta = focal_beta

    def focal_loss(self, pred, target):
        """
        Focal loss for heatmap prediction.
        
        Args:
            pred (torch.Tensor): Predicted heatmap [B, C, H, W]
            target (torch.Tensor): Target heatmap [B, C, H, W]
        """
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()
        
        neg_weights = torch.pow(1 - target, self.focal_beta)
        
        loss = 0
        
        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.focal_alpha) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.focal_alpha) * neg_weights * neg_inds
        
        num_pos = pos_inds.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        
        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
            
        return loss
    
    def reg_l1_loss(self, pred, target, mask):
        """
        L1 regression loss for width/height and offset prediction.
        
        Args:
            pred (torch.Tensor): Predicted values
            target (torch.Tensor): Target values
            mask (torch.Tensor): Mask for valid predictions
        """
        mask = mask.unsqueeze(1).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss

    def forward(self, predictions, targets):
        """
        Forward pass computing the total loss.
        
        Args:
            predictions (dict): Dictionary containing:
                - 'heatmap': Predicted heatmap [B, C, H, W]
                - 'wh': Predicted width/height [B, 2, H, W]
                - 'offset': Predicted offset [B, 2, H, W]
            targets (dict): Dictionary containing:
                - 'heatmap': Target heatmap [B, C, H, W]
                - 'wh': Target width/height [B, 2, H, W]
                - 'offset': Target offset [B, 2, H, W]
                - 'reg_mask': Mask for valid regression targets [B, H, W]
        """
        heatmap_loss = self.focal_loss(predictions['heatmap'], targets['heatmap'])
        wh_loss = self.reg_l1_loss(predictions['wh'], targets['wh'], targets['reg_mask'])
        offset_loss = self.reg_l1_loss(predictions['offset'], targets['offset'], targets['reg_mask'])
        
        total_loss = (
            self.heatmap_weight * heatmap_loss +
            self.wh_weight * wh_loss +
            self.offset_weight * offset_loss
        )
        
        return {
            'loss': total_loss,
            'heatmap_loss': heatmap_loss,
            'wh_loss': wh_loss,
            'offset_loss': offset_loss
        }

if __name__ == "__main__":
    # Create dummy batch: batch_size=2, num_classes=80, feature_map=128x128
    batch_size, num_classes = 2, 80
    feature_size = 128
    
    # Create dummy predictions (output from CenterNet model)
    predictions = {
        'heatmap': torch.sigmoid(torch.randn(batch_size, num_classes, feature_size, feature_size)),  # Already sigmoided
        'wh': torch.randn(batch_size, 2, feature_size, feature_size),  # Raw width/height predictions
        'offset': torch.randn(batch_size, 2, feature_size, feature_size)  # Raw offset predictions
    }
    
    # Create dummy targets
    # For heatmap: create sparse target with a few positive samples
    heatmap_target = torch.zeros(batch_size, num_classes, feature_size, feature_size)
    heatmap_target[:, :, 64, 64] = 1  # Set center points
    
    # Create dummy regression targets and mask
    targets = {
        'heatmap': heatmap_target,
        'wh': torch.randn(batch_size, 2, feature_size, feature_size),
        'offset': torch.randn(batch_size, 2, feature_size, feature_size),
        'reg_mask': torch.zeros(batch_size, feature_size, feature_size)
    }
    # Set regression mask to 1 around positive samples
    targets['reg_mask'][:, 63:66, 63:66] = 1
    
    # Initialize loss function
    criterion = CenterNetLoss()
    
    # Compute loss
    loss_dict = criterion(predictions, targets)
    
    # Print results
    print("Input shapes:")
    print(f"- Heatmap prediction: {predictions['heatmap'].shape}")
    print(f"- Width/Height prediction: {predictions['wh'].shape}")
    print(f"- Offset prediction: {predictions['offset'].shape}")
    print("\nLoss components:")
    for k, v in loss_dict.items():
        print(f"- {k}: {v.item():.4f}") 