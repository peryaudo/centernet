import torch
import torch.nn.functional as F
import numpy as np

def gaussian2D(shape, sigma=1):
    """
    Generate a 2D Gaussian kernel.
    
    Args:
        shape (tuple): Height and width of the kernel
        sigma (float): Standard deviation of Gaussian distribution
        
    Returns:
        torch.Tensor: 2D Gaussian kernel
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return torch.from_numpy(h).float()

def get_targets(boxes, labels, output_size, num_classes):
    """
    Generate CenterNet targets from normalized bounding boxes.
    
    Args:
        boxes (torch.Tensor): Normalized bounding boxes in [x1, y1, x2, y2] format, shape [N, 4]
        labels (torch.Tensor): Class labels, shape [N]
        output_size (tuple): Size of output feature map (H, W)
        num_classes (int): Number of object classes
        
    Returns:
        dict: Dictionary containing:
            - 'heatmap': Target heatmap [num_classes, H, W]
            - 'wh': Target width/height [2, H, W]
            - 'offset': Target offset [2, H, W]
            - 'reg_mask': Mask for valid regression targets [H, W]
    """
    H, W = output_size
    device = boxes.device
    
    # Initialize targets
    heatmap = torch.zeros((num_classes, H, W), device=device)
    wh = torch.zeros((2, H, W), device=device)
    offset = torch.zeros((2, H, W), device=device)
    reg_mask = torch.zeros((H, W), device=device)
    
    # Convert normalized coordinates to feature map coordinates
    boxes = boxes.clone()
    boxes[:, [0, 2]] *= W
    boxes[:, [1, 3]] *= H
    
    # Calculate centers, width, and height
    ct_x = (boxes[:, 0] + boxes[:, 2]) / 2
    ct_y = (boxes[:, 1] + boxes[:, 3]) / 2
    box_w = boxes[:, 2] - boxes[:, 0]
    box_h = boxes[:, 3] - boxes[:, 1]
    
    # Get integer center coordinates
    ct_x_int = ct_x.long()
    ct_y_int = ct_y.long()
    
    # Create Gaussian heatmap for each object
    for i in range(len(boxes)):
        # Skip if center is outside feature map
        if not (0 <= ct_x_int[i] < W and 0 <= ct_y_int[i] < H):
            continue
            
        # Calculate Gaussian radius based on object size
        radius = max(0, int(gaussian_radius((box_h[i], box_w[i]), min_overlap=0.3)))

        diameter = 2 * radius + 1
        gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
        
        # Get top-left and bottom-right corners of the Gaussian area
        left = min(ct_x_int[i], radius)
        top = min(ct_y_int[i], radius)
        right = min(W - ct_x_int[i], radius + 1)
        bottom = min(H - ct_y_int[i], radius + 1)
        
        # Get top-left and bottom-right corners in the gaussian kernel
        gaussian_left = radius - left
        gaussian_top = radius - top
        gaussian_right = radius + right
        gaussian_bottom = radius + bottom
        
        # Get the gaussian area in the feature map
        center_y, center_x = ct_y_int[i], ct_x_int[i]
        gaussian_area = gaussian[gaussian_top:gaussian_bottom, gaussian_left:gaussian_right]
        
        # Update heatmap (use maximum in case of overlapping objects)
        heatmap[labels[i],
                center_y - top:center_y + bottom,
                center_x - left:center_x + right] = torch.maximum(
                    heatmap[labels[i],
                           center_y - top:center_y + bottom,
                           center_x - left:center_x + right],
                    gaussian_area
                )
        
        # Update regression targets
        reg_mask[center_y, center_x] = 1
        wh[0, center_y, center_x] = box_w[i]
        wh[1, center_y, center_x] = box_h[i]
        offset[0, center_y, center_x] = ct_x[i] - center_x
        offset[1, center_y, center_x] = ct_y[i] - center_y
    
    return {
        'heatmap': heatmap,
        'wh': wh,
        'offset': offset,
        'reg_mask': reg_mask
    }

def gaussian_radius(det_size, min_overlap=0.3):
    """
    Calculate Gaussian radius based on object size.
    
    Args:
        det_size (tuple): Height and width of the object
        min_overlap (float): Minimum overlap threshold
        
    Returns:
        float: Radius for Gaussian kernel
    """
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 - sq1) / (2 * a1)

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 - sq2) / (2 * a2)

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / (2 * a3)
    
    return min(r1, r2, r3) 