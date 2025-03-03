import os
import cv2
import numpy as np
import colorsys
from tqdm import tqdm
import concurrent.futures

def is_image(filepath):
    """Check if a file is an image based on its extension"""
    ext = os.path.splitext(filepath.lower())[1]
    return ext in ['.jpg', '.jpeg', '.png', '.bmp']


def get_hls_palette(n_colors: int, lightness: float = 0.5, saturation: float = 0.7):
    """Generate a color palette with distinct colors"""
    hues = np.linspace(0, 1, int(n_colors) + 1)[1:-1]
    palette = [(0.0, 0.0, 0.0)] + [
        colorsys.hls_to_rgb(h_i, lightness, saturation) for h_i in hues
    ]
    return (255 * np.asarray(palette)).astype("uint8")


def remove_background(images, mask, bg_idx=0):
    """Remove background from images using a mask"""
    if not images or not mask:
        return []
    
    images_without_background = []
    for img, m in zip(images, mask):
        if m is None:
            images_without_background.append(img)
            continue
        
        # Remove background
        out_f = img.copy()
        out_f[m == bg_idx] = 0
        images_without_background.append(out_f)
    
    return images_without_background


def colorize_masks(images, index_masks, fac: float = 0.5):
    """Colorize masks and blend with original images using multi-threading"""
    if not index_masks or not images:
        return [], []
    
    max_idx = max([m.max() for m in index_masks if m is not None]) if index_masks else 0
    palette = get_hls_palette(max_idx + 1)
    
    color_masks = [None] * len(images)
    out_frames = [None] * len(images)
    
    def process_single(idx, img, mask):
        """Process a single image-mask pair"""
        if mask is None:
            return idx, np.zeros_like(img), img
            
        clr_mask = palette[mask.astype("int")]
        
        # Compose image with mask
        out_f = fac * img / 255 + (1 - fac) * clr_mask / 255
        out_u = (255 * out_f).astype("uint8")
        
        return idx, clr_mask, out_u
    
    # Determine optimal number of workers
    num_workers = min(os.cpu_count() or 4, len(images))
    
    with tqdm(total=len(images), desc="Colorizing masks") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(process_single, i, img, mask) 
                      for i, (img, mask) in enumerate(zip(images, index_masks))]
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                idx, clr_mask, out_u = future.result()
                color_masks[idx] = clr_mask
                out_frames[idx] = out_u
                pbar.update(1)
    
    return out_frames, color_masks


def draw_points(img, points, labels):
    """Draw points on the image"""
    out = img.copy()
    for p, label in zip(points, labels):
        x, y = int(p[0]), int(p[1])
        color = (0, 255, 0) if label == 1.0 else (255, 0, 0)
        out = cv2.circle(out, (x, y), 10, color, -1)
    return out


def compose_img_mask(img, color_mask, fac: float = 0.5):
    """Blend image with mask"""
    out_f = fac * img / 255 + (1 - fac) * color_mask / 255
    out_u = (255 * out_f).astype("uint8")
    return out_u


def browse_directory():
    """Open a file dialog to browse for a directory"""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()
    root.destroy()
    return folder_path
