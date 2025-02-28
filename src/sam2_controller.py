import os
import cv2
import subprocess
import numpy as np
import torch
from loguru import logger as guru

from .utils import *

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    print("Warning: sam2 module not found. Make sure SAM2 is properly installed.")


class SAM2Controller:
    def __init__(self, checkpoint_dir, model_cfg):
        self.checkpoint_dir = checkpoint_dir
        self.model_cfg = model_cfg
        self.sam_model = None
        self.tracker = None

        self.selected_points = []
        self.selected_labels = []
        self.cur_label_val = 1.0  # Positive points by default

        self.frame_index = 0
        self.image = None
        self.cur_mask_idx = 0
        # Store masks and logits
        self.cur_masks = {}
        self.cur_logits = {}
        self.index_masks_all = []
        self.color_masks_all = []

        self.img_dir = ""
        self.img_paths = []
        self.init_sam_model()

        # Enable hardware acceleration if available
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        

    # ... existing methods remain the same, only the class name changes ...
    # Just changing the class name, all methods stay the same
    def init_sam_model(self):
        if self.sam_model is None:
            try:
                self.sam_model = build_sam2_video_predictor(self.model_cfg, self.checkpoint_dir)
                guru.info(f"Loaded model checkpoint {self.checkpoint_dir}")
            except Exception as e:
                guru.error(f"Failed to load SAM2 model: {str(e)}")

    def clear_points(self):
        self.selected_points.clear()
        self.selected_labels.clear()
        message = "Cleared points, select new points to update mask"
        return None, None, message

    def add_new_mask(self):
        self.cur_mask_idx += 1
        self.clear_points()
        message = f"Creating new mask with index {self.cur_mask_idx}"
        return None, message

    def make_index_mask(self, masks):
        if not masks:
            return None
        
        idcs = list(masks.keys())
        idx_mask = masks[idcs[0]].astype("uint8")
        for i in idcs:
            mask = masks[i]
            idx_mask[mask] = i + 1
        return idx_mask

    def _clear_image(self):
        """Clears image and all masks/logits for that image"""
        self.image = None
        self.cur_mask_idx = 0
        self.frame_index = 0
        self.cur_masks = {}
        self.cur_logits = {}
        self.index_masks_all = []
        self.color_masks_all = []
        self.selected_points = []
        self.selected_labels = []

    def reset(self):
        self._clear_image()
        if hasattr(self, 'inference_state'):
            self.sam_model.reset_state(self.inference_state)
        return "Reset completed. Select new frames or points."

    def set_img_dir(self, img_dir: str):
        """Set the directory containing the frames"""
        self._clear_image()
        self.img_dir = img_dir
        
        if not os.path.isdir(img_dir):
            return 0, "Invalid directory path"
        
        self.img_paths = [
            os.path.join(img_dir, p) for p in sorted(os.listdir(img_dir)) 
            if is_image(p)
        ]
        
        return len(self.img_paths), f"Found {len(self.img_paths)} images in {img_dir}"

    def set_input_image(self, i: int = 0):
        """Set the current frame to the given index"""
        if not self.img_paths or i < 0 or i >= len(self.img_paths):
            return self.image, f"Invalid frame index: {i}"
        
        self.selected_points = []
        self.selected_labels = []
        self.frame_index = i
        
        try:
            image = cv2.imread(self.img_paths[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image = image
            return image, f"Loaded frame {i+1}/{len(self.img_paths)}"
        except Exception as e:
            return None, f"Error loading image: {str(e)}"

    def extract_frames(self, video_path, output_dir, fps=5, height=540):
        """Extract frames from a video file"""
        if not video_path or not os.path.isfile(video_path):
            return "Invalid video path"
            
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Use ffmpeg to extract frames
            cmd = (
                f"ffmpeg -i {video_path} "
                f"-vf 'scale=-1:{height},fps={fps}' "
                f"{output_dir}/%05d.jpg"
            )
            subprocess.call(cmd, shell=True)
            
            # Set the new directory as the current image directory
            num_images, message = self.set_img_dir(output_dir)
            return f"Extracted {num_images} frames to {output_dir}"
            
        except Exception as e:
            return f"Error extracting frames: {str(e)}"

    def initialize_sam(self, frame_idx=0):
        """Initialize SAM for the current frame"""
        if not self.img_paths:
            return None, "No images loaded"
            
        if frame_idx < 0 or frame_idx >= len(self.img_paths):
            frame_idx = 0
        
        try:
            # Initialize SAM state
            self.inference_state = self.sam_model.init_state(video_path=os.path.dirname(self.img_paths[0]))
            self.sam_model.reset_state(self.inference_state)
            
            # Set the current frame
            image, message = self.set_input_image(frame_idx)
            
            return image, "SAM initialized. Click points to update the mask."
        except Exception as e:
            return None, f"Failed to initialize SAM: {str(e)}"

    def set_positive(self):
        """Set point selection mode to positive"""
        self.cur_label_val = 1.0
        return "Selecting positive points. Click on object to segment."

    def set_negative(self):
        """Set point selection mode to negative"""
        self.cur_label_val = 0.0
        return "Selecting negative points. Click to exclude regions."

    def add_point(self, frame_idx, i, j):
        """Add a point at the clicked position"""
        guru.debug(f"Adding point at coordinates: ({j}, {i})")
        
        if not hasattr(self, 'inference_state'):
            guru.warning("SAM not initialized - please initialize SAM first")
            return self.image
        
        # Get the current image if it's None
        if self.image is None:
            if len(self.img_paths) > 0 and 0 <= self.frame_index < len(self.img_paths):
                try:
                    self.image = cv2.cvtColor(cv2.imread(self.img_paths[self.frame_index]), cv2.COLOR_BGR2RGB)
                    guru.info(f"Loaded image from path: {self.img_paths[self.frame_index]}")
                except Exception as e:
                    guru.error(f"Failed to load image: {str(e)}")
                    return None
            else:
                guru.warning("No image available - please load frames first")
                return None
        
        # Add the point (j is x, i is y)
        self.selected_points.append([j, i])
        self.selected_labels.append(self.cur_label_val)
        
        # Get the mask for the points
        try:
            masks = self.get_sam_mask(
                frame_idx,
                np.array(self.selected_points, dtype=np.float32),
                np.array(self.selected_labels, dtype=np.float32)
            )
            
            if masks:
                mask = self.make_index_mask(masks)
                if mask is not None:
                    # Get a colored version of the mask
                    palette = get_hls_palette(self.cur_mask_idx + 2)  # +2 for background and first mask
                    color_mask = palette[mask]
                    
                    # Blend with image
                    vis_img = compose_img_mask(self.image, color_mask)
                    
                    # Draw points
                    vis_img = draw_points(vis_img, self.selected_points, self.selected_labels)
                    
                    return vis_img
            
            # If no mask was generated, just show points
            vis_img = draw_points(self.image.copy(), self.selected_points, self.selected_labels)
            return vis_img
            
        except Exception as e:
            guru.error(f"Error adding point: {str(e)}")
            return self.image

    def get_sam_mask(self, frame_idx, input_points, input_labels):
        """Get SAM mask for the given points"""
        if self.sam_model is None:
            return {}
        
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
            try:
                _, out_obj_ids, out_mask_logits = self.sam_model.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=self.cur_mask_idx,
                    points=input_points,
                    labels=input_labels,
                )
                
                return {
                    out_obj_id: (out_mask_logits[i] > 0.0).squeeze().cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            except Exception as e:
                guru.error(f"Error in get_sam_mask: {str(e)}")
                return {}

    def run_tracker(self):
        """Propagate masks through the video sequence"""
        if not hasattr(self, 'inference_state') or not self.img_paths:
            return None, "No frames or points selected"
        
        if not self.selected_points:
            return None, "Please add at least one point before tracking"
        
        try:
            # Read all images
            images = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in self.img_paths]
            
            # Run the tracker
            video_segments = {}
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                for out_frame_idx, out_obj_ids, out_mask_logits in self.sam_model.propagate_in_video(
                    self.inference_state, start_frame_idx=0
                ):
                    masks = {
                        out_obj_id: (out_mask_logits[i] > 0.0).squeeze().cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                    video_segments[out_frame_idx] = masks
            
            # Create index masks for all frames
            self.index_masks_all = [self.make_index_mask(v) for k, v in sorted(video_segments.items())]
            
            # Create colorized output
            output_frames, self.color_masks_all = colorize_masks(images, self.index_masks_all)
            
            # Save video to a temporary file
            import tempfile
            from pathlib import Path
            import imageio.v2 as iio
            
            temp_dir = Path(tempfile.gettempdir())
            out_path = str(temp_dir / "tracked_video.mp4")
            
            # Save frames as video
            iio.mimwrite(out_path, output_frames, fps=10)
            
            return out_path, f"Tracking complete. Processed {len(output_frames)} frames."
            
        except Exception as e:
            guru.error(f"Error in run_tracker: {str(e)}")
            return None, f"Error running tracker: {str(e)}"

    def save_masks(self, output_dir, export_options):
        """Save masks to the specified directory with various export options"""
        if not output_dir or not self.index_masks_all:
            return "No masks to save or invalid directory"
            
        try:
            # Get export options
            export_color = export_options.get('color_mask', True)
            export_index = export_options.get('index_mask', True)
            export_raw = export_options.get('raw_mask', False)
            export_no_bg = export_options.get('images_no_bg', False)
            export_video = export_options.get('as_video', False)
            
            # Create main output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Create subdirectories for different export types
            if export_color:
                os.makedirs(os.path.join(output_dir, "color_masks"), exist_ok=True)
            if export_index:
                os.makedirs(os.path.join(output_dir, "index_masks"), exist_ok=True)
            if export_raw:
                os.makedirs(os.path.join(output_dir, "raw_masks"), exist_ok=True)
            if export_no_bg:
                os.makedirs(os.path.join(output_dir, "no_background"), exist_ok=True)
                
            export_count = 0
            
            # Get original images for no-background export
            if export_no_bg:
                images = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in self.img_paths]
                images_no_bg = remove_background(images, self.index_masks_all)
                
            # If exporting as video
            if export_video:
                import tempfile
                from pathlib import Path
                import imageio.v2 as iio
                
                temp_dir = Path(tempfile.gettempdir())
                
                # Export color mask video
                if export_color and self.color_masks_all:
                    out_path = os.path.join(output_dir, "color_masks", "color_masks.mp4")
                    iio.mimwrite(out_path, self.color_masks_all, fps=10)
                    export_count += 1
                
                # Export no background video
                if export_no_bg and images_no_bg:
                    out_path = os.path.join(output_dir, "no_background", "no_background.mp4")
                    iio.mimwrite(out_path, images_no_bg, fps=10)
                    export_count += 1
            
            # Export as individual files
            else:
                # Save each mask type as separate files
                for i, (color_mask, index_mask) in enumerate(zip(self.color_masks_all, self.index_masks_all)):
                    # Get base filename from original frame
                    base_name = os.path.splitext(os.path.basename(self.img_paths[i]))[0]
                    
                    # Save color mask as PNG
                    if export_color:
                        color_path = os.path.join(output_dir, "color_masks", f"{base_name}.png")
                        cv2.imwrite(color_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
                        export_count += 1
                    
                    # Save index mask as PNG
                    if export_index:
                        idx_path = os.path.join(output_dir, "index_masks", f"{base_name}.png")
                        cv2.imwrite(idx_path, index_mask)
                        export_count += 1
                    
                    # Save raw mask data as NPY
                    if export_raw:
                        np_path = os.path.join(output_dir, "raw_masks", f"{base_name}.npy")
                        np.save(np_path, index_mask)
                        export_count += 1
                        
                    # Save no-background image as PNG
                    if export_no_bg:
                        nobg_path = os.path.join(output_dir, "no_background", f"{base_name}.png")
                        cv2.imwrite(nobg_path, cv2.cvtColor(images_no_bg[i], cv2.COLOR_RGB2BGR))
                        export_count += 1
            
            return f"Exported {export_count} files to {output_dir}"
            
        except Exception as e:
            guru.error(f"Error saving masks: {str(e)}")
            return f"Error saving masks: {str(e)}"
