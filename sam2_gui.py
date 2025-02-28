import torch
import os
import cv2
import datetime
import subprocess
import numpy as np
import gradio as gr
import colorsys

from loguru import logger as guru

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    print("Warning: sam2 module not found. Make sure SAM2 is properly installed.")


class SAM2Wizard:
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

    def save_masks(self, output_dir):
        """Save masks to the specified directory"""
        if not output_dir or not self.index_masks_all:
            return "No masks to save or invalid directory"
            
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save both colorized masks and raw masks
            for i, (color_mask, index_mask) in enumerate(zip(self.color_masks_all, self.index_masks_all)):
                # Get base filename from original frame
                base_name = os.path.splitext(os.path.basename(self.img_paths[i]))[0]
                
                # Save color mask as PNG
                color_path = os.path.join(output_dir, f"{base_name}_color.png")
                cv2.imwrite(color_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
                
                # Save index mask as PNG
                idx_path = os.path.join(output_dir, f"{base_name}_index.png") 
                cv2.imwrite(idx_path, index_mask)
                
                # Save raw mask data as NPY
                np_path = os.path.join(output_dir, f"{base_name}_mask.npy")
                np.save(np_path, index_mask)
            
            return f"Saved {len(self.index_masks_all)} masks to {output_dir}"
            
        except Exception as e:
            return f"Error saving masks: {str(e)}"


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


def colorize_masks(images, index_masks, fac: float = 0.5):
    """Colorize masks and blend with original images"""
    if not index_masks or not images:
        return [], []
    
    max_idx = max([m.max() for m in index_masks if m is not None]) if index_masks else 0
    palette = get_hls_palette(max_idx + 1)
    
    color_masks = []
    out_frames = []
    
    for img, mask in zip(images, index_masks):
        if mask is None:
            color_masks.append(np.zeros_like(img))
            out_frames.append(img)
            continue
            
        clr_mask = palette[mask.astype("int")]
        color_masks.append(clr_mask)
        
        # Compose image with mask
        out_f = fac * img / 255 + (1 - fac) * clr_mask / 255
        out_u = (255 * out_f).astype("uint8")
        out_frames.append(out_u)
        
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

def create_wizard_interface(checkpoint_dir, model_cfg):
    """Create the Gradio interface for SAM2 Wizard"""
    wizard = SAM2Wizard(checkpoint_dir, model_cfg)
    
    with gr.Blocks(title="SAM2 Video Segmentation Wizard") as interface:
        gr.Markdown("# SAM2 Video Segmentation Wizard")
        gr.Markdown("### Step-by-step segmentation workflow")
        
        # Status message
        status_msg = gr.Textbox(label="Status", value="Welcome! Start by selecting a video or image directory.")
        
        with gr.Tabs() as tabs:
            # Step 1: Input Selection
            with gr.Tab("1. Input Selection"):
                gr.Markdown("### Select your input source")
                
                with gr.Row():
                    with gr.Column():
                        input_type = gr.Radio(
                            ["Video File", "Image Directory"], 
                            label="Input Type",
                            value="Video File"
                        )
                        
                        # Video input
                        video_input = gr.Video(label="Select Video")
                        
                        with gr.Row():
                            fps = gr.Slider(1, 30, value=5, step=1, label="Frame Rate (FPS)")
                            height = gr.Slider(240, 2160, value=1080, step=60, label="Output Height")
                        
                        # Directory input
                        dir_input = gr.Textbox(label="Image Directory Path")
                        browse_dir = gr.Button("Browse Directory")
                        
                        # Output directory
                        output_dir = gr.Textbox(label="Output Directory (for extracted frames)")
                        browse_output = gr.Button("Browse Output")
                        
                        # Process button
                        process_input = gr.Button("Process Input", variant="primary")
                    
                    # Preview of the first frame or input info
                    with gr.Column():
                        input_info = gr.Textbox(label="Input Information")
                        preview_img = gr.Image(label="Preview")
            
            # Step 2: Combined Point Selection, Tracking & Export
            with gr.Tab("2. Segment & Export"):
                gr.Markdown("### Select points, track objects, and export results")
                
                with gr.Row():
                    # Left Column: Controls
                    with gr.Column(scale=1):
                        # Frame navigation
                        frame_slider = gr.Slider(0, 100, value=0, step=1, label="Frame Index")
                        
                        with gr.Row():
                            prev_btn = gr.Button("Previous Frame")
                            next_btn = gr.Button("Next Frame")
                        
                        gr.Markdown("### Points")
                        # Point type selection
                        with gr.Row():
                            pos_btn = gr.Button("Positive Point", variant="success")
                            neg_btn = gr.Button("Negative Point", variant="stop")
                            
                        # Controls
                        with gr.Row():
                            init_sam_btn = gr.Button("Initialize SAM")
                            clear_btn = gr.Button("Clear Points")
                            new_mask_btn = gr.Button("New Mask")
                            
                        gr.Markdown("### Tracking")
                        track_btn = gr.Button("Run Tracking", variant="primary")
                        
                        gr.Markdown("### Export")
                        export_dir = gr.Textbox(label="Export Directory")
                        browse_export = gr.Button("Browse Export Location")
                        export_btn = gr.Button("Export Masks")
                    
                    # Middle Column: Image for point selection
                    with gr.Column(scale=2):
                        point_image = gr.Image(label="Select Points")
                    
                    # Right Column: Results preview
                    with gr.Column(scale=2):
                        result_video = gr.Video(label="Tracking Preview")
        
        # Event handlers
        # Step 1: Input Processing
        def update_browse_visibility(input_choice):
            if input_choice == "Video File":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)
        
        input_type.change(
            update_browse_visibility,
            inputs=[input_type],
            outputs=[video_input, dir_input]
        )
        
        def browse_directory():
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            folder_path = filedialog.askdirectory()
            root.destroy()
            return folder_path
        
        browse_dir.click(
            browse_directory,
            outputs=[dir_input]
        )
        
        browse_output.click(
            browse_directory,
            outputs=[output_dir]
        )
        
        def process_input_fn(input_choice, video_path, dir_path, out_dir, fps_val, height_val):
            if input_choice == "Video File" and video_path:
                # Extract frames from video
                if not out_dir:
                    out_dir = os.path.join(os.path.dirname(video_path), "frames")
                    
                result = wizard.extract_frames(video_path, out_dir, fps_val, height_val)
                num_frames, _ = wizard.set_img_dir(out_dir)
                
                # Update frame slider max value
                slider_update = gr.Slider(maximum=num_frames-1 if num_frames > 0 else 0)
                
                # Get preview of first frame
                if num_frames > 0:
                    preview, _ = wizard.set_input_image(0)
                    return result, preview, slider_update, out_dir
                else:
                    return result, None, slider_update, out_dir
                
            elif input_choice == "Image Directory" and dir_path:
                # Set image directory
                num_frames, result = wizard.set_img_dir(dir_path)
                
                # Update frame slider max value
                slider_update = gr.Slider(maximum=num_frames-1 if num_frames > 0 else 0)
                
                # Get preview of first frame
                if num_frames > 0:
                    preview, _ = wizard.set_input_image(0)
                    return result, preview, slider_update, dir_path
                else:
                    return result, None, slider_update, dir_path
            else:
                return "Please select a video file or image directory", None, gr.Slider(), ""
        
        process_input.click(
            process_input_fn,
            inputs=[input_type, video_input, dir_input, output_dir, fps, height],
            outputs=[input_info, preview_img, frame_slider, output_dir]
        )
        
        # Step 2: Point Selection
        def update_frame(idx):
            image, msg = wizard.set_input_image(int(idx))
            return image, msg
        
        frame_slider.change(
            update_frame,
            inputs=[frame_slider],
            outputs=[point_image, status_msg]
        )
        
        prev_btn.click(
            lambda idx: max(0, idx-1),
            inputs=[frame_slider],
            outputs=[frame_slider]
        )
        
        # Fix the next button click handler
        def go_to_next_frame(idx):
            # Get the current maximum value directly from the wizard
            max_idx = len(wizard.img_paths) - 1 if wizard.img_paths else 0
            return min(int(idx) + 1, max_idx)
        
        next_btn.click(
            go_to_next_frame,
            inputs=[frame_slider],
            outputs=[frame_slider]
        )
        
        init_sam_btn.click(
            wizard.initialize_sam,
            inputs=[frame_slider],
            outputs=[point_image, status_msg]
        )
        
        pos_btn.click(
            lambda: wizard.set_positive(),
            outputs=[status_msg]
        )
        
        neg_btn.click(
            lambda: wizard.set_negative(),
            outputs=[status_msg]
        )
        
        def get_select_coords(frame_idx, img, evt: gr.SelectData):
            """Handle click events on the image"""
            i = evt.index[1]  # y coordinate
            j = evt.index[0]  # x coordinate
            guru.debug(f"Click at coordinates: ({j}, {i})")
            
            # Add point and return updated image
            return wizard.add_point(frame_idx, i, j)
        
        point_image.select(
            get_select_coords,
            inputs=[frame_slider, point_image],
            outputs=[point_image]
        )
        
        clear_btn.click(
            lambda: (wizard.clear_points()[2], wizard.set_input_image(wizard.frame_index)[0]),
            outputs=[status_msg, point_image]
        )
        
        new_mask_btn.click(
            wizard.add_new_mask,
            outputs=[point_image, status_msg]
        )
        
        # Step 3: Tracking and Export
        track_btn.click(
            wizard.run_tracker,
            outputs=[result_video, status_msg]
        )
        
        browse_export.click(
            browse_directory,
            outputs=[export_dir]
        )
        
        export_btn.click(
            wizard.save_masks,
            inputs=[export_dir],
            outputs=[status_msg]
        )
        
    return interface

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--checkpoint_dir", type=str, default="../sam2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--model_cfg", type=str, default="../sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
    args = parser.parse_args()
    
    # Enable hardware acceleration if available
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Create and launch the interface
    interface = create_wizard_interface(args.checkpoint_dir, args.model_cfg)
    interface.launch(server_port=args.port, share=False)