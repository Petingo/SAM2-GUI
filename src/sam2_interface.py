import os
import gradio as gr

from .utils import browse_directory
from .sam2_controller import SAM2Controller

class SAM2Interface:
    """Class that manages the SAM2 Gradio interface"""
    
    def __init__(self, checkpoint_dir, model_cfg, server_port=8890):
        """Initialize the interface with model parameters"""
        self.server_port = server_port
        self.controller = SAM2Controller(checkpoint_dir, model_cfg)
        
        # Interface components
        with gr.Blocks(title="SAM2 Video Segmentation Tool") as self.interface:
            gr.Markdown("# SAM2 Video Segmentation Tool")
            
            # Status message - global status for all tabs
            self.status_msg = gr.Textbox(label="Status", value="Welcome! Start by selecting a video or image directory.")
            
            with gr.Tabs() as self.tabs:
                self._build_input_tab()
                self._build_segment_tab()
                self._build_export_tab()
            
            # Connect event handlers
            self._connect_events()

    def launch(self, share=False):
        """Launch the interface"""
        # check if the interface is built
        if self.interface is None:
            raise ValueError("Interface not built. Call build() before launching.")
        
        self.interface.launch(server_port=self.server_port, share=share)
    
    def _build_input_tab(self):
        """Build the input selection tab"""
        with gr.Tab("1. Input Selection"):
            gr.Markdown("### Select your input source")
            
            with gr.Row():
                with gr.Column():
                    self.input_type = gr.Radio(
                        ["Video File", "Image Directory"], 
                        label="Input Type",
                        value="Video File"
                    )
                    
                    # Video input
                    self.video_input = gr.Video(label="Select Video")
                    
                    # Directory input
                    self.dir_input = gr.Textbox(label="Image Directory Path")
                    self.dir_input.visible = False
                    self.browse_dir = gr.Button("Browse Directory")
                    self.browse_dir.visible = False
                    
                    # Output directory
                    self.output_dir = gr.Textbox(label="Output Directory (for extracted frames)")
                    self.browse_output = gr.Button("Select Output Folder")
                    
                    with gr.Row(visible=True) as self.fps_height_row:
                        self.fps = gr.Slider(1, 30, value=5, step=1, label="Frame Rate (FPS)")
                        self.height = gr.Slider(240, 2160, value=2160, step=60, label="Output Height")
                        
                    # Process button
                    self.process_input = gr.Button("Process Input", variant="primary")
                
                # Preview of the first frame
                with gr.Column():
                    self.preview_img = gr.Image(label="Preview")

    def _build_segment_tab(self):
        """Build the segmentation and tracking tab"""
        with gr.Tab("2. Segment & Track"):
            gr.Markdown("### Select key points for tracking.")
            
            # Simple layout with initialize button first
            with gr.Row():
                self.init_sam_btn = gr.Button("Initialize SAM", variant="primary")
            
            # We'll use this container to show/hide the initialization overlay
            with gr.Column(visible=True) as self.overlay_container:
                # Overlay message when SAM is not initialized
                self.sam_not_initialized_msg = gr.HTML(
                    """<div style="text-align: center; padding: 40px; margin: 20px; 
                          border: 2px dashed #ccc; border-radius: 10px; background-color: #f9f9f9;">
                    <h2 style="color: #666;">⚠️</h2>
                    <p style="font-size: 18px; margin-top: 20px;">
                       Click the "Initialize SAM" button above to start.
                    </p>
                    <p style="font-size: 14px; color: #888; margin-top: 20px;">
                       This will prepare the model for segmentation and tracking.
                    </p>
                    </div>"""
                )
            
            # Main segmentation interface (will be enabled after initialization)
            with gr.Column(visible=False) as self.segmentation_container:
                with gr.Row():
                    # Left Column: Controls
                    with gr.Column(scale=1):
                        # Frame navigation
                        self.frame_slider = gr.Slider(0, 100, value=0, step=1, label="Frame Index")
                        
                        with gr.Row():
                            self.prev_btn = gr.Button("Previous Frame")
                            self.next_btn = gr.Button("Next Frame")
                        
                        gr.Markdown("### Points")
                        # Point type selection
                        with gr.Row():
                            self.pos_btn = gr.Button("Positive Point")
                            self.neg_btn = gr.Button("Negative Point")
                            
                        # Controls
                        with gr.Row():
                            self.clear_btn = gr.Button("Clear Points")
                            self.new_mask_btn = gr.Button("New Mask")
                            
                        gr.Markdown("### Tracking")
                        self.track_btn = gr.Button("Run Tracking", variant="primary")
                    
                    # Middle Column: Image for point selection
                    with gr.Column(scale=2):
                        self.point_image = gr.Image(label="Select Points")
                    
                    # Right Column: Results preview
                    with gr.Column(scale=2):
                        self.result_video = gr.Video(label="Tracking Preview")

    def _build_export_tab(self):
        """Build the export tab"""
        with gr.Tab("3. Export Results"):
            gr.Markdown("### Export segmentation masks")
            
            with gr.Row():
                with gr.Column():
                    # Export location
                    gr.Markdown("#### Select output directory:")
                    self.export_dir = gr.Textbox(label="Export Directory")
                    self.browse_export = gr.Button("Browse Export Location")
                    
                    # Export options
                    gr.Markdown("#### Export Options:")
                    with gr.Row():
                        with gr.Column():
                            self.export_color = gr.Checkbox(label="Color Masks", value=True)
                            self.export_index = gr.Checkbox(label="Index Masks", value=True)
                            self.export_raw = gr.Checkbox(label="Raw Numpy Data", value=False)
                        with gr.Column():
                            self.export_no_bg = gr.Checkbox(label="No Background", value=False)
                            self.export_as_video = gr.Checkbox(label="Export as Video", value=False)
                    
                    # Submit button
                    self.export_btn = gr.Button("Export Masks", variant="primary")
                    
                    # Info about export formats
                    gr.Markdown("""
                    ### Export formats explanation:
                    - **Color Masks**: PNG files with colored visualization of masks
                    - **Index Masks**: PNG files with integer labels for each object
                    - **Raw Numpy Data**: NPY files containing raw mask arrays
                    - **No Background**: Original frames with background removed (set to black)
                    - **Export as Video**: Save as video file instead of separate images
                    """)
                
                # Preview column
                with gr.Column():
                    self.export_preview = gr.Image(label="Mask Preview")
                    self.export_result = gr.Textbox(label="Export Status")

    def _connect_events(self):
        """Connect all event handlers to UI elements"""
        # Input tab events
        self.input_type.change(
            self._update_browse_visibility,
            inputs=[self.input_type],
            outputs=[self.video_input, self.dir_input, self.browse_dir, 
                     self.output_dir, self.browse_output, self.fps_height_row]
        )
        
        self.browse_dir.click(
            browse_directory,
            outputs=[self.dir_input]
        )
        
        self.browse_output.click(
            browse_directory,
            outputs=[self.output_dir]
        )
        
        self.process_input.click(
            self._process_input_fn,
            inputs=[self.input_type, self.video_input, self.dir_input, 
                   self.output_dir, self.fps, self.height],
            outputs=[self.status_msg, self.preview_img, self.frame_slider, self.output_dir]
        )
        
        # Segmentation tab events
        self.init_sam_btn.click(
            self._initialize_sam,
            inputs=[self.frame_slider],
            outputs=[self.point_image, self.status_msg, self.overlay_container, self.segmentation_container]
        )
        
        self.point_image.select(
            self._add_point_from_click,
            inputs=[self.frame_slider, self.point_image],
            outputs=[self.point_image]
        )
        
        self.frame_slider.change(
            self._safe_update_frame,
            inputs=[self.frame_slider],
            outputs=[self.point_image, self.status_msg]
        )
        
        self.prev_btn.click(
            lambda idx: max(0, idx-1),
            inputs=[self.frame_slider],
            outputs=[self.frame_slider]
        )
        
        self.next_btn.click(
            self._go_to_next_frame,
            inputs=[self.frame_slider],
            outputs=[self.frame_slider]
        )
        
        self.track_btn.click(
            self._run_tracker,
            outputs=[self.result_video, self.status_msg]
        )
        
        self.pos_btn.click(
            lambda: self.controller.set_positive() if self._check_sam_initialized() else "Please initialize SAM first",
            outputs=[self.status_msg]
        )
        
        self.neg_btn.click(
            lambda: self.controller.set_negative() if self._check_sam_initialized() else "Please initialize SAM first",
            outputs=[self.status_msg]
        )
        
        self.clear_btn.click(
            self._clear_points,
            outputs=[self.status_msg, self.point_image]
        )
        
        self.new_mask_btn.click(
            self._new_mask,
            outputs=[self.point_image, self.status_msg]
        )
        
        # Export tab events
        self.browse_export.click(
            browse_directory,
            outputs=[self.export_dir]
        )
        
        self.tabs.change(
            self._show_mask_preview,
            inputs=None,
            outputs=[self.export_preview, self.status_msg]
        )
        
        self.export_btn.click(
            self._export_masks,
            inputs=[self.export_dir, self.export_color, self.export_index, 
                   self.export_raw, self.export_no_bg, self.export_as_video],
            outputs=[self.export_result]
        )

    # Event handler implementations
    def _update_browse_visibility(self, input_choice):
        """Update visibility of browse UI elements"""
        if input_choice == "Video File":
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
        else:
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    def _process_input_fn(self, input_choice, video_path, dir_path, out_dir, fps_val, height_val):
        """Process the input (video or directory)"""
        if input_choice == "Video File" and video_path:
            # Extract frames from video
            if not out_dir:
                out_dir = os.path.join(os.path.dirname(video_path), "frames")
                
            result = self.controller.extract_frames(video_path, out_dir, fps_val, height_val)
            num_frames, _ = self.controller.set_img_dir(out_dir)
            
            # Update frame slider max value
            slider_update = gr.Slider(maximum=num_frames-1 if num_frames > 0 else 0)
            
            # Get preview of first frame
            if num_frames > 0:
                preview, _ = self.controller.set_input_image(0)
                return result, preview, slider_update, out_dir
            else:
                return result, None, slider_update, out_dir
            
        elif input_choice == "Image Directory" and dir_path:
            # Set image directory
            num_frames, result = self.controller.set_img_dir(dir_path)
            
            # Update frame slider max value
            slider_update = gr.Slider(maximum=num_frames-1 if num_frames > 0 else 0)
            
            # Get preview of first frame
            if num_frames > 0:
                preview, _ = self.controller.set_input_image(0)
                return result, preview, slider_update, dir_path
            else:
                return result, None, slider_update, dir_path
        else:
            return "Please select a video file or image directory", None, gr.Slider(), ""
    
    def _check_sam_initialized(self):
        """Check if SAM is initialized by checking container visibility"""
        return self.segmentation_container.visible
    
    def _initialize_sam(self, frame_idx):
        """Initialize SAM model and update UI"""
        image, message = self.controller.initialize_sam(frame_idx)
        success = "initialized" in message.lower() and image is not None
        
        if success:
            # Show the segmentation container and hide the overlay
            self.segmentation_container.visible = True
            self.overlay_container.visible = False
            return (
                image,  # point_image
                message,  # status_msg
                gr.update(visible=False),  # overlay_container
                gr.update(visible=True)  # segmentation_container
            )
        else:
            # Keep overlay visible, segmentation hidden
            self.segmentation_container.visible = False
            self.overlay_container.visible = True
            return (
                None,  # point_image
                message or "SAM initialization failed. Please try again.",  # status_msg
                gr.update(visible=True),  # overlay_container
                gr.update(visible=False)  # segmentation_container
            )
    
    def _add_point_from_click(self, frame_idx, img, evt: gr.SelectData):
        """Add a point when the image is clicked"""
        # Only process clicks when segmentation container is visible
        if self.segmentation_container.visible:
            i = evt.index[1]  # y coordinate
            j = evt.index[0]  # x coordinate
            return self.controller.add_point(frame_idx, i, j)
        else:
            return img  # Just return the input image if not initialized
    
    def _safe_update_frame(self, idx):
        """Update the displayed frame safely"""
        if not self._check_sam_initialized():
            return None, "Please initialize SAM first"
        return self.controller.set_input_image(int(idx))
    
    def _go_to_next_frame(self, idx):
        """Go to the next frame"""
        # Get the current maximum value directly from the controller
        max_idx = len(self.controller.img_paths) - 1 if self.controller.img_paths else 0
        return min(int(idx) + 1, max_idx)
    
    def _run_tracker(self):
        """Run the video tracker"""
        if not self._check_sam_initialized():
            return None, "Please initialize SAM first"
        return self.controller.run_tracker()
    
    def _clear_points(self):
        """Clear all points"""
        if not self._check_sam_initialized():
            return "Please initialize SAM first", None
        return (self.controller.clear_points()[2], 
                self.controller.set_input_image(self.controller.frame_index)[0])
    
    def _new_mask(self):
        """Create a new mask"""
        if not self._check_sam_initialized():
            return None, "Please initialize SAM first"
        return self.controller.add_new_mask()
    
    def _show_mask_preview(self):
        """Show a preview of a mask if available"""
        if self.controller.color_masks_all and len(self.controller.color_masks_all) > 0:
            return self.controller.color_masks_all[0], "Mask preview loaded. Ready to export."
        return None, "No masks available. Run tracking first before exporting."
    
    def _export_masks(self, export_dir, color_mask, index_mask, raw_mask, images_no_bg, as_video):
        """Export masks with the selected options"""
        export_options = {
            'color_mask': color_mask,
            'index_mask': index_mask,
            'raw_mask': raw_mask,
            'images_no_bg': images_no_bg,
            'as_video': as_video
        }
        return self.controller.save_masks(export_dir, export_options)


# def create_interface(checkpoint_dir, model_cfg):
#     """Create the Gradio interface for SAM2 Video Segmentation Tool"""
#     interface_manager = SAM2Interface(checkpoint_dir, model_cfg)
#     return interface_manager.build()
