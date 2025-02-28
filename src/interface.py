import os
import gradio as gr

from .utils import browse_directory
from .sam2_controller import SAM2Controller

def create_interface(checkpoint_dir, model_cfg):
    """Create the Gradio interface for SAM2 Video Segmentation Tool"""
    sam2_controller = SAM2Controller(checkpoint_dir, model_cfg)
    
    with gr.Blocks(title="SAM2 Video Segmentation Tool") as interface:
        gr.Markdown("# SAM2 Video Segmentation Tool")
        
        # Status message - global status for all tabs
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
                        
                        # Directory input
                        dir_input = gr.Textbox(label="Image Directory Path")
                        dir_input.visible = False
                        browse_dir = gr.Button("Browse Directory")
                        browse_dir.visible = False
                        
                        # Output directory
                        output_dir = gr.Textbox(label="Output Directory (for extracted frames)")
                        browse_output = gr.Button("Select Output Folder")
                        
                        with gr.Row(visible=True) as fps_height_row:
                            fps = gr.Slider(1, 30, value=5, step=1, label="Frame Rate (FPS)")
                            height = gr.Slider(240, 2160, value=2160, step=60, label="Output Height")
                            
                        # Process button
                        process_input = gr.Button("Process Input", variant="primary")
                    
                    # Preview of the first frame
                    with gr.Column():
                        preview_img = gr.Image(label="Preview")
            
            # Step 2: Segment & Track
            with gr.Tab("2. Segment & Track"):
                gr.Markdown("### Select key points for tracking.")
                
                # Simple layout with initialize button first
                with gr.Row():
                    init_sam_btn = gr.Button("Initialize SAM", variant="primary")
                
                # We'll use this container to show/hide the initialization overlay
                with gr.Column(visible=True) as overlay_container:
                    # Overlay message when SAM is not initialized
                    sam_not_initialized_msg = gr.HTML(
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
                with gr.Column(visible=False) as segmentation_container:
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
                                pos_btn = gr.Button("Positive Point")
                                neg_btn = gr.Button("Negative Point")
                                
                            # Controls
                            with gr.Row():
                                clear_btn = gr.Button("Clear Points")
                                new_mask_btn = gr.Button("New Mask")
                                
                            gr.Markdown("### Tracking")
                            track_btn = gr.Button("Run Tracking", variant="primary")
                        
                        # Middle Column: Image for point selection
                        with gr.Column(scale=2):
                            point_image = gr.Image(label="Select Points")
                        
                        # Right Column: Results preview
                        with gr.Column(scale=2):
                            result_video = gr.Video(label="Tracking Preview")
            
            # Step 3: Export Results (new tab)
            with gr.Tab("3. Export Results"):
                gr.Markdown("### Export segmentation masks")
                
                with gr.Row():
                    with gr.Column():
                        # Export location
                        gr.Markdown("#### Select output directory:")
                        export_dir = gr.Textbox(label="Export Directory")
                        browse_export = gr.Button("Browse Export Location")
                        
                        # Export options
                        gr.Markdown("#### Export Options:")
                        with gr.Row():
                            with gr.Column():
                                export_color = gr.Checkbox(label="Color Masks", value=True)
                                export_index = gr.Checkbox(label="Index Masks", value=True)
                                export_raw = gr.Checkbox(label="Raw Numpy Data", value=False)
                            with gr.Column():
                                export_no_bg = gr.Checkbox(label="No Background", value=False)
                                export_as_video = gr.Checkbox(label="Export as Video", value=False)
                        
                        # Submit button
                        export_btn = gr.Button("Export Masks", variant="primary")
                        
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
                        export_preview = gr.Image(label="Mask Preview")
                        export_result = gr.Textbox(label="Export Status")
        
        # Event handlers
        # Step 1: Input Processing
        def update_browse_visibility(input_choice):
            if input_choice == "Video File":
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
            else:
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        
        input_type.change(
            update_browse_visibility,
            inputs=[input_type],
            outputs=[video_input, dir_input, browse_dir, output_dir, browse_output, fps_height_row]
        )
        
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
                    
                result = sam2_controller.extract_frames(video_path, out_dir, fps_val, height_val)
                num_frames, _ = sam2_controller.set_img_dir(out_dir)
                
                # Update frame slider max value
                slider_update = gr.Slider(maximum=num_frames-1 if num_frames > 0 else 0)
                
                # Get preview of first frame
                if num_frames > 0:
                    preview, _ = sam2_controller.set_input_image(0)
                    return result, preview, slider_update, out_dir
                else:
                    return result, None, slider_update, out_dir
                
            elif input_choice == "Image Directory" and dir_path:
                # Set image directory
                num_frames, result = sam2_controller.set_img_dir(dir_path)
                
                # Update frame slider max value
                slider_update = gr.Slider(maximum=num_frames-1 if num_frames > 0 else 0)
                
                # Get preview of first frame
                if num_frames > 0:
                    preview, _ = sam2_controller.set_input_image(0)
                    return result, preview, slider_update, dir_path
                else:
                    return result, None, slider_update, dir_path
            else:
                return "Please select a video file or image directory", None, gr.Slider(), ""
        
        process_input.click(
            process_input_fn,
            inputs=[input_type, video_input, dir_input, output_dir, fps, height],
            outputs=[status_msg, preview_img, frame_slider, output_dir]
        )
        
        # Step 2: Point Selection
        def update_frame(idx):
            image, msg = sam2_controller.set_input_image(int(idx))
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
            # Get the current maximum value directly from the gui
            max_idx = len(sam2_controller.img_paths) - 1 if sam2_controller.img_paths else 0
            return min(int(idx) + 1, max_idx)
        
        next_btn.click(
            go_to_next_frame,
            inputs=[frame_slider],
            outputs=[frame_slider]
        )
        
        # Initialize SAM and update the interface state
        def initialize_sam_and_update_ui(frame_idx):
            image, message = sam2_controller.initialize_sam(frame_idx)
            success = "initialized" in message.lower() and image is not None
            
            if success:
                # Show the segmentation container and hide the overlay
                segmentation_container.visible = True
                overlay_container.visible = False
                return (
                    image,  # point_image
                    message,  # status_msg
                    gr.update(visible=False),  # overlay_container
                    gr.update(visible=True)  # segmentation_container
                )
            else:
                # Keep overlay visible, segmentation hidden
                segmentation_container.visible = False
                overlay_container.visible = True
                return (
                    None,  # point_image
                    message or "SAM initialization failed. Please try again.",  # status_msg
                    gr.update(visible=True),  # overlay_container
                    gr.update(visible=False)  # segmentation_container
                )
        
        init_sam_btn.click(
            initialize_sam_and_update_ui,
            inputs=[frame_slider],
            outputs=[
                point_image, 
                status_msg, 
                overlay_container, 
                segmentation_container
            ]
        )
        
        # Check if SAM is initialized before handling clicks
        def add_point_from_click(frame_idx, img, evt: gr.SelectData):
            # Only process clicks when segmentation container is visible
            if segmentation_container.visible:
                i = evt.index[1]  # y coordinate
                j = evt.index[0]  # x coordinate
                return sam2_controller.add_point(frame_idx, i, j)
            else:
                return img  # Just return the input image if not initialized
        
        point_image.select(
            add_point_from_click,
            inputs=[frame_slider, point_image],
            outputs=[point_image]
        )
        
        # For other functions, we can use a simpler approach that doesn't depend on state checking
        
        def check_sam_initialized():
            # Simple function to check if SAM is ready (we use visibility as a proxy for initialization)
            return segmentation_container.visible
        
        # Track button handling
        def run_tracker():
            if not check_sam_initialized():
                return None, "Please initialize SAM first"
            return sam2_controller.run_tracker()
        
        track_btn.click(
            run_tracker,
            outputs=[result_video, status_msg]
        )
        
        # Button handlers that check visibility instead of state
        pos_btn.click(
            lambda: sam2_controller.set_positive() if check_sam_initialized() else "Please initialize SAM first",
            outputs=[status_msg]
        )
        
        neg_btn.click(
            lambda: sam2_controller.set_negative() if check_sam_initialized() else "Please initialize SAM first",
            outputs=[status_msg]
        )
        
        clear_btn.click(
            lambda: (sam2_controller.clear_points()[2], sam2_controller.set_input_image(sam2_controller.frame_index)[0]) 
                if check_sam_initialized() 
                else ("Please initialize SAM first", None),
            outputs=[status_msg, point_image]
        )
        
        new_mask_btn.click(
            lambda: sam2_controller.add_new_mask() if check_sam_initialized() else (None, "Please initialize SAM first"),
            outputs=[point_image, status_msg]
        )
        
        def safe_update_frame(idx):
            if not check_sam_initialized():
                return None, "Please initialize SAM first"
            return sam2_controller.set_input_image(int(idx))
        
        frame_slider.change(
            safe_update_frame,
            inputs=[frame_slider],
            outputs=[point_image, status_msg]
        )
        
        # Step 3: Export handling
        browse_export.click(
            browse_directory,
            outputs=[export_dir]
        )
        
        def show_mask_preview():
            """Show a preview of a mask if available"""
            if sam2_controller.color_masks_all and len(sam2_controller.color_masks_all) > 0:
                return sam2_controller.color_masks_all[0], "Mask preview loaded. Ready to export."
            return None, "No masks available. Run tracking first before exporting."
        
        # When switching to export tab, show preview if available
        tabs.change(
            show_mask_preview,
            inputs=None,
            outputs=[export_preview, status_msg]
        )
        
        # Collect export options and call save_masks
        def export_masks(export_dir, color_mask, index_mask, raw_mask, images_no_bg, as_video):
            export_options = {
                'color_mask': color_mask,
                'index_mask': index_mask,
                'raw_mask': raw_mask,
                'images_no_bg': images_no_bg,
                'as_video': as_video
            }
            return sam2_controller.save_masks(export_dir, export_options)
        
        export_btn.click(
            export_masks,
            inputs=[
                export_dir, 
                export_color, 
                export_index, 
                export_raw, 
                export_no_bg,
                export_as_video
            ],
            outputs=[export_result]
        )
        
    return interface
