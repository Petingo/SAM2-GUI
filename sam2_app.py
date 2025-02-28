from src.sam2_interface import SAM2Interface

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--share", type=bool, default=False)
    parser.add_argument("--checkpoint_dir", type=str, default="../sam2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--model_cfg", type=str, default="../sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
    args = parser.parse_args()
    
    # Create and launch the interface
    interface = SAM2Interface(args.checkpoint_dir, args.model_cfg, args.port)
    interface.launch(share=args.share)