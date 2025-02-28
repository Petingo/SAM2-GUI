import torch
from src.interface import create_interface

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
    interface = create_interface(args.checkpoint_dir, args.model_cfg)
    interface.launch(server_port=args.port, share=False)