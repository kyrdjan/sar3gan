"""
Convert old checkpoint format to new format

might be needed for compatibility with new model architecture.

sample usage:python convert_checkpoint.py network-snapshot-000000000.pkl network-snapshot-000000000-converted.pkl 

"""
import pickle
import torch

def convert_old_to_new_checkpoint(old_pkl_path, new_pkl_path):
    """
    Convert a checkpoint saved with old GeneratorStage (self.Layers)
    to new GeneratorStage format (self.Transition, self.Blocks, self.Attention)
    """
    
    print(f"Loading old checkpoint from: {old_pkl_path}")
    with open(old_pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # Get the generator
    G = data['G_ema']
    
    # Iterate through all GeneratorStage modules
    for name, module in G.named_modules():
        if type(module).__name__ == 'GeneratorStage':
            if hasattr(module, 'Layers'):
                print(f"Converting GeneratorStage: {name}")
                
                # Extract components from old Layers structure
                layers_list = list(module.Layers)
                
                # First layer is Transition (GenerativeBasis or UpsampleLayer)
                module.Transition = layers_list[0]
                
                # Middle layers are ResidualBlocks
                blocks = []
                attention = None
                for layer in layers_list[1:]:
                    if type(layer).__name__ == 'ResidualBlock':
                        blocks.append(layer)
                    elif type(layer).__name__ == 'SelfAttention':
                        attention = layer
                
                module.Blocks = torch.nn.ModuleList(blocks)
                
                # Attention layer (if exists, otherwise create new one)
                if attention is not None:
                    module.Attention = attention
                else:
                    # Create a new attention layer if it didn't exist
                    from R3GAN.Networks import SelfAttention
                    output_channels = module.Transition.LinearLayer.out_features if hasattr(module.Transition, 'LinearLayer') else blocks[0].LinearLayer1.Layer.out_channels
                    module.Attention = SelfAttention(output_channels)
                
                # Remove old Layers attribute
                del module.Layers
                
                print(f"  Converted: Transition + {len(blocks)} Blocks + Attention")
    
    # Save the converted checkpoint
    print(f"\nSaving converted checkpoint to: {new_pkl_path}")
    with open(new_pkl_path, 'wb') as f:
        pickle.dump(data, f)
    
    print("✓ Conversion complete!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python convert_checkpoint.py <old_checkpoint.pkl> <new_checkpoint.pkl>")
        sys.exit(1)
    
    old_pkl = sys.argv[1]
    new_pkl = sys.argv[2]
    
    convert_old_to_new_checkpoint(old_pkl, new_pkl)