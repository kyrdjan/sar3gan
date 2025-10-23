import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# ====== Import your Generator definition ======
from cheapsar3gan import Generator  # replace with your generator file (no .py)

# --- Safe checkpoint loading ---
def load_weights_safely(checkpoint_path, device):
    try:
        # New PyTorch (>=2.5) safer way
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        # Fallback for older PyTorch
        state_dict = torch.load(checkpoint_path, map_location=device)
    return state_dict

def load_generator(checkpoint_path, device="cuda"):
    model = Generator().to(device)
    state_dict = load_weights_safely(checkpoint_path, device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def preprocess_image(img_path):
    """Load LR image and preprocess to tensor."""
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)  # add batch dim

def postprocess_and_save(tensor, save_path):
    """Convert model output tensor back to image and save."""
    save_image(tensor, save_path, normalize=True, value_range=(-1, 1))

def run_inference(model, input_folder, output_folder, device="cuda"):
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    print(f"Found {len(files)} input images in {input_folder}")

    for f in files:
        lr_tensor = preprocess_image(os.path.join(input_folder, f)).to(device)
        with torch.no_grad():
            sr_tensor = model(lr_tensor)  # generator outputs 256x256
        postprocess_and_save(sr_tensor, os.path.join(output_folder, f))
        print(f"[✓] Saved SR image: {f}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = "training_snapshots/best_generator.pth"
    input_folder = "training_snapshots/input/00000"       # put your 64x64 images here
    output_folder = "training_snapshots/output"     # fixed typo "ouput"

    G = load_generator(checkpoint, device)
    run_inference(G, input_folder, output_folder, device)
