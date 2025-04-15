import socket
import torch
import json
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import threading
import time
from model import Autoencoder  # Import the model

# Configuration
HOST = '127.0.0.1'  # Localhost for testing
PORT = 12345
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = Autoencoder().to(device)
model.load_state_dict(torch.load("autoencoder_flickr8k.pth", map_location=device))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Receiver function
def receiver():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        print("Receiver is waiting for a connection...")
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            data = conn.recv(65536).decode('utf-8')  # Receive latent vector
            latent_array = json.loads(data)  # Deserialize latent vector
            latent_tensor = torch.tensor(latent_array, device=device, dtype=torch.float32)

            # Decode latent vector into an image
            with torch.no_grad():
                reconstructed_image = model.decoder(latent_tensor)

            # Post-process the image for saving
            reconstructed_image = reconstructed_image.squeeze(0).cpu()  # Remove batch dimension
            reconstructed_image = (reconstructed_image * 0.5 + 0.5).clamp(0, 1)  # De-normalize

            # Convert to PIL image and save
            reconstructed_pil = to_pil_image(reconstructed_image)
            reconstructed_pil.save("reconstructed_image.jpg")
            print("Reconstructed image saved as reconstructed_image.jpg.")

# Sender function
def sender():
    time.sleep(1)  # Ensure the receiver is ready
    image_path = "C:/Users/priya/Desktop/Projects/Autoencoder/example.jpg"  
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Encode the image
    with torch.no_grad():
        latent_vector = model.encoder(image_tensor)

    # Convert latent vector to JSON
    latent_numpy = latent_vector.cpu().numpy()
    latent_json = json.dumps(latent_numpy.tolist())

    # Send the latent vector over a socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(latent_json.encode('utf-8'))
        print("Latent vector sent successfully.")

# Run receiver and sender in separate threads
receiver_thread = threading.Thread(target=receiver)
sender_thread = threading.Thread(target=sender)

receiver_thread.start()
sender_thread.start()

receiver_thread.join()
sender_thread.join()
