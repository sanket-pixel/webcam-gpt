import base64
import io
import torch
from flask import Flask, render_template
from flask_socketio import SocketIO
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
# Use a specific revision for reproducibility
MODEL_ID = "vikhyatk/moondream2"
MODEL_REVISION = "2025-06-21"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Model Loading ---
# Load the model and tokenizer once when the server starts.
# This is memory-intensive and should only happen once.
print(f"Loading model '{MODEL_ID}'...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    trust_remote_code=True,
    torch_dtype=torch.float16, # Use float16 for better performance
    device_map={"": DEVICE}
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
print("Model loaded successfully.")

# --- Flask App Initialization ---
app = Flask(__name__)
# A secret key is needed for session management
app.config['SECRET_KEY'] = 'your-very-secret-key'
# Initialize SocketIO for real-time communication
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Web Routes ---
@app.route('/')
def index():
    """
    Serves the main HTML page that contains the webcam feed and chat interface.
    """
    return render_template('index.html')

# --- WebSocket Event Handlers ---
@socketio.on('query')
def handle_query(data):
    """
    Handles an incoming query from the client.
    The data dictionary is expected to have 'question' and 'image' keys.
    """
    question = data['question']
    # The image comes in as a Base64 encoded string.
    # We need to remove the header part "data:image/jpeg;base64,"
    image_b64 = data['image'].split(',')[1]

    print(f"Received query: '{question}'")

    try:
        # Decode the Base64 string into bytes, then open it as a PIL Image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes))

        # --- THIS IS THE FIX ---
        # Convert the image to RGB format to remove any alpha channel
        image = image.convert('RGB')
        # ---------------------

        print(f"Image processed successfully, size: {image.size}, mode: {image.mode}")

        # Use the model's high-level .query() method. This is the "magic" part.
        result = model.query(image, question)
        answer = result['answer']

        print(f"Generated answer: '{answer}'")

        # Send the answer back to the client
        socketio.emit('response', {'answer': answer})

    except Exception as e:
        print(f"An error occurred: {e}")
        socketio.emit('response', {'answer': f"Sorry, an error occurred: {e}"})

# --- Main Entry Point ---
if __name__ == '__main__':
    print("Starting Flask + SocketIO server...")
    # Run the app on all available network interfaces, useful for testing
    socketio.run(app, host='0.0.0.0', port=5000)

