# localAiVoiceChat

## Demo on YouTube
[![Watch the video](https://img.youtube.com/vi/RGdeECvPW1k/0.jpg)](https://www.youtube.com/watch?v=RGdeECvPW1k)

A Python-based application for voice conversations with an AI assistant, running entirely on your local machine.

---

## How to Use This Repository

Follow these steps to set up and use **localAiVoiceChat** on your system.

---

### 1. Prerequisites

- **Python 3.8 or newer** must be installed.
- A **microphone** and **speakers/headphones** connected to your computer.
- At least **16GB RAM** is recommended.
- (Optional but recommended) **Nvidia GPU** with CUDA drivers for faster AI processing.
- **Git** must be installed.

---

### 2. Clone the Repository

Open your terminal or command prompt and run:
git clone https://github.com/rahulhingve/localAiVoiceChat.git
cd localAiVoiceChat


---

### 3. Install Required Applications and Models

#### a. **Install ESPnet**

- Go to the ESPnet [official page](https://espnet.github.io/espnet/), download and install the application for your OS.

#### b. **Install Ollama**

- Download and install [Ollama](https://ollama.com/) for your operating system.

#### c. **Download the Mistral Model**

After installing Ollama, open your command prompt and run:

---

### 4. Create and Activate a Virtual Environment

It’s best to use a virtual environment to keep dependencies isolated.

#### On Windows:
python -m venv venv
venv\Scripts\activate

#### On Mac/Linux:
python3 -m venv venv
source venv/bin/activate


---

### 5. Install Python Dependencies


---

### 6. Install PyTorch

Go to [PyTorch official site](https://pytorch.org/get-started/locally/) and select your OS, package manager, Python version, and CUDA version (if you have a GPU). Copy the generated `pip` command and run it in your terminal.

Example for Windows with CUDA 12.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

If you don't have a GPU, select CPU-only and use the corresponding command.

---

### 7. Authenticate with Hugging Face

The script downloads models from Hugging Face. You need to log in:

pip install huggingface_hub
huggingface-cli login
- Go to [Hugging Face](https://huggingface.co/), create an account if needed.
- Generate an access token in your profile and use it to log in.

---

### 8. Run the Application

python main.py


- On first run, required models will be downloaded automatically.
- When prompted, **press and hold the space bar to speak**.
- Release the space bar to let the AI process and respond.

---

### 9. Custom Commands (Optional)

- You can view or edit supported commands in the `commands.txt` file.
- To add new commands, open `commands.txt` in a text editor and follow the existing format.

---

## Project Structure

| File/Folder        | Description                                    |
|--------------------|------------------------------------------------|
| `main.py`          | Main script to run the application             |
| `stt.py`           | Handles speech-to-text conversion              |
| `generate.py`      | Generates AI responses                         |
| `commands.txt`     | List of available voice commands               |
| `requirements.txt` | Python dependencies                            |

---

## Troubleshooting

- Ensure your microphone is working and all dependencies are installed.
- If you have issues with PyTorch, check your CUDA or CPU installation.
- For Hugging Face authentication issues, ensure your token is valid and has the right scopes.
- If models fail to download, check your internet connection and Hugging Face login.

---

## Contributing

Pull requests and issues are welcome!

---


