import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os
from dotenv import load_dotenv
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import keyboard
import threading
import tempfile
import time
import shutil

load_dotenv()

class OfflineSTT:
    def __init__(self):
        self.model_id = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_id, local_files_only=False)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id, local_files_only=False)
        # Move model to GPU if available
        self.model = self.model.to(self.device)
        
        # Save model locally after first load
        self.processor.save_pretrained("./local_model")
        self.model.save_pretrained("./local_model")

        # Create temp directory for audio files
        self.temp_dir = os.path.join(os.path.dirname(__file__), 'tmp_stt_audio')
        os.makedirs(self.temp_dir, exist_ok=True)

    def cleanup_temp_files(self):
        """Clean up temporary audio files"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            os.makedirs(self.temp_dir)

    def transcribe_audio(self, audio_path):
        try:
            # Load audio file
            speech_array, sampling_rate = librosa.load(audio_path, sr=16_000)
            
            # Process audio
            inputs = self.processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Perform inference
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            # Decode the output
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            return transcription
        except Exception as e:
            print(f"Error in transcription: {e}")
            return ""

    def record_audio(self):
        # Audio parameters
        sample_rate = 16000
        recording = []
        is_recording = False
        
        # Clean up any old temp files
        self.cleanup_temp_files()
        
        def audio_callback(indata, frames, time, status):
            nonlocal recording
            if status:
                print(status)
            if is_recording:
                recording.append(indata.copy())
        
        # Create an input stream
        with sd.InputStream(callback=audio_callback,
                          channels=1,
                          samplerate=sample_rate,
                          blocksize=8000) as stream:
            print("Press and hold SPACE to record, ESC to quit...")
            while True:
                try:
                    if keyboard.is_pressed('space'):
                        if not is_recording:
                            recording = []  # Clear recording buffer
                            is_recording = True
                            print("\nRecording...", flush=True)
                    else:
                        if is_recording:
                            is_recording = False
                            if len(recording) > 0:
                                print("\nProcessing...", flush=True)
                                # Convert recording to numpy array
                                audio_data = np.concatenate(recording, axis=0)
                                
                                try:
                                    # Use custom temp directory
                                    temp_path = os.path.join(self.temp_dir, f'audio_{int(time.time())}.wav')
                                    
                                    # Write WAV file
                                    wav.write(temp_path, sample_rate, audio_data)
                                    
                                    # Process the audio
                                    text = self.transcribe_audio(temp_path)
                                    print("\nTranscription:", text, flush=True)
                                    
                                except Exception as e:
                                    print("\nError during transcription:", str(e), flush=True)
                                finally:
                                    # Clean up temp file
                                    try:
                                        if os.path.exists(temp_path):
                                            os.remove(temp_path)
                                    except Exception as e:
                                        print(f"\nWarning: Could not delete temp file: {str(e)}", flush=True)
                                
                                print("\nPress and hold SPACE to record, ESC to quit...", flush=True)
                    
                    if keyboard.is_pressed('esc'):
                        print("\nStopping...", flush=True)
                        break
                        
                    sd.sleep(10)  # Small sleep to prevent CPU overload
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"\nError: {str(e)}", flush=True)
                    break

    def __del__(self):
        """Cleanup on object destruction"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except:
            pass

if __name__ == "__main__":
    stt = OfflineSTT()
    stt.record_audio()
