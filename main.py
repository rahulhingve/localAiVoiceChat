import json
import requests
from stt import OfflineSTT
from generate import KPipeline
import soundfile as sf
import numpy as np
import re
import time
import os

class AIVoiceChatBot:
    def __init__(self):
        self.stt = OfflineSTT()
        self.tts_pipeline = KPipeline(lang_code='a')
        self.ollama_url = "http://localhost:11434/api/generate"
        self.system_prompt = """You are a helpful AI assistant. Keep your responses short, clear, and friendly. 
        Limit responses to 1-3 sentences maximum. Avoid using emojis or special characters."""
        self.response_audio_file = 'current_response.wav'  # Fixed filename for response audio
    
    def remove_emojis(self, text):
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def get_llm_response(self, user_input):
        try:
            payload = {
                "model": "mistral",
                "prompt": f"{self.system_prompt}\nUser: {user_input}\nAssistant:",
                "stream": False
            }
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            response_text = response.json()['response']
            return self.remove_emojis(response_text.strip())                                                                                                                                                           
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Ollama: {e}")
            return "I apologize, but I'm having trouble connecting to my language model right now."
        except Exception as e:
            print(f"Unexpected error: {e}")
            return "I encountered an unexpected error. Please try again."

    def generate_and_play_audio(self, text):
        try:
            print("AI Response:", text)
            print("Generating audio response...")
            
            # Delete previous response file if it exists
            if os.path.exists(self.response_audio_file):
                os.remove(self.response_audio_file)
            
            generator = self.tts_pipeline(text, voice='af_heart')
            combined_audio = []
            for _, _, audio in generator:
                combined_audio.append(audio)
            
            final_audio = np.concatenate(combined_audio)
            sf.write(self.response_audio_file, final_audio, 24000)
            
            # Play audio
            with sf.SoundFile(self.response_audio_file) as f:
                import sounddevice as sd
                sd.play(f.read(), f.samplerate)
                sd.wait()
            
        except Exception as e:
            print(f"Error in audio generation/playback: {e}")

    def __del__(self):
        """Cleanup on object destruction"""
        try:
            if os.path.exists(self.response_audio_file):
                os.remove(self.response_audio_file)
        except:
            pass

    def run(self):
        print("Starting AI Voice Chatbot...")
        print("Press and hold SPACE to speak, release to process, or ESC to quit.")
        
        def process_stt_result(text):
            if not text or text.isspace():
                return
            print("\nYou said:", text)
            
            # Get LLM response
            ai_response = self.get_llm_response(text)
            
            # Generate and play audio response
            self.generate_and_play_audio(ai_response)
            
            print("\nPress and hold SPACE to speak, release to process, or ESC to quit.")

        # Modify OfflineSTT to accept callback
        original_transcribe = self.stt.transcribe_audio
        def transcribe_with_callback(audio_path):
        
            text = original_transcribe(audio_path)
            process_stt_result(text)
            return text
        
        self.stt.transcribe_audio = transcribe_with_callback
        
        # Start the recording loop
        self.stt.record_audio()

if __name__ == "__main__":
    bot = AIVoiceChatBot()
    bot.run()