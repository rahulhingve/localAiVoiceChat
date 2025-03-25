import torch
from kokoro import KPipeline
import soundfile as sf
from playsound import playsound
import numpy as np

def main():
    # Initialize device and print info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    try:
        # Initialize pipeline
        pipeline = KPipeline(lang_code='a')
        
        # Text to generate
        text = "A paragraph is a distinct section of writing, typically consisting of several sentences, that focuses on a single idea or topic, usually indicated by an indent at the beginning of the first sentence. Heres a more detailed breakdown: Definition: A paragraph is a subdivision of a written composition that consists of one or more sentences, deals with one point or gives the words of one speaker, and begins on a new usually indented line. Purpose: Paragraphs help to organize and structure written work, making it easier for readers to follow the authors thoughts. Structure: A well-structured paragraph typically includes a topic sentence (introducing the main idea), supporting sentences (providing details and evidence), and a concluding sentence (summarizing or transitioning to the next paragraph). Length:While theres no strict rule, paragraphs are often around 100-200 words, with a maximum of five sentences. Coherence:A paragraph should have a clear and consistent flow, with all sentences related to the main topic.Indentation:Paragraphs are usually distinguished by an indentation at the beginning of the first line, though this can vary depending on the writing style or context. "
        
        # Generate audio using pipeline
        print("Generating audio...")
        generator = pipeline(text, voice='af_heart')
        
        # Combine all audio segments
        combined_audio = []
        for _, _, audio in generator:
            combined_audio.append(audio)
        
        # Concatenate all audio segments
        final_audio = np.concatenate(combined_audio)
        
        # Save as single file
        output_file = 'output.wav'
        sf.write(output_file, final_audio, 24000)
        print(f"Audio saved to {output_file}")
        
        # Auto-play the complete audio
        print("Playing audio...")
        playsound(output_file)
        
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()