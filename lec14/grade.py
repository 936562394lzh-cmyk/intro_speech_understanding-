import os
import tempfile
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment
import librosa
import IPython

def synthesize(text, lang, filename):
  
    tts = gTTS(text=text, lang=lang, slow=False)
    
    tts.save(filename)
    
    print(f"Speech synthesized and saved to {filename}")

def make_a_corpus(texts, languages, filenames):
   
    recognized_texts = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, (text, lang, filename) in enumerate(zip(texts, languages, filenames)):

            mp3_filename = f"{filename}.mp3"
            synthesize(text, lang, mp3_filename)
            
            wav_filename = os.path.join(temp_dir, f"temp_{i}.wav")
            try:

                audio = AudioSegment.from_mp3(mp3_filename)
                audio.export(wav_filename, format="wav")
            except Exception as e:
                print(f"Error converting MP3 to WAV: {e}")

                try:
                    y, sr = librosa.load(mp3_filename)
                    librosa.output.write_wav(wav_filename, y, sr)
                except Exception as e2:
                    print(f"Fallback conversion also failed: {e2}")
                    recognized_texts.append(f"ERROR: Could not process {filename}")
                    continue
            
            try:

                recognizer = sr.Recognizer()
                
                with sr.AudioFile(wav_filename) as source:

                    recognizer.adjust_for_ambient_noise(source)

                    audio_data = recognizer.record(source)
                
                recognized_text = recognizer.recognize_google(audio_data)
                recognized_texts.append(recognized_text)
                
                print(f"Text '{text}' was recognized as '{recognized_text}'")
                
            except sr.UnknownValueError:
                print(f"Google Speech Recognition could not understand audio from {filename}")
                recognized_texts.append("ERROR: Could not recognize speech")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")

                try:
                    recognized_text = recognizer.recognize_sphinx(audio_data)
                    recognized_texts.append(recognized_text)
                except:
                    recognized_texts.append(f"ERROR: Recognition failed for {filename}")
            except Exception as e:
                print(f"Unexpected error in speech recognition: {e}")
                recognized_texts.append(f"ERROR: {e}")
            

    return recognized_texts


if __name__ == "__main__":

    print("Testing synthesize function...")
    synthesize("This is speech synthesis!", "en", "english.mp3")
    
    if os.path.isfile("english.mp3"):
        print("✓ english.mp3 created successfully")

        try:
            y, sr = librosa.load("english.mp3")
            IPython.display.display(IPython.display.Audio(data=y, rate=sr))
        except:
            pass
    else:
        print("✗ Failed to create english.mp3")
    
    print("\nTesting make_a_corpus function...")
    texts = ['hello', 'my name is', 'my name is', 'will the real shady please stand up']
    languages = ['en', 'en', 'en', 'en']
    filenames = ['file1', 'file2', 'file3', 'file4']
    
    recognized_texts = make_a_corpus(texts, languages, filenames)
    
    print("\nResults:")
    for original, recognized in zip(texts, recognized_texts):
        print(f"'{original}' was recognized as '{recognized}'")
