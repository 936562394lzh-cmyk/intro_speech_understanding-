import speech_recognition as sr

def transcribe_wavefile(filename, language):
    """
    Use sr.Recognizer.AudioFile(filename) as the source,
    recognize from that source,
    and return the recognized text.

    @params:
        filename (str) - the filename from which to read the audio
        language (str) - the language of the audio

    @returns:
        text (str) - the recognized speech
    """
   
    r = sr.Recognizer()
    
    with sr.AudioFile(filename) as source:
        
        r.adjust_for_ambient_noise(source)
        
        audio_data = r.record(source)
    
    try:
       
        text = r.recognize_google(audio_data, language=language)
        return text
    except sr.UnknownValueError:
        
        return ""
    except sr.RequestError as e:
       
        raise RuntimeError(f"Could not request results from Google Speech Recognition service; {e}")
