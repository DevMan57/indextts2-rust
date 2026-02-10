import wave
import math
import struct

def generate_sine_wave(filename, duration=1.0, freq=440.0, sample_rate=16000):
    n_frames = int(duration * sample_rate)
    with wave.open(filename, 'w') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        
        data = []
        for i in range(n_frames):
            value = int(32767.0 * math.sin(2.0 * math.pi * freq * i / sample_rate))
            data.append(struct.pack('<h', value))
            
        w.writeframes(b''.join(data))
    print(f"Generated {filename}")

if __name__ == "__main__":
    generate_sine_wave("speaker.wav")
