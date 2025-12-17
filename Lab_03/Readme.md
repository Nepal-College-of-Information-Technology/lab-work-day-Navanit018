# *Lab : 3*

# *MIDI Note Processing and Algorithmic Music Generation*


## *Objectives*

- Print MIDI channels, notes, and their corresponding frequencies.  
- Generate simple beat, harmony, and melody patterns using MIDI notes.  
- Play individual MIDI tracks (.mid files).  
- Convert MIDI tracks into standard audio (.wav) files using Python tools.

  

## *Background Theory*

### *1. MIDI Standard*

- *MIDI (Musical Instrument Digital Interface)* is a communication protocol used to send digital music performance data.  
- It stores musical instructions such as:
  - note_on  
  - note_off  
  - MIDI channel  
  - Velocity  
  - Note number (0–127)
- MIDI Note Numbers map to musical notes.  
  - Example: 60 = Middle C, 69 = A4 (440 Hz)
 
### *2. Music Theory (Basic)*

#### *Beat*
- The basic unit of time in a musical composition.

#### *Harmony*
- Two or more notes played together (chords).

#### *Melody*
- A sequential pattern of musical notes forming the main tune.

---



#### *Frequency Formula*

To convert a MIDI note number n to its frequency: f = 440 × 2^((n - 69) / 12)


---

### *3. Python Libraries Used*

#### *Mido*
- For reading, writing, and analyzing MIDI files.

#### *Pygame*
- For playing MIDI files and basic audio handling.

---

## *Procedure*

1. Import required libraries:

    python
    import mido
    import math
    import pygame
    

2. Load the MIDI file:

    python
    mid = mido.MidiFile("input.mid")
    

3. Iterate and print MIDI messages, channels, and note numbers:

    python
    for msg in mid:
        if msg.type == "note_on":
            print(msg.note, msg.channel)
    

4. Convert MIDI note numbers to frequency:

    python
    def midi_to_freq(n):
        return 440 * (2 ** ((n - 69) / 12))
    

5. Generate:
   - Beat pattern  
   - Harmony (simple triads)  
   - Melody sequence  

6. Save generated tracks as:

    - beat.mid
    - harmony.mid
    - melody.mid

7. Play the MIDI file:

    python
    pygame.mixer.music.load("melody.mid")
    pygame.mixer.music.play()
    

8. Convert MIDI to WAV (if synthesizer or pygame MIDI backend is available).

---

## *Output*

- Printed MIDI:
  - Channels  
  - Notes  
  - Frequencies  

- Generated MIDI tracks:
  - beat.mid
  - harmony.mid
  - melody.mid

- Played audio using Pygame.

- Converted .wav files from MIDI tracks (if tools installed).

---



