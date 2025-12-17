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
