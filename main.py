#!/usr/bin/env python3
"""
Main entry point for Emotion Detector Pro App
"""

import sys
import os

# Add the utils directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from app import EmotionDetectorProApp
import tkinter as tk

def main():
    """Launch the application"""
    try:
        root = tk.Tk()
        app = EmotionDetectorProApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Failed to start application: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()