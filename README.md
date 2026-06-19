# Anime Power Simulator 🚀

Anime Power Simulator is a real-time computer vision application that captures your webcam stream, tracks facial landmarks and hand gestures using MediaPipe, and overlays dynamic anime-inspired visual effects. The app features procedural audio synthesis and live MP4 recording.

---

## Features & Supported Characters

- **Luffy (Gum Gum Stretch)**: Stretch your mouth vertically. The mouth area on screen warps and stretches elastically based on a spring-mass physics model.
- **Naruto (Rasengan)**: Bring both hands together in front of you. A swirling blue Rasengan orb forms with rotating energy lines and swirling particles.
- **Goku (Kamehameha)**: Hold both hands close together below your face level to charge up. Push them forward/apart to unleash a massive energy blast across the screen.
- **Iron Man (Repulsor Blast)**: Show an open palm facing the camera. A red/gold lock-on targeting reticle centers on your hand, firing a laser beam downwards.
- **Doctor Strange (Eldritch Portal)**: Sweep your hand in a circular path. Renders a rotating orange magical rune ring surrounded by sparks.

---

## Prerequisites

- **Python 3.11+**
- A working webcam connected to your computer.

---

## Installation & Setup

1. **Open a terminal** in the project directory.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Simulator

Launch the main script using python:
```bash
python src/main.py
```

---

## Key Bindings & Controls

| Key | Action |
| --- | --- |
| `1` | Select Luffy Mode |
| `2` | Select Naruto Mode |
| `3` | Select Goku Mode |
| `4` | Select Iron Man Mode |
| `5` | Select Doctor Strange Mode |
| `SPACE` | Cycle to the next character |
| `D` | Toggle Telemetry Debug Overlays (draws tracking wireframes) |
| `R` | Start / Stop MP4 video recording (saved under `assets/recordings/`) |
| `ESC` | Exit the simulator cleanly |

---

## File Structure

```text
anime-power-simulator/
├── assets/
│   └── recordings/              # Output recordings (.mp4 files)
├── src/
│   ├── audio/
│   │   ├── __init__.py
│   │   └── audio_engine.py      # Real-time procedural audio synthesis
│   ├── gestures/
│   │   ├── __init__.py
│   │   └── gesture_detector.py  # Rotation-invariant gesture checks
│   ├── trackers/
│   │   ├── __init__.py
│   │   ├── camera.py            # OpenCV live webcam frame acquisition
│   │   ├── face_tracker.py      # MediaPipe Face Mesh wrapper
│   │   ├── hand_tracker.py      # MediaPipe Hand Tracking wrapper
│   │   ├── mouth_tracker.py     # Normalized mouth dimensions
│   │   └── pose_tracker.py      # MediaPipe Pose tracking
│   ├── physics/
│   │   ├── __init__.py
│   │   ├── physics_engine.py    # Spring-mass math simulation
│   │   └── smoothing.py         # Exponential Moving Average jitter reduction
│   ├── rendering/
│   │   ├── __init__.py
│   │   ├── glow_engine.py       # Layered transparency concentric glows
│   │   ├── particle_engine.py   # Spawning, gravity, and decay physics
│   │   ├── trail_engine.py      # Vector trailing motion
│   │   └── warp_engine.py       # Elastic mouth ROI cropping and stretching
│   ├── effects/
│   │   ├── __init__.py
│   │   ├── luffy_effect.py
│   │   ├── naruto_effect.py
│   │   ├── goku_effect.py
│   │   ├── ironman_effect.py
│   │   └── strange_effect.py
│   ├── config.py                # Global configurations & color templates
│   └── main.py                  # Pygame execution loop & telemetry UI
├── README.md
└── requirements.txt
```
