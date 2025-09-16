# Depth-Controlled Flappy Bird

A hands-free Flappy Bird game controlled by hand depth using Intel RealSense camera and MediaPipe hand detection.

## Overview

This project implements two versions of a depth-controlled Flappy Bird game:
- **Single Window Version** (`depth_game.py`) - Camera feed and game combined side-by-side
- **Multi-Window Version** (`depth_jetmap_game.py`) - Separate game window and camera feedback window

Control the bird's height by moving your hand closer to or further from the camera - no clicking required!

## Hardware Requirements

- **Intel RealSense Depth Camera** (D435, D455, or compatible model)
- USB 3.0 port for camera connection
- Computer with decent processing power for real-time depth processing

## Software Requirements

### Dependencies
```bash
pip install opencv-python pyrealsense2 mediapipe numpy
```

### Detailed Requirements
- **Python 3.7+**
- **OpenCV** - Computer vision operations and display
- **PyRealSense2** - Intel RealSense SDK Python wrapper
- **MediaPipe** - Google's hand detection and tracking
- **NumPy** - Numerical operations and array handling

## Installation

1. **Install Intel RealSense SDK**
   - Download from [Intel RealSense website](https://www.intelrealsense.com/sdk-2/)
   - Follow platform-specific installation instructions

2. **Clone/Download the code**
   ```bash
   git clone <your-repo-url>
   cd depth-flappy-bird
   ```

3. **Install Python dependencies**
   ```bash
   pip install opencv-python pyrealsense2 mediapipe numpy
   ```

4. **Connect your RealSense camera** via USB 3.0

## Usage

### Single Window Version
```bash
python depth_game.py
```
- Shows camera feed (left) and game (right) in one window
- Immediate gameplay - no start screen

### Multi-Window Version  
```bash
python depth_jetmap_game.py
```
- **Game Window**: Clean Flappy Bird interface
- **Feedback Window**: RGB camera feed (left) + colorized depth map (right)
- **Start Screen**: Press 'S' to begin

## Controls

### Game Controls
- **'S'** - Start game (multi-window version)
- **'R'** - Restart/Reset game
- **'Q'** - Quit application

### Hand Controls
- **Move hand closer** to camera → Bird flies **UP**
- **Move hand further** from camera → Bird flies **DOWN**
- **No hand detected** → Bird maintains current position

## How It Works

### 1. Hand Detection
- MediaPipe detects hand landmarks in RGB camera feed
- Calculates bounding box around detected hand
- Draws landmarks and bounding box for visual feedback

### 2. Depth Measurement
- Samples depth values in region around hand centroid
- Uses median depth for stability (reduces noise)
- Applies smoothing filter for responsive but stable control

### 3. Game Control
- Maps hand depth (300-800mm range) to bird Y-position
- Closer hand = higher bird position
- Further hand = lower bird position
- Smooth movement interpolation prevents jerky motion

### 4. Visual Feedback
- **RGB Feed**: Shows hand landmarks, bounding box, and depth readings
- **Depth Map**: Colorized heat map (red=close, blue=far) with hand position
- **Game Display**: Traditional Flappy Bird with score, level, and status

## Game Features

- **Progressive Difficulty**: Speed increases with score
- **Level System**: Every 5 points = new level
- **Real-time Status**: Hand detection and depth readings displayed
- **Collision Detection**: Standard Flappy Bird physics
- **Score Tracking**: Points for each pipe cleared

## Troubleshooting

### Camera Issues
- **"RealSense camera not detected"**
  - Check USB 3.0 connection
  - Verify camera drivers installed
  - Try different USB port

### Hand Detection Issues
- **"No Hand Detected"**
  - Ensure adequate lighting
  - Position hand clearly in camera view
  - Avoid cluttered background
  - Check if hand is within 300-800mm range

### Performance Issues
- **Lag or low FPS**
  - Close other applications using camera
  - Reduce MediaPipe detection confidence
  - Check USB bandwidth (use dedicated USB controller)

### Game Control Issues
- **Bird not responding to hand movement**
  - Verify depth readings are changing (check feedback window)
  - Ensure hand is within 300-800mm range
  - Try recalibrating by moving hand through full range

## Technical Details

### Depth Processing
- **Sampling**: 30x30 pixel region around hand center
- **Filtering**: Median filter for noise reduction  
- **Smoothing**: Exponential moving average (0.6 current + 0.4 previous)
- **Range**: 300-800mm optimal detection range

### Hand Detection Settings
- **Detection Confidence**: 0.7 minimum
- **Tracking Confidence**: 0.5 minimum
- **Max Hands**: 1 (single hand control)

### Game Parameters
- **Bird Speed**: Varies by level (3.0 - 8.0 pixels/frame)
- **Pipe Gap**: 150 pixels
- **Pipe Width**: 60 pixels
- **Bird Radius**: 15 pixels

## File Structure

```
├── depth_game.py              # Single window version
├── depth_jetmap_game.py       # Multi-window version  
└── README.md                  # This file
```

## Version Differences

| Feature | Single Window | Multi-Window |
|---------|---------------|--------------|
| Layout | Side-by-side | Separate windows |
| Start Screen | No | Yes |
| Depth Visualization | Basic | Colorized heat map |
| Window Management | Simple | More complex |
| Best For | Quick testing | Development/analysis |

## Development Notes

### Code Architecture
- **Class-based design** with clear separation of concerns
- **Error handling** for camera disconnection and detection failures  
- **Real-time processing** optimized for 30 FPS
- **Modular functions** for easy modification and testing

### Customization Options
- Adjust depth range (`min_depth`, `max_depth`)
- Modify smoothing factor (`depth_smoothing`)
- Change game difficulty progression
- Customize visual feedback and colors

## License

Open source - feel free to modify and distribute.

## Contributing

Contributions welcome! Areas for improvement:
- Additional gesture controls
- Multiple hand support
- Calibration interface  
- Sound effects
- High score persistence
