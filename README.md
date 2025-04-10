# Flappy Bird: Q-Learning

A reimagined Flappy Bird with a vibrant 90s arcade aesthetic, powered by Q-learning reinforcement learning. This project blends retro-style graphics with an AI agent that learns to navigate pipes, offering both training and human-playable modes.

<div align="center">
  <img src="gameplay.gif" alt="Gameplay GIF" width="300">
</div>

## Overview

This project transforms the classic Flappy Bird into a nostalgic arcade experience, featuring neon colors, pixelated sprites, and effects like explosions and score flashes. A Q-learning agent trains to master the game with adaptive difficulty, while players can also take control manually. The core challenge remains: guide the bird through pipes by flapping its wings, avoiding collisions.

The AI leverages a Q-table with experience replay for learning, paired with a retro frontend that evokes 90s arcade vibes.

## Key Features

- **90s Arcade Style**: Neon palette, animated pixel bird, tiled backgrounds, and collision effects.
- **Q-Learning AI**: Reinforcement learning with a replay buffer for stable training.
- **Adaptive Difficulty**: Four levels (Easy, Normal, Hard, Expert) adjusting pipe gaps dynamically.
- **Visual Flourishes**: Flap particles, flashing scores, and explosion animations.
- **Modes**:
  - Train or refine an AI agent.
  - Visualize AI performance.
  - Play manually with keyboard input.
- **Arcade Menu**: Navigate options with UP/DOWN keys and ENTER.
- **Persistence**: Save and load the AI’s Q-table for ongoing training.

## Installation

### Requirements
- Python 3.6+
- Libraries: `pygame`, `numpy`

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/samarthya04/Flappy-Bird.git
   cd Flappy-Bird
   ```

2. **Install Dependencies**:
   ```bash
   pip install pygame numpy
   ```

3. **Launch the Game**:
   ```bash
   python mainV2.py
   ```

## How to Use

### Starting the Game
Run `python mainV2.py` to access the main menu. Use **UP** and **DOWN** keys to navigate, and **ENTER** to select.

### Menu Options
- **Train Agent**: Starts a 500-episode training session (resumes from `q_agent.pkl` if present). Press `S` to save manually.
- **Load & Visualize Agent**: Runs 5 episodes with a loaded agent.
- **Play Game**: Human mode—press **SPACE** to flap.
- **Exit**: Closes the game.

### Training Insights
- Saves Q-table to `q_agent.pkl` every 100 episodes and on completion.
- Prints progress (score, average, epsilon) every 10 episodes.

### Human Mode
Flap with **SPACE** to dodge pipes. Collisions reset the game, showing your score.

## Project Structure
```
Flappy-Bird/
├── mainV2.py       # Core game script
├── q_agent.pkl     # Saved Q-table (auto-generated)
├── README.md       # Documentation
└── gameplay.gif    # Gameplay demo
```

## Technical Notes
- **Graphics**: Pygame-driven with custom sprites for bird, pipes, and effects.
- **AI**: Q-learning with discretized states (height, velocity, pipe distance) and replay buffer.
- **Difficulty**: Pipe gaps adjust at scores 10, 20, and 30.
- **Dependencies**: Pygame for visuals, NumPy for computations.

## Contributing
Fork this repo and submit pull requests with your enhancements. Suggestions:
- Add retro sound effects (beeps, boops).
- Include a persistent high score feature.
- Introduce dynamic background elements (clouds, stars).

## License
Open-source under the [MIT License](LICENSE).

## Credits
Created by Samarthya Earnest Chattree. Inspired by Flappy Bird and 90s arcade classics.
