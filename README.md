# Flappy Bird: Q-Learning

A modern take on the classic Flappy Bird game with a 90s arcade aesthetic, powered by Q-learning reinforcement learning. This project combines retro-style graphics with an AI agent that learns to navigate through pipes, offering both training and human-playable modes.

<div align="center"> <img src="gameplay.gif" alt="Gameplay GIF" width="300"> </div>

## Project Description

This project reimagines Flappy Bird with a vibrant 90s arcade look, featuring neon colors, pixelated sprites, and classic visual effects like explosion animations and score flashing. The game includes a Q-learning agent that trains to master the game, with adaptive difficulty levels to challenge its skills. Players can train the AI, visualize its performance, or play the game manually.

The core mechanics remain true to Flappy Bird: navigate a bird through a series of pipes by flapping its wings, avoiding collisions. The AI uses a Q-table with experience replay to learn optimal actions, while the retro frontend adds a nostalgic arcade flair.

## Features

- **90s Arcade Aesthetics**: Neon colors, pixelated bird with animation frames, tiled background, and explosion effects.
- **Q-Learning AI**: An agent that learns to play using reinforcement learning with a replay buffer for improved training stability.
- **Adaptive Difficulty**: Four difficulty levels (Easy, Normal, Hard, Expert) that adjust pipe gaps based on performance.
- **Visual Effects**: Flap particles, score flashing, and collision explosions for an authentic arcade feel.
- **Game Modes**:
  - Train a new AI agent or continue training a saved one.
  - Visualize the trained agent's performance.
  - Play manually with keyboard controls.
- **Menu System**: Arcade-style menu with keyboard navigation for mode selection.
- **Save/Load**: Persist the agent's Q-table and parameters to disk for continued training.

## Installation

### Prerequisites
- Python 3.6+
- Required libraries: `pygame`, `numpy`

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/samarthya04/Flappy-Bird.git
   cd Flappy-Bird
   ```

2. **Install Dependencies**:
   ```bash
   pip install pygame numpy
   ```

3. **Run the Game**:
   ```bash
   python mainV2.py
   ```

## Usage

### Running the Game
- Launch the script (`python mainV2.py`) to access the main menu.
- Use the **UP** and **DOWN** arrow keys to navigate the menu, and **ENTER** to select an option.

### Menu Options
1. **Train Agent**: Trains a new Q-learning agent for 500 episodes (or continues from a saved state if `q_agent.pkl` exists).
   - Press `S` during training to save progress manually.
2. **Load & Visualize Agent**: Loads a trained agent and runs 5 visualization episodes.
3. **Play Game**: Enter human play mode.
   - Press **SPACE** to flap the bird.
4. **Exit**: Quit the game.

### Training Details
- The agent saves its Q-table to `q_agent.pkl` every 100 episodes and at the end of training.
- Progress is printed every 10 episodes, showing score, average score, and epsilon value.

### Human Play Mode
- Navigate the bird through pipes by pressing **SPACE** to flap.
- The game resets automatically after a collision, displaying your score.

## Project Structure
```
retro-flappy-bird/
├── main.py       # Main game script
├── q_agent.pkl          # Saved Q-table (generated after training)
├── README.md          
└── gameplay.gif         # GIF of the game
```

## Technical Details
- **Graphics**: Uses Pygame for rendering with custom sprite creation for the bird, pipes, and effects.
- **AI**: Implements Q-learning with a discretized state space (relative height, velocity, distance to pipe) and experience replay.
- **Adaptive Difficulty**: Adjusts pipe gap sizes based on score thresholds (10, 20, 30).
- **Dependencies**: Pygame for graphics, NumPy for numerical operations.

## Contributing
Feel free to fork this repository and submit pull requests with improvements! Ideas for enhancement:
- Add sound effects (beeps, boops) for a fuller arcade experience.
- Implement a high score system with persistent storage.
- Enhance the background with moving elements (e.g., clouds or stars).

## License
This project is open-source and available under the [MIT License](LICENSE).

## Credits
Developed by Samarthya Earnest Chattree. Inspired by the original Flappy Bird and classic 90s arcade games.
