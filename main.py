import pygame
import numpy as np
import random
import pickle
import os
from collections import defaultdict, deque

# Initialize Pygame
pygame.init()

# Game constants
WIDTH, HEIGHT = 400, 600
BIRD_SIZE = 30
PIPE_WIDTH = 60
PIPE_GAP = 150
GRAVITY = 0.6
FLAP_STRENGTH = -10
PIPE_SPEED = 4
FRAMESKIP = 4  # Process every nth frame for faster training

# Pipe generation constraints
MIN_PIPE_HEIGHT = 80  # Minimum height of bottom pipe
MAX_PIPE_HEIGHT = HEIGHT - 80 - PIPE_GAP  # Maximum height considering gap

# Colors
SKY_BLUE = (135, 206, 235)
GREEN = (0, 200, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)

# Set up display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird with Random Obstacles and Q-Learning")
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 16)

class FlappyBirdGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.bird_y = HEIGHT // 2
        self.bird_vel = 0
        self.pipes = []
        self.score = 0
        self.frames = 0
        self.spawn_pipe()
        return self.get_state()

    def spawn_pipe(self):
        # Generate random pipe height within constraints
        gap_top = random.randint(MIN_PIPE_HEIGHT, MAX_PIPE_HEIGHT)
        self.pipes.append([WIDTH, gap_top])

    def step(self, action):
        # Apply action and simulate multiple frames for frameskipping
        rewards = []
        next_state = None
        done = False
        
        for _ in range(FRAMESKIP):
            if _ == 0 and action == 1:  # Only apply flap on first frame
                self.bird_vel = FLAP_STRENGTH
            
            self.bird_vel += GRAVITY
            self.bird_y += self.bird_vel
            self.frames += 1

            # Move pipes
            for pipe in self.pipes[:]:
                pipe[0] -= PIPE_SPEED
                if pipe[0] < -PIPE_WIDTH:
                    self.pipes.remove(pipe)
                    self.score += 1
                    if len(self.pipes) < 2:
                        self.spawn_pipe()

            # Check collisions
            bird_rect = pygame.Rect(50, self.bird_y, BIRD_SIZE, BIRD_SIZE)
            frame_reward = 0.1  # Base survival reward
            
            next_pipe = self.pipes[0] if self.pipes else None
            if next_pipe:
                gap_center = next_pipe[1] + PIPE_GAP // 2
                dist_to_gap = abs(self.bird_y - gap_center)
                frame_reward += (100 - dist_to_gap) * 0.01  # Reward for proximity to gap
                
                top_pipe = pygame.Rect(next_pipe[0], 0, PIPE_WIDTH, next_pipe[1])
                bottom_pipe = pygame.Rect(next_pipe[0], next_pipe[1] + PIPE_GAP, PIPE_WIDTH, HEIGHT - next_pipe[1] - PIPE_GAP)
                if bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe):
                    done = True
                    frame_reward = -100
                elif 50 <= next_pipe[0] + PIPE_WIDTH <= 50 + PIPE_SPEED:  # More precise passing pipe detection
                    frame_reward += 10
            
            if self.bird_y < 0 or self.bird_y > HEIGHT:
                done = True
                frame_reward = -100
            
            rewards.append(frame_reward)
            next_state = self.get_state()
            
            if done:
                break
        
        # Return the final state and cumulative reward
        return next_state, sum(rewards), done

    def get_state(self):
        if not self.pipes:
            return (0, self.bird_vel, WIDTH)
            
        next_pipe = self.pipes[0]
        distance_to_pipe = next_pipe[0] - 50  # Distance from bird to next pipe
        gap_center = next_pipe[1] + PIPE_GAP // 2
        relative_height = self.bird_y - gap_center
        return (relative_height, self.bird_vel, distance_to_pipe)

    def render(self, agent=None, current_action=None):
        screen.fill(SKY_BLUE)
        
        # Draw pipes
        for pipe in self.pipes:
            # Draw top pipe
            pygame.draw.rect(screen, GREEN, (pipe[0], 0, PIPE_WIDTH, pipe[1]))
            pygame.draw.rect(screen, BLACK, (pipe[0], pipe[1] - 10, PIPE_WIDTH, 10))
            
            # Draw bottom pipe
            bottom_pipe_y = pipe[1] + PIPE_GAP
            pygame.draw.rect(screen, GREEN, (pipe[0], bottom_pipe_y, PIPE_WIDTH, HEIGHT - bottom_pipe_y))
            pygame.draw.rect(screen, BLACK, (pipe[0], bottom_pipe_y, PIPE_WIDTH, 10))
        
        # Draw bird
        pygame.draw.circle(screen, YELLOW, (50 + BIRD_SIZE // 2, int(self.bird_y) + BIRD_SIZE // 2), BIRD_SIZE // 2)
        pygame.draw.circle(screen, WHITE, (50 + BIRD_SIZE // 2 + 5, int(self.bird_y) + 5), 5)
        pygame.draw.circle(screen, BLACK, (50 + BIRD_SIZE // 2 + 5, int(self.bird_y) + 5), 2)
        pygame.draw.polygon(screen, RED, [(50 + BIRD_SIZE, self.bird_y + 5), (50 + BIRD_SIZE + 10, self.bird_y + 10), 
                                          (50 + BIRD_SIZE, self.bird_y + 15)])
        
        # Display score
        score_text = font.render(f"Score: {self.score}", True, BLACK)
        screen.blit(score_text, (10, 10))
        
        # Display agent info if provided
        if agent:
            state = self.get_state()
            discrete_state = agent.discretize_state(state)
            
            # Display state information
            state_text = font.render(f"State: Height={int(state[0])}, Vel={int(state[1])}, Dist={int(state[2])}", True, BLACK)
            screen.blit(state_text, (10, 30))
            
            # Display Q-values if available
            q_values = agent.q_table[discrete_state]
            q_text = font.render(f"Q: No-Flap={q_values[0]:.2f}, Flap={q_values[1]:.2f}", True, BLACK)
            screen.blit(q_text, (10, 50))
            
            # Indicate current action
            if current_action is not None:
                action_text = font.render(f"Action: {'Flap' if current_action == 1 else 'No-Flap'}", True, BLUE)
                screen.blit(action_text, (10, 70))
            
            # Display epsilon value
            epsilon_text = font.render(f"Epsilon: {agent.epsilon:.4f}", True, BLACK)
            screen.blit(epsilon_text, (10, 90))
            
            # Display current difficulty
            if hasattr(agent, 'difficulty_level'):
                difficulty_text = font.render(f"Difficulty: {agent.difficulty_level}", True, BLACK)
                screen.blit(difficulty_text, (10, 110))
        
        pygame.display.flip()

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
    
    def __len__(self):
        return len(self.buffer)

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.5, epsilon_decay=0.995, epsilon_min=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = defaultdict(lambda: [0.0, 0.0])
        self.actions = [0, 1]
        self.replay_buffer = ReplayBuffer()
        self.batch_size = 32
        self.difficulty_level = "Normal"  # Track difficulty level

    def discretize_state(self, state):
        # Discretize the continuous state space
        relative_height, vel, distance = state
        # More fine-grained discretization
        rel_height_disc = int(relative_height // 20)
        vel_disc = int(vel // 2)
        dist_disc = int(min(distance, 400) // 50)  # Cap distance and discretize
        return (rel_height_disc, vel_disc, dist_disc)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        discrete_state = self.discretize_state(state)
        q_values = self.q_table[discrete_state]
        return np.argmax(q_values)

    def update_q_table(self, state, action, reward, next_state, done):
        # Store experience in replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        # Perform experience replay if enough samples
        if len(self.replay_buffer) >= self.batch_size:
            self._experience_replay()
    
    def _experience_replay(self):
        # Sample a batch of experiences and update Q-values
        batch = self.replay_buffer.sample(self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            current_state = self.discretize_state(state)
            if not done:
                next_state_disc = self.discretize_state(next_state)
                next_max_q = max(self.q_table[next_state_disc])
                target = reward + self.gamma * next_max_q
            else:
                target = reward
            
            # Update Q-value
            current_q = self.q_table[current_state][action]
            self.q_table[current_state][action] = current_q + self.alpha * (target - current_q)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filename='q_agent.pkl'):
        """Save the agent's Q-table and parameters to a file"""
        # Convert defaultdict to regular dict for saving
        q_table_dict = dict(self.q_table)
        
        data = {
            'q_table': q_table_dict,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'difficulty_level': self.difficulty_level
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Agent saved to {filename}")
    
    @classmethod
    def load(cls, filename='q_agent.pkl'):
        """Load an agent from a file"""
        if not os.path.exists(filename):
            print(f"File {filename} not found. Creating new agent.")
            return cls()
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        agent = cls(alpha=data['alpha'], gamma=data['gamma'], epsilon=data['epsilon'])
        
        # Convert dict back to defaultdict
        q_table_dict = data['q_table']
        for state, actions in q_table_dict.items():
            agent.q_table[state] = actions
        
        # Load difficulty level if present
        if 'difficulty_level' in data:
            agent.difficulty_level = data['difficulty_level']
        
        print(f"Agent loaded from {filename}")
        return agent

class AdaptiveDifficulty:
    """Manages adaptive difficulty based on agent performance"""
    def __init__(self, game):
        self.game = game
        self.difficulty_levels = {
            "Easy": {"min_height": 150, "max_height": HEIGHT - 150 - PIPE_GAP},
            "Normal": {"min_height": 100, "max_height": HEIGHT - 100 - PIPE_GAP},
            "Hard": {"min_height": 80, "max_height": HEIGHT - 80 - PIPE_GAP},
            "Expert": {"min_height": 60, "max_height": HEIGHT - 60 - PIPE_GAP}
        }
        self.current_level = "Normal"
        self.score_thresholds = {
            "Easy": 10,    # Move from Easy to Normal at score 10
            "Normal": 20,  # Move from Normal to Hard at score 20
            "Hard": 30     # Move from Hard to Expert at score 30
        }
    
    def update_difficulty(self, score):
        """Update difficulty based on current score"""
        previous_level = self.current_level
        
        # Check if we should increase difficulty
        if self.current_level in self.score_thresholds and score >= self.score_thresholds[self.current_level]:
            levels = list(self.difficulty_levels.keys())
            current_index = levels.index(self.current_level)
            if current_index < len(levels) - 1:
                self.current_level = levels[current_index + 1]
        
        # If difficulty changed, return True
        return previous_level != self.current_level
    
    def get_pipe_height_range(self):
        """Get min and max pipe heights for current difficulty"""
        config = self.difficulty_levels[self.current_level]
        return config["min_height"], config["max_height"]

def train_agent(episodes=800, load_agent=False, save_interval=100, adaptive=True):
    game = FlappyBirdGame()
    agent = QLearningAgent.load() if load_agent and os.path.exists('q_agent.pkl') else QLearningAgent()
    scores = []
    running = True
    
    # Initialize adaptive difficulty if enabled
    difficulty_manager = AdaptiveDifficulty(game) if adaptive else None
    
    # Override spawn_pipe method if using adaptive difficulty
    if adaptive:
        original_spawn_pipe = game.spawn_pipe
        def adaptive_spawn_pipe():
            min_height, max_height = difficulty_manager.get_pipe_height_range()
            gap_top = random.randint(min_height, max_height)
            game.pipes.append([WIDTH, gap_top])
        game.spawn_pipe = adaptive_spawn_pipe
        agent.difficulty_level = difficulty_manager.current_level
    
    print("Training Progress:")
    for episode in range(episodes):
        if not running:
            break
            
        state = game.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and running:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:  # Save on 'S' key press
                        agent.save()
            
            # Agent decides action
            action = agent.choose_action(state)
            
            # Take action and observe
            next_state, reward, done = game.step(action)
            
            # Update agent
            agent.update_q_table(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            total_reward += reward
            steps += 1
            
            # Update adaptive difficulty if enabled
            if adaptive and difficulty_manager.update_difficulty(game.score):
                agent.difficulty_level = difficulty_manager.current_level
                print(f"Difficulty increased to {difficulty_manager.current_level} at score {game.score}")
            
            # Render game with agent info
            game.render(agent, action)
            clock.tick(60)
        
        # End of episode updates
        scores.append(game.score)
        agent.decay_epsilon()
        
        # Save agent periodically
        if episode % save_interval == 0:
            agent.save(f'q_agent_episode_{episode}.pkl')
        
        # Print progress
        if episode % 10 == 0:
            avg_score = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
            print(f"Episode {episode}/{episodes} | Steps: {steps} | Score: {game.score} | "
                  f"Avg Score (last 50): {avg_score:.2f} | Epsilon: {agent.epsilon:.4f}")
    
    # Final save
    agent.save()
    return scores, agent

def visualize_agent(agent, episodes=10, adaptive=True):
    """Visualize the agent's performance"""
    game = FlappyBirdGame()
    
    # Initialize adaptive difficulty if enabled
    difficulty_manager = AdaptiveDifficulty(game) if adaptive else None
    
    # Override spawn_pipe method if using adaptive difficulty
    if adaptive:
        original_spawn_pipe = game.spawn_pipe
        def adaptive_spawn_pipe():
            min_height, max_height = difficulty_manager.get_pipe_height_range()
            gap_top = random.randint(min_height, max_height)
            game.pipes.append([WIDTH, gap_top])
        game.spawn_pipe = adaptive_spawn_pipe
        difficulty_manager.current_level = agent.difficulty_level
    
    print("\nVisualization Phase:")
    running = True
    
    for episode in range(episodes):
        if not running:
            break
            
        state = game.reset()
        done = False
        
        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        difficulty_manager.current_level = "Easy"
                        print(f"Difficulty set to Easy")
                    elif event.key == pygame.K_2:
                        difficulty_manager.current_level = "Normal"
                        print(f"Difficulty set to Normal")
                    elif event.key == pygame.K_3:
                        difficulty_manager.current_level = "Hard"
                        print(f"Difficulty set to Hard")
                    elif event.key == pygame.K_4:
                        difficulty_manager.current_level = "Expert"
                        print(f"Difficulty set to Expert")
            
            # No exploration during visualization
            old_epsilon = agent.epsilon
            agent.epsilon = 0
            action = agent.choose_action(state)
            agent.epsilon = old_epsilon
            
            state, _, done = game.step(action)
            
            # Update adaptive difficulty if enabled
            if adaptive and difficulty_manager.update_difficulty(game.score):
                print(f"Difficulty increased to {difficulty_manager.current_level} at score {game.score}")
            
            game.render(agent, action)
            clock.tick(30)  # Slower for better visualization
        
        print(f"Episode {episode+1}: Score = {game.score}")

def evaluate_agent(agent, episodes=100, adaptive=True):
    """Evaluate the agent's performance without rendering"""
    game = FlappyBirdGame()
    scores = []
    
    # Initialize adaptive difficulty if enabled
    difficulty_manager = AdaptiveDifficulty(game) if adaptive else None
    
    # Override spawn_pipe method if using adaptive difficulty
    if adaptive:
        original_spawn_pipe = game.spawn_pipe
        def adaptive_spawn_pipe():
            min_height, max_height = difficulty_manager.get_pipe_height_range()
            gap_top = random.randint(min_height, max_height)
            game.pipes.append([WIDTH, gap_top])
        game.spawn_pipe = adaptive_spawn_pipe
        difficulty_manager.current_level = agent.difficulty_level
    
    print("\nEvaluation Phase:")
    old_epsilon = agent.epsilon
    agent.epsilon = 0  # No exploration during evaluation
    
    results_by_difficulty = {level: [] for level in difficulty_manager.difficulty_levels.keys()}
    
    # Test on each difficulty level
    for difficulty in difficulty_manager.difficulty_levels.keys():
        difficulty_manager.current_level = difficulty
        difficulty_scores = []
        
        for episode in range(episodes // len(difficulty_manager.difficulty_levels)):
            state = game.reset()
            done = False
            while not done:
                action = agent.choose_action(state)
                state, _, done = game.step(action)
            difficulty_scores.append(game.score)
            scores.append(game.score)
        
        results_by_difficulty[difficulty] = difficulty_scores
        avg_score = np.mean(difficulty_scores)
        print(f"Difficulty {difficulty}: Avg Score = {avg_score:.2f}, Max = {max(difficulty_scores)}")
    
    agent.epsilon = old_epsilon  # Restore epsilon
    
    avg_score = np.mean(scores)
    print(f"\nOverall Evaluation Results over {episodes} episodes:")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Max Score: {max(scores)}")
    print(f"Min Score: {min(scores)}")
    return scores, results_by_difficulty

def human_play():
    """Allow human player to try the game"""
    game = FlappyBirdGame()
    difficulty_manager = AdaptiveDifficulty(game)
    
    # Override spawn_pipe method for adaptive difficulty
    original_spawn_pipe = game.spawn_pipe
    def adaptive_spawn_pipe():
        min_height, max_height = difficulty_manager.get_pipe_height_range()
        gap_top = random.randint(min_height, max_height)
        game.pipes.append([WIDTH, gap_top])
    game.spawn_pipe = adaptive_spawn_pipe
    
    print("\nHuman Play Mode - Press SPACE to flap, 1-4 to change difficulty")
    running = True
    state = game.reset()
    done = False
    
    while running:
        if done:
            print(f"Game over! Score: {game.score}")
            state = game.reset()
            done = False
        
        action = 0  # Default: no flap
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = 1  # Flap
                elif event.key == pygame.K_1:
                    difficulty_manager.current_level = "Easy"
                    print(f"Difficulty set to Easy")
                elif event.key == pygame.K_2:
                    difficulty_manager.current_level = "Normal"
                    print(f"Difficulty set to Normal")
                elif event.key == pygame.K_3:
                    difficulty_manager.current_level = "Hard"
                    print(f"Difficulty set to Hard")
                elif event.key == pygame.K_4:
                    difficulty_manager.current_level = "Expert"
                    print(f"Difficulty set to Expert")
        
        state, _, done = game.step(action)
        
        # Update adaptive difficulty
        if difficulty_manager.update_difficulty(game.score):
            print(f"Difficulty increased to {difficulty_manager.current_level} at score {game.score}")
        
        # Render custom info for human player
        game.render()
        
        # Display current difficulty
        difficulty_text = font.render(f"Difficulty: {difficulty_manager.current_level}", True, BLACK)
        screen.blit(difficulty_text, (10, 30))
        
        # Display controls
        controls_text = font.render("SPACE: Flap | 1-4: Change Difficulty", True, BLACK)
        screen.blit(controls_text, (10, 50))
        
        pygame.display.flip()
        clock.tick(30)

def main():
    """Main function with menu options"""
    running = True
    
    while running:
        screen.fill(SKY_BLUE)
        title = pygame.font.SysFont('Arial', 32).render("Flappy Bird Q-Learning", True, BLACK)
        screen.blit(title, (WIDTH//2 - title.get_width()//2, 100))
        
        options = [
            "1. Train New Agent",
            "2. Continue Training Saved Agent",
            "3. Visualize Trained Agent",
            "4. Evaluate Agent Performance",
            "5. Human Play Mode",
            "6. Exit"
        ]
        
        for i, option in enumerate(options):
            text = font.render(option, True, BLACK)
            screen.blit(text, (WIDTH//2 - text.get_width()//2, 200 + i*40))
        
        pygame.display.flip()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    print("Training new agent...")
                    scores, agent = train_agent(episodes=800)
                elif event.key == pygame.K_2:
                    print("Continuing training...")
                    scores, agent = train_agent(episodes=300, load_agent=True)
                elif event.key == pygame.K_3:
                    print("Visualizing agent...")
                    agent = QLearningAgent.load()
                    visualize_agent(agent)
                elif event.key == pygame.K_4:
                    print("Evaluating agent...")
                    agent = QLearningAgent.load()
                    evaluate_agent(agent)
                elif event.key == pygame.K_5:
                    print("Starting human play mode...")
                    human_play()
                elif event.key == pygame.K_6:
                    running = False
    
    pygame.quit()

if __name__ == "__main__":
    main()