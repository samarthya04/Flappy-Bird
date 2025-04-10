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

# 90s Arcade Colors
DARK_BLUE = (25, 25, 112)
NEON_GREEN = (57, 255, 20)
NEON_PINK = (255, 20, 147)
ARCADE_YELLOW = (255, 215, 0)
ARCADE_RED = (255, 69, 0)
ARCADE_PURPLE = (148, 0, 211)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
ELECTRIC_BLUE = (10, 10, 220)

# Set up display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Retro Flappy Bird - 90s Arcade Edition")
clock = pygame.time.Clock()
arcade_font = pygame.font.SysFont('Arial', 16, bold=True)

# Load and create assets
def create_arcade_assets():
    assets = {}
    
    # Create bird animation frames
    bird_frames = []
    for i in range(3):
        # Create a surface for each bird frame
        bird_surface = pygame.Surface((BIRD_SIZE, BIRD_SIZE), pygame.SRCALPHA)
        
        # Draw bird body (base oval shape)
        pygame.draw.ellipse(bird_surface, ARCADE_YELLOW, (0, 0, BIRD_SIZE, BIRD_SIZE))
        
        # Draw wing (different positions for animation)
        wing_offset = i * 3
        wing_points = [(BIRD_SIZE//4, BIRD_SIZE//2), 
                       (BIRD_SIZE//4 - 2, BIRD_SIZE//2 + wing_offset),
                       (BIRD_SIZE//2, BIRD_SIZE//2 + wing_offset + 5)]
        pygame.draw.polygon(bird_surface, ARCADE_RED, wing_points)
        
        # Draw eye
        pygame.draw.circle(bird_surface, WHITE, (BIRD_SIZE-10, BIRD_SIZE//3), 5)
        pygame.draw.circle(bird_surface, BLACK, (BIRD_SIZE-8, BIRD_SIZE//3), 2)
        
        # Draw pixel-style beak
        beak_points = [(BIRD_SIZE-2, BIRD_SIZE//2), 
                       (BIRD_SIZE+5, BIRD_SIZE//2-3),
                       (BIRD_SIZE+5, BIRD_SIZE//2+3)]
        pygame.draw.polygon(bird_surface, ARCADE_RED, beak_points)
        
        # Add black outline (pixel style)
        pygame.draw.ellipse(bird_surface, BLACK, (0, 0, BIRD_SIZE, BIRD_SIZE), 2)
        
        bird_frames.append(bird_surface)
    
    assets['bird_frames'] = bird_frames
    
    # Create background tiles (arcade style grid)
    bg_tile = pygame.Surface((40, 40))
    bg_tile.fill(DARK_BLUE)
    pygame.draw.line(bg_tile, ELECTRIC_BLUE, (0, 0), (40, 0), 1)
    pygame.draw.line(bg_tile, ELECTRIC_BLUE, (0, 0), (0, 40), 1)
    pygame.draw.circle(bg_tile, ELECTRIC_BLUE, (20, 20), 2)
    assets['bg_tile'] = bg_tile
    
    # Create pipe textures
    pipe_top = pygame.Surface((PIPE_WIDTH, HEIGHT), pygame.SRCALPHA)
    pipe_bottom = pygame.Surface((PIPE_WIDTH, HEIGHT), pygame.SRCALPHA)
    
    # Base pipe color
    pipe_color = NEON_GREEN
    
    # Draw pipe body
    for pipe_surface in [pipe_top, pipe_bottom]:
        # Fill with base color
        pygame.draw.rect(pipe_surface, pipe_color, (0, 0, PIPE_WIDTH, HEIGHT))
        
        # Add highlights (arcade style)
        pygame.draw.line(pipe_surface, WHITE, (5, 0), (5, HEIGHT), 2)
        
        # Add edge
        pygame.draw.rect(pipe_surface, BLACK, (0, 0, PIPE_WIDTH, HEIGHT), 2)
        
        # Add pixel-style rivets
        for y in range(0, HEIGHT, 30):
            pygame.draw.circle(pipe_surface, BLACK, (PIPE_WIDTH//2, y), 3)
            pygame.draw.circle(pipe_surface, WHITE, (PIPE_WIDTH//2-1, y-1), 1)
    
    # Create special cap for pipe ends
    pipe_cap_height = 20
    pipe_cap = pygame.Surface((PIPE_WIDTH + 10, pipe_cap_height))
    pipe_cap.fill(ARCADE_RED)
    pygame.draw.rect(pipe_cap, BLACK, (0, 0, PIPE_WIDTH + 10, pipe_cap_height), 2)
    for x in range(5, PIPE_WIDTH + 5, 10):
        pygame.draw.line(pipe_cap, BLACK, (x, 0), (x, pipe_cap_height), 1)
        
    assets['pipe_top'] = pipe_top
    assets['pipe_bottom'] = pipe_bottom
    assets['pipe_cap'] = pipe_cap
    
    # Create explosion effect frames for collisions
    explosion_frames = []
    for size in [10, 20, 30, 25, 15, 5]:
        exp_surface = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
        pygame.draw.circle(exp_surface, ARCADE_YELLOW, (size, size), size)
        pygame.draw.circle(exp_surface, ARCADE_RED, (size, size), size*0.7)
        explosion_frames.append(exp_surface)
    assets['explosion'] = explosion_frames
    
    # Create score digit sprites (pixelated)
    digit_sprites = []
    for i in range(10):
        digit_surface = pygame.Surface((20, 30), pygame.SRCALPHA)
        text = arcade_font.render(str(i), True, NEON_PINK)
        # Add shadow for arcade effect
        shadow = arcade_font.render(str(i), True, BLACK)
        digit_surface.blit(shadow, (2, 2))
        digit_surface.blit(text, (0, 0))
        digit_sprites.append(digit_surface)
    assets['digits'] = digit_sprites
    
    # Create particles for flap effect
    flap_particles = []
    for i in range(5):
        size = 5 - i
        particle = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(particle, WHITE, (size//2, size//2), size//2)
        flap_particles.append(particle)
    assets['flap_particles'] = flap_particles
    
    return assets

# Create game assets
ASSETS = create_arcade_assets()

class FlappyBirdGame:
    def __init__(self):
        self.reset()
        self.bg_offset = 0
        self.animation_frame = 0
        self.animation_counter = 0
        self.particles = []  # For visual effects
        self.explosion_frame = -1  # For collision animation
        self.score_flash = 0  # For score flash effect

    def reset(self):
        self.bird_y = HEIGHT // 2
        self.bird_vel = 0
        self.pipes = []
        self.score = 0
        self.frames = 0
        self.spawn_pipe()
        self.particles = []
        self.explosion_frame = -1
        self.score_flash = 0
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
                # Add flap particles for visual effect
                for _ in range(3):
                    self.particles.append({
                        'x': 50 + BIRD_SIZE//2,
                        'y': self.bird_y + BIRD_SIZE//2,
                        'vx': random.uniform(-1, -3),
                        'vy': random.uniform(-0.5, 0.5),
                        'life': random.randint(5, 15),
                        'type': 'flap'
                    })
            
            self.bird_vel += GRAVITY
            self.bird_y += self.bird_vel
            self.frames += 1
            
            # Animate bird
            if self.frames % 5 == 0:
                self.animation_counter = (self.animation_counter + 1) % 3
                
            # Scroll background
            self.bg_offset = (self.bg_offset + 1) % 40

            # Update particles
            for particle in self.particles[:]:
                particle['x'] += particle['vx']
                particle['y'] += particle['vy']
                particle['life'] -= 1
                if particle['life'] <= 0:
                    self.particles.remove(particle)

            # Move pipes
            for pipe in self.pipes[:]:
                pipe[0] -= PIPE_SPEED
                if pipe[0] < -PIPE_WIDTH:
                    self.pipes.remove(pipe)
                    self.score += 1
                    self.score_flash = 10  # Flash score effect
                    if len(self.pipes) < 2:
                        self.spawn_pipe()

            # Check collisions
            bird_rect = pygame.Rect(50, self.bird_y, BIRD_SIZE-5, BIRD_SIZE-5)  # Slightly smaller hitbox for fairness
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
                    self.explosion_frame = 0  # Start explosion animation
                elif 50 <= next_pipe[0] + PIPE_WIDTH <= 50 + PIPE_SPEED:  # More precise passing pipe detection
                    frame_reward += 10
            
            if self.bird_y < 0 or self.bird_y > HEIGHT:
                done = True
                frame_reward = -100
                self.explosion_frame = 0  # Start explosion animation
            
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
        # Draw tiled background (arcade grid style)
        for y in range(0, HEIGHT, 40):
            for x in range(-(self.bg_offset % 40), WIDTH, 40):
                screen.blit(ASSETS['bg_tile'], (x, y))
        
        # Draw a "ground" line
        pygame.draw.rect(screen, NEON_GREEN, (0, HEIGHT-20, WIDTH, 20))
        for x in range(0, WIDTH, 20):
            pygame.draw.rect(screen, DARK_BLUE, (x, HEIGHT-20, 10, 10))
            pygame.draw.rect(screen, DARK_BLUE, (x+10, HEIGHT-10, 10, 10))
        
        # Draw pipes with arcade-style textures
        for pipe in self.pipes:
            # Draw top pipe
            screen.blit(ASSETS['pipe_top'], (pipe[0], 0), 
                       (0, 0, PIPE_WIDTH, pipe[1]))
            # Draw bottom pipe
            bottom_pipe_y = pipe[1] + PIPE_GAP
            screen.blit(ASSETS['pipe_bottom'], (pipe[0], bottom_pipe_y),
                       (0, 0, PIPE_WIDTH, HEIGHT - bottom_pipe_y))
            
            # Add pipe caps
            screen.blit(ASSETS['pipe_cap'], (pipe[0] - 5, pipe[1] - 20))
            screen.blit(ASSETS['pipe_cap'], (pipe[0] - 5, pipe[1] + PIPE_GAP))
        
        # Draw particles
        for particle in self.particles:
            if particle['type'] == 'flap':
                particle_idx = min(len(ASSETS['flap_particles'])-1, particle['life'] // 3)
                screen.blit(ASSETS['flap_particles'][particle_idx], 
                           (int(particle['x']), int(particle['y'])))
        
        # Calculate bird angle based on velocity (for rotation effect)
        bird_angle = max(-30, min(30, -self.bird_vel * 3))
        
        # Draw bird with animation
        bird_frame = ASSETS['bird_frames'][self.animation_counter]
        rotated_bird = pygame.transform.rotate(bird_frame, bird_angle)
        screen.blit(rotated_bird, (50, int(self.bird_y)))
        
        # Draw explosion effect if collision occurred
        if self.explosion_frame >= 0:
            if self.explosion_frame < len(ASSETS['explosion']):
                exp_surface = ASSETS['explosion'][self.explosion_frame]
                screen.blit(exp_surface, 
                           (50 + BIRD_SIZE//2 - exp_surface.get_width()//2, 
                            int(self.bird_y) + BIRD_SIZE//2 - exp_surface.get_height()//2))
                self.explosion_frame += 1
        
        # Draw arcade-style score with sprite digits
        score_str = str(self.score)
        score_width = len(score_str) * 20
        score_x = WIDTH//2 - score_width//2
        
        # Draw score with glowing effect when points increase
        glow_factor = 0
        if self.score_flash > 0:
            glow_factor = self.score_flash
            self.score_flash -= 1
            
            # Draw extra glow around score
            if glow_factor > 5:
                glow_surface = pygame.Surface((score_width + 20, 40), pygame.SRCALPHA)
                pygame.draw.rect(glow_surface, (255, 255, 255, glow_factor * 10), 
                                (0, 0, score_width + 20, 40), 
                                border_radius=10)
                screen.blit(glow_surface, (score_x - 10, 5))
        
        for i, digit in enumerate(score_str):
            digit_sprite = ASSETS['digits'][int(digit)]
            screen.blit(digit_sprite, (score_x + i*20, 10))
        
        # Draw arcade marquee-style title at top
        if self.frames % 20 < 10:  # Blinking effect
            title_text = arcade_font.render("* FLAPPY BIRD ARCADE *", True, NEON_PINK)
            screen.blit(title_text, (WIDTH//2 - title_text.get_width()//2, 5))
        
        # Display agent info if provided
        if agent:
            # Arcade-style info panel
            pygame.draw.rect(screen, BLACK, (5, HEIGHT-130, 180, 110))
            pygame.draw.rect(screen, NEON_PINK, (5, HEIGHT-130, 180, 110), 2)
            
            state = self.get_state()
            discrete_state = agent.discretize_state(state)
            
            # Display state information
            state_text = arcade_font.render(f"HEIGHT:{int(state[0])}", True, NEON_GREEN)
            screen.blit(state_text, (10, HEIGHT-120))
            
            vel_text = arcade_font.render(f"VEL:{int(state[1])}", True, NEON_GREEN)
            screen.blit(vel_text, (10, HEIGHT-100))
            
            dist_text = arcade_font.render(f"DIST:{int(state[2])}", True, NEON_GREEN)
            screen.blit(dist_text, (10, HEIGHT-80))
            
            # Display Q-values
            q_values = agent.q_table[discrete_state]
            q0_text = arcade_font.render(f"Q-HOLD:{q_values[0]:.2f}", True, ARCADE_YELLOW)
            screen.blit(q0_text, (10, HEIGHT-60))
            
            q1_text = arcade_font.render(f"Q-FLAP:{q_values[1]:.2f}", True, ARCADE_YELLOW)
            screen.blit(q1_text, (10, HEIGHT-40))
            
            # Display current action with arcade-style indicator
            if current_action is not None:
                action_label = "FLAP!" if current_action == 1 else "HOLD"
                action_color = ARCADE_RED if current_action == 1 else ARCADE_YELLOW
                action_text = arcade_font.render(action_label, True, action_color)
                pygame.draw.rect(screen, BLACK, (WIDTH-100, HEIGHT-40, 90, 30))
                pygame.draw.rect(screen, action_color, (WIDTH-100, HEIGHT-40, 90, 30), 2)
                screen.blit(action_text, (WIDTH-85, HEIGHT-35))
            
            # Display epsilon with arcade-style meter
            if hasattr(agent, 'epsilon'):
                pygame.draw.rect(screen, BLACK, (WIDTH-110, 10, 100, 20))
                meter_width = int(agent.epsilon * 100)
                pygame.draw.rect(screen, ARCADE_PURPLE, (WIDTH-110, 10, meter_width, 20))
                pygame.draw.rect(screen, WHITE, (WIDTH-110, 10, 100, 20), 1)
                eps_text = arcade_font.render(f"RAND:{agent.epsilon:.2f}", True, WHITE)
                screen.blit(eps_text, (WIDTH-100, 30))
            
            # Display difficulty level with arcade-style indicator
            if hasattr(agent, 'difficulty_level'):
                diff_colors = {
                    "Easy": NEON_GREEN,
                    "Normal": ARCADE_YELLOW,
                    "Hard": ARCADE_RED,
                    "Expert": NEON_PINK
                }
                diff_color = diff_colors.get(agent.difficulty_level, WHITE)
                diff_text = arcade_font.render(f"LEVEL:{agent.difficulty_level}", True, diff_color)
                screen.blit(diff_text, (WIDTH-110, 50))
        
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
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            
            # Agent decides action with minimal exploration
            action = agent.choose_action(state) if random.random() > 0.05 else random.choice([0, 1])
            
            # Take action and observe
            next_state, reward, done = game.step(action)
            
            # Update state
            state = next_state
            
            # Update adaptive difficulty if enabled
            if adaptive and difficulty_manager.update_difficulty(game.score):
                agent.difficulty_level = difficulty_manager.current_level
            
            # Render game with agent info
            game.render(agent, action)
            clock.tick(60)
        
        print(f"Episode {episode+1}/{episodes} | Score: {game.score}")

def play_human_mode():
    """Let a human play the game"""
    game = FlappyBirdGame()
    state = game.reset()
    running = True
    
    print("\nHuman Play Mode - Press SPACE to flap")
    
    while running:
        action = 0  # Default: no flap
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = 1  # Flap
        
        # Take action and observe
        next_state, reward, done = game.step(action)
        
        # Update state
        state = next_state
        
        # Render game without agent info
        game.render()
        
        if done:
            print(f"Game Over! Score: {game.score}")
            # Wait a moment to show the explosion animation
            pygame.time.wait(1000)
            state = game.reset()
        
        clock.tick(60)

def show_menu():
    """Display a menu to select game mode"""
    menu_running = True
    selected_option = 0
    options = ["Train Agent", "Load & Visualize Agent", "Play Game", "Exit"]
    
    while menu_running:
        screen.fill(DARK_BLUE)
        
        # Draw title
        title_font = pygame.font.SysFont('Arial', 40, bold=True)
        title_text = title_font.render("FLAPPY BIRD RL", True, NEON_PINK)
        screen.blit(title_text, (WIDTH//2 - title_text.get_width()//2, 80))
        
        # Draw options
        option_font = pygame.font.SysFont('Arial', 24)
        for i, option in enumerate(options):
            color = ARCADE_YELLOW if i == selected_option else WHITE
            option_text = option_font.render(option, True, color)
            screen.blit(option_text, (WIDTH//2 - option_text.get_width()//2, 200 + i*50))
            
            # Draw selection indicator
            if i == selected_option:
                pygame.draw.rect(screen, ARCADE_YELLOW, 
                               (WIDTH//2 - option_text.get_width()//2 - 20, 200 + i*50 + 10, 
                                10, 10))
        
        # Draw instructions
        inst_font = pygame.font.SysFont('Arial', 16)
        inst_text = inst_font.render("UP/DOWN: Select   ENTER: Confirm", True, WHITE)
        screen.blit(inst_text, (WIDTH//2 - inst_text.get_width()//2, HEIGHT - 50))
        
        pygame.display.flip()
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "Exit"
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected_option = (selected_option - 1) % len(options)
                elif event.key == pygame.K_DOWN:
                    selected_option = (selected_option + 1) % len(options)
                elif event.key == pygame.K_RETURN:
                    return options[selected_option]
        
        clock.tick(30)

if __name__ == "__main__":
    while True:
        choice = show_menu()
        
        if choice == "Train Agent":
            episodes = 500
            scores, agent = train_agent(episodes=episodes, load_agent=True, adaptive=True)
            print(f"Training completed. Highest score: {max(scores)}")
            
        elif choice == "Load & Visualize Agent":
            agent = QLearningAgent.load()
            visualize_agent(agent, episodes=5, adaptive=True)
            
        elif choice == "Play Game":
            play_human_mode()
            
        elif choice == "Exit":
            pygame.quit()
            break