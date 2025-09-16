import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
import random
import time

class DepthFlappyBird:
    def __init__(self):
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        try:
            self.pipeline.start(config)
        except Exception as e:
            print(f"Error starting RealSense camera: {e}")
            print("Make sure your RealSense camera is connected")
            return

        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Game parameters
        self.width = 640
        self.height = 480
        self.bird_x = 100
        self.bird_y = 240
        self.bird_radius = 15

        # Depth control parameters
        self.min_depth = 300  # mm
        self.max_depth = 800  # mm
        self.depth_smoothing = 0.3  # Lower = more smoothing
        self.smoothed_depth = 500
        self.hand_detected = False

        # Pipe parameters
        self.pipe_width = 60
        self.pipe_gap = 150
        self.pipe_speed = 3
        self.pipes = []

        # Game state
        self.score = 0
        self.game_over = False
        self.level = 1
        self.targets_per_level = 5

        # Colors
        self.bird_color = (0, 255, 255)  # Yellow
        self.pipe_color = (0, 128, 0)   # Green
        self.bg_color = (135, 206, 235) # Sky blue

        # Initialize first pipes
        self.spawn_pipe()

    def spawn_pipe(self):
        """Spawn a new pipe at the right edge"""
        gap_center = random.randint(100, self.height - 100)
        pipe = {
            'x': self.width,
            'top': gap_center - self.pipe_gap // 2,
            'bottom': gap_center + self.pipe_gap // 2,
            'scored': False
        }
        self.pipes.append(pipe)

    def update_level(self):
        """Update game level and speed"""
        new_level = (self.score // self.targets_per_level) + 1
        if new_level > self.level:
            self.level = new_level
            self.pipe_speed = min(3 + (self.level - 1) * 0.5, 8)  # Cap max speed

    def get_hand_depth(self, frame, depth_frame):
        """Detect hand and get depth at bounding box centroid"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    try:
                        # Draw hand landmarks
                        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                        # Get all landmark coordinates
                        landmarks = hand_landmarks.landmark
                        if not landmarks or len(landmarks) == 0:
                            continue

                        x_coords = [lm.x for lm in landmarks if lm.x is not None]
                        y_coords = [lm.y for lm in landmarks if lm.y is not None]

                        if not x_coords or not y_coords:
                            continue

                        # Calculate bounding box with validation
                        min_x = int(min(x_coords) * self.width)
                        max_x = int(max(x_coords) * self.width)
                        min_y = int(min(y_coords) * self.height)
                        max_y = int(max(y_coords) * self.height)

                        # Validate bounding box dimensions
                        if min_x >= max_x or min_y >= max_y:
                            continue

                        # Add padding and ensure bounds are valid
                        padding = 10
                        min_x = max(0, min_x - padding)
                        max_x = min(self.width - 1, max_x + padding)
                        min_y = max(0, min_y - padding)
                        max_y = min(self.height - 1, max_y + padding)

                        # Final validation after padding
                        if min_x >= max_x or min_y >= max_y:
                            continue

                        # Calculate bounding box centroid
                        bbox_center_x = (min_x + max_x) // 2
                        bbox_center_y = (min_y + max_y) // 2

                        # Validate centroid coordinates
                        if not (0 <= bbox_center_x < self.width and 0 <= bbox_center_y < self.height):
                            continue

                        # Since color frame is flipped, flip x coordinate for depth lookup
                        depth_x = self.width - bbox_center_x - 1
                        depth_y = bbox_center_y

                        # Sample depth values in a region around the bounding box center
                        depth_values = []
                        sample_radius = 15  # Sample in 30x30 area around center

                        for dx in range(-sample_radius, sample_radius + 1, 3):  # Step by 3 for efficiency
                            for dy in range(-sample_radius, sample_radius + 1, 3):
                                try:
                                    px = max(0, min(self.width - 1, depth_x + dx))
                                    py = max(0, min(self.height - 1, depth_y + dy))
                                    depth_val = depth_frame.get_distance(px, py) * 1000  # Convert to mm
                                    if 200 < depth_val < 1500:  # Valid depth range for hand detection
                                        depth_values.append(depth_val)
                                except Exception:
                                    continue

                        if len(depth_values) > 5:  # Reduced threshold for more reliable detection
                            # Use median depth for stability
                            depth_value = np.median(depth_values)

                            # Initialize smoothed_depth if first detection
                            if not hasattr(self, 'smoothed_depth') or self.smoothed_depth == 0:
                                self.smoothed_depth = depth_value

                            # Moderate smoothing to balance responsiveness and stability
                            self.smoothed_depth = (0.6 * depth_value + 0.4 * self.smoothed_depth)
                            self.hand_detected = True

                            # Draw bounding box with validation
                            try:
                                cv2.rectangle(frame, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)

                                # Draw bounding box center
                                cv2.circle(frame, (int(bbox_center_x), int(bbox_center_y)), 8, (255, 0, 0), -1)
                                cv2.circle(frame, (int(bbox_center_x), int(bbox_center_y)), 12, (255, 255, 255), 2)

                                # Display depth info with safe positioning
                                text_y = max(25, min_y)
                                cv2.putText(frame, f"Depth: {int(depth_value)}mm",
                                          (int(min_x), int(text_y - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                                cv2.putText(frame, f"Smooth: {int(self.smoothed_depth)}mm",
                                          (int(min_x), int(text_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 255), 1)
                                cv2.putText(frame, f"Samples: {len(depth_values)}",
                                          (int(min_x), int(min(self.height - 5, max_y + 15))), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 200), 1)
                            except Exception as e:
                                print(f"Drawing error: {e}")
                                # Continue without drawing if there's an issue
                                pass

                            return True

                    except Exception as e:
                        print(f"Hand processing error: {e}")
                        continue

            # No hand detected
            self.hand_detected = False
            return False

        except Exception as e:
            print(f"Hand depth detection error: {e}")
            self.hand_detected = False
            return False

    def update_bird_position(self):
        """Update bird position based on hand depth"""
        if self.hand_detected:
            # Map depth to bird Y position (closer = higher)
            depth_range = self.max_depth - self.min_depth
            depth_normalized = max(0, min(1, (self.max_depth - self.smoothed_depth) / depth_range))
            target_y = int(depth_normalized * (self.height - 2 * self.bird_radius) + self.bird_radius)

            # Smooth bird movement
            self.bird_y = int(0.7 * target_y + 0.3 * self.bird_y)

        # Keep bird within bounds
        self.bird_y = max(self.bird_radius, min(self.height - self.bird_radius, self.bird_y))

    def update_pipes(self):
        """Update pipe positions and handle collisions"""
        for pipe in self.pipes[:]:
            pipe['x'] -= self.pipe_speed

            # Remove pipes that have moved off screen
            if pipe['x'] + self.pipe_width < 0:
                self.pipes.remove(pipe)
                continue

            # Check for scoring
            if not pipe['scored'] and pipe['x'] + self.pipe_width < self.bird_x:
                pipe['scored'] = True
                self.score += 1
                self.update_level()

            # Check for collision
            if (self.bird_x + self.bird_radius > pipe['x'] and
                self.bird_x - self.bird_radius < pipe['x'] + self.pipe_width):
                if (self.bird_y - self.bird_radius < pipe['top'] or
                    self.bird_y + self.bird_radius > pipe['bottom']):
                    self.game_over = True

        # Spawn new pipes
        if len(self.pipes) == 0 or self.pipes[-1]['x'] < self.width - 200:
            self.spawn_pipe()

    def draw_game(self, frame):
        """Draw all game elements"""
        try:
            # Clear background
            game_surface = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)

            # Draw pipes with validation
            for pipe in self.pipes:
                try:
                    # Validate pipe coordinates
                    x1 = int(pipe['x'])
                    x2 = int(pipe['x'] + self.pipe_width)
                    top = int(pipe['top'])
                    bottom = int(pipe['bottom'])

                    # Ensure coordinates are within bounds and valid
                    x1 = max(0, min(self.width - 1, x1))
                    x2 = max(x1 + 1, min(self.width, x2))
                    top = max(0, min(self.height - 1, top))
                    bottom = max(top + 1, min(self.height, bottom))

                    # Only draw if pipe is visible
                    if x2 > 0 and x1 < self.width:
                        # Top pipe
                        if top > 0:
                            cv2.rectangle(game_surface,
                                         (x1, 0),
                                         (x2, top),
                                         self.pipe_color, -1)
                            cv2.rectangle(game_surface,
                                         (x1, 0),
                                         (x2, top),
                                         (0, 100, 0), 3)

                        # Bottom pipe
                        if bottom < self.height:
                            cv2.rectangle(game_surface,
                                         (x1, bottom),
                                         (x2, self.height),
                                         self.pipe_color, -1)
                            cv2.rectangle(game_surface,
                                         (x1, bottom),
                                         (x2, self.height),
                                         (0, 100, 0), 3)

                except Exception as e:
                    print(f"Pipe drawing error: {e}, pipe data: {pipe}")
                    continue

            # Draw bird with validation
            try:
                bird_x = int(self.bird_x)
                bird_y = int(self.bird_y)
                bird_radius = int(self.bird_radius)

                # Ensure bird coordinates are valid
                if 0 <= bird_x < self.width and 0 <= bird_y < self.height:
                    cv2.circle(game_surface, (bird_x, bird_y), bird_radius, self.bird_color, -1)
                    cv2.circle(game_surface, (bird_x, bird_y), bird_radius, (0, 200, 200), 3)

                    # Draw eye
                    eye_x = bird_x + 5
                    eye_y = bird_y - 3
                    if 0 <= eye_x < self.width and 0 <= eye_y < self.height:
                        cv2.circle(game_surface, (eye_x, eye_y), 3, (0, 0, 0), -1)
            except Exception as e:
                print(f"Bird drawing error: {e}")

            # Draw UI with validation
            try:
                cv2.putText(game_surface, f"Score: {self.score}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(game_surface, f"Level: {self.level}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(game_surface, f"Speed: {self.pipe_speed:.1f}", (10, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Hand detection status
                status_color = (0, 255, 0) if self.hand_detected else (0, 0, 255)
                status_text = "Hand Detected" if self.hand_detected else "No Hand"
                cv2.putText(game_surface, status_text, (10, self.height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                # Instructions
                cv2.putText(game_surface, "Move hand closer/further to control bird",
                           (self.width - 350, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except Exception as e:
                print(f"UI drawing error: {e}")

            if self.game_over:
                try:
                    # Game over overlay
                    overlay = game_surface.copy()
                    cv2.rectangle(overlay, (150, 180), (490, 300), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.8, game_surface, 0.2, 0, game_surface)

                    cv2.putText(game_surface, "GAME OVER!", (200, 220),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.putText(game_surface, f"Final Score: {self.score}", (220, 250),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(game_surface, "Press 'R' to restart", (210, 280),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                except Exception as e:
                    print(f"Game over drawing error: {e}")

            # Combine camera frame and game with validation
            try:
                if frame is not None and game_surface is not None:
                    if frame.shape[0] == game_surface.shape[0]:  # Same height
                        combined = np.hstack([frame, game_surface])
                        return combined
                    else:
                        print(f"Frame size mismatch: frame={frame.shape}, game={game_surface.shape}")
                        return game_surface
                else:
                    return game_surface
            except Exception as e:
                print(f"Frame combination error: {e}")
                return game_surface

        except Exception as e:
            print(f"Draw game error: {e}")
            # Return a simple error frame
            error_surface = np.full((self.height, self.width, 3), (50, 50, 50), dtype=np.uint8)
            cv2.putText(error_surface, "Drawing Error", (200, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return error_surface

    def reset_game(self):
        """Reset game to initial state"""
        self.bird_y = 240
        self.pipes = []
        self.score = 0
        self.level = 1
        self.pipe_speed = 3
        self.game_over = False
        self.spawn_pipe()

    def run(self):
        """Main game loop"""
        print("Depth-Controlled Flappy Bird")
        print("Controls:")
        print("- Move your hand closer to camera to make bird go up")
        print("- Move your hand further to make bird go down")
        print("- Press 'R' to restart when game over")
        print("- Press 'Q' to quit")

        try:
            while True:
                # Get frames
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                # Convert to numpy arrays
                frame = np.asanyarray(color_frame.get_data())

                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                # Get hand depth
                self.get_hand_depth(frame, depth_frame)

                if not self.game_over:
                    # Update game
                    self.update_bird_position()
                    self.update_pipes()

                # Draw everything
                combined_frame = self.draw_game(frame)

                # Display
                cv2.imshow('Depth Flappy Bird', combined_frame)

                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r') and self.game_over:
                    self.reset_game()
                elif key == ord('r') and not self.game_over:
                    # Allow restart even during gameplay
                    self.reset_game()

        except KeyboardInterrupt:
            print("\nGame interrupted by user")
        except Exception as e:
            print(f"Error during game loop: {e}")
        finally:
            # Cleanup
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        game = DepthFlappyBird()
        game.run()
    except Exception as e:
        print(f"Failed to start game: {e}")
        print("\nRequired dependencies:")
        print("pip install opencv-python pyrealsense2 mediapipe numpy")
        print("\nMake sure your Intel RealSense camera is connected!")
