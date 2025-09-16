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
        self.hand_bbox = None  # Store bounding box for depth visualization

        # Pipe parameters
        self.pipe_width = 60
        self.pipe_gap = 150
        self.pipe_speed = 3
        self.pipes = []

        # Game state
        self.score = 0
        self.game_over = False
        self.game_started = False  # Add game started state
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

    def get_hand_depth(self, rgb_frame, depth_frame):
        """Detect hand and get depth at bounding box centroid"""
        # Create display frame for RGB visualization
        rgb_display = rgb_frame.copy()

        try:
            rgb_converted = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_converted)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    try:
                        # Draw hand landmarks on RGB frame
                        self.mp_draw.draw_landmarks(rgb_display, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

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

                        # Store bounding box for depth visualization
                        self.hand_bbox = (min_x, min_y, max_x, max_y)

                        # Calculate bounding box centroid
                        bbox_center_x = (min_x + max_x) // 2
                        bbox_center_y = (min_y + max_y) // 2

                        # Validate centroid coordinates
                        if not (0 <= bbox_center_x < self.width and 0 <= bbox_center_y < self.height):
                            continue

                        # Since RGB frame is flipped, flip x coordinate for depth lookup
                        depth_x = self.width - bbox_center_x - 1
                        depth_y = bbox_center_y

                        # Sample depth values in a region around the bounding box center
                        depth_values = []
                        sample_radius = 15  # Sample in 30x30 area around center

                        for dx in range(-sample_radius, sample_radius + 1, 3):
                            for dy in range(-sample_radius, sample_radius + 1, 3):
                                try:
                                    px = max(0, min(self.width - 1, depth_x + dx))
                                    py = max(0, min(self.height - 1, depth_y + dy))
                                    depth_val = depth_frame.get_distance(px, py) * 1000  # Convert to mm
                                    if 200 < depth_val < 1500:  # Valid depth range for hand detection
                                        depth_values.append(depth_val)
                                except Exception:
                                    continue

                        if len(depth_values) > 5:
                            # Use median depth for stability
                            depth_value = np.median(depth_values)

                            # Initialize smoothed_depth if first detection
                            if not hasattr(self, 'smoothed_depth') or self.smoothed_depth == 0:
                                self.smoothed_depth = depth_value

                            # Moderate smoothing to balance responsiveness and stability
                            self.smoothed_depth = (0.6 * depth_value + 0.4 * self.smoothed_depth)
                            self.hand_detected = True

                            # Draw bounding box on RGB frame
                            cv2.rectangle(rgb_display, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

                            # Draw bounding box center
                            cv2.circle(rgb_display, (bbox_center_x, bbox_center_y), 8, (255, 0, 0), -1)
                            cv2.circle(rgb_display, (bbox_center_x, bbox_center_y), 12, (255, 255, 255), 2)

                            # Display depth info on RGB frame
                            text_y = max(25, min_y)
                            cv2.putText(rgb_display, f"Raw Depth: {int(depth_value)}mm",
                                      (min_x, text_y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(rgb_display, f"Smooth: {int(self.smoothed_depth)}mm",
                                      (min_x, text_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                            cv2.putText(rgb_display, f"Samples: {len(depth_values)}",
                                      (min_x, text_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                            return rgb_display, True

                    except Exception as e:
                        print(f"Hand processing error: {e}")
                        continue

            # No hand detected
            self.hand_detected = False
            self.hand_bbox = None
            return rgb_display, False

        except Exception as e:
            print(f"Hand depth detection error: {e}")
            self.hand_detected = False
            self.hand_bbox = None
            return rgb_display, False

    def create_depth_visualization(self, depth_frame):
        """Create colorized depth map with hand bounding box"""
        try:
            # Create depth colormap
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Flip depth map to match RGB orientation
            depth_colormap = cv2.flip(depth_colormap, 1)

            # Draw bounding box on depth map if hand is detected
            if self.hand_bbox is not None:
                min_x, min_y, max_x, max_y = self.hand_bbox
                cv2.rectangle(depth_colormap, (min_x, min_y), (max_x, max_y), (255, 255, 255), 2)

                # Draw center point
                center_x = (min_x + max_x) // 2
                center_y = (min_y + max_y) // 2
                cv2.circle(depth_colormap, (center_x, center_y), 8, (0, 0, 255), -1)
                cv2.circle(depth_colormap, (center_x, center_y), 12, (255, 255, 255), 2)

                # Display depth value on depth map
                if hasattr(self, 'smoothed_depth'):
                    cv2.putText(depth_colormap, f"Depth: {int(self.smoothed_depth)}mm",
                              (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Add status text
            status_color = (255, 255, 255) if self.hand_detected else (0, 0, 255)
            status_text = "Hand Detected" if self.hand_detected else "No Hand Detected"
            cv2.putText(depth_colormap, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            return depth_colormap

        except Exception as e:
            print(f"Depth visualization error: {e}")
            # Return a default depth map
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

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

    def draw_game(self):
        """Draw game window only"""
        try:
            # Clear background
            game_surface = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)

            # Show start screen if game hasn't started
            if not self.game_started:
                # Start screen overlay
                overlay = game_surface.copy()
                cv2.rectangle(overlay, (100, 150), (540, 330), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.8, game_surface, 0.2, 0, game_surface)

                cv2.putText(game_surface, "DEPTH FLAPPY BIRD", (150, 190),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                cv2.putText(game_surface, "Move your hand closer/further", (140, 230),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(game_surface, "to control the bird's height", (150, 250),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(game_surface, "Press 'S' to Start", (220, 290),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(game_surface, "Press 'Q' to Quit", (230, 315),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # Show hand detection status on start screen
                status_color = (0, 255, 0) if self.hand_detected else (0, 0, 255)
                status_text = "Hand Ready!" if self.hand_detected else "Show Hand to Camera"
                cv2.putText(game_surface, status_text, (200, 360),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                return game_surface

            # Draw pipes only if game has started
            for pipe in self.pipes:
                try:
                    x1 = max(0, int(pipe['x']))
                    x2 = min(self.width, int(pipe['x'] + self.pipe_width))
                    top = max(0, int(pipe['top']))
                    bottom = min(self.height, int(pipe['bottom']))

                    if x2 > x1:
                        # Top pipe
                        if top > 0:
                            cv2.rectangle(game_surface, (x1, 0), (x2, top), self.pipe_color, -1)
                            cv2.rectangle(game_surface, (x1, 0), (x2, top), (0, 100, 0), 3)

                        # Bottom pipe
                        if bottom < self.height:
                            cv2.rectangle(game_surface, (x1, bottom), (x2, self.height), self.pipe_color, -1)
                            cv2.rectangle(game_surface, (x1, bottom), (x2, self.height), (0, 100, 0), 3)

                except Exception as e:
                    print(f"Pipe drawing error: {e}")
                    continue

            # Draw bird
            try:
                bird_x = int(self.bird_x)
                bird_y = int(self.bird_y)
                bird_radius = int(self.bird_radius)

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

            # Draw UI
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
                cv2.putText(game_surface, status_text, (10, self.height - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                # Control instructions
                cv2.putText(game_surface, "Move hand closer/further to control bird",
                           (10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Depth info if available
                if hasattr(self, 'smoothed_depth') and self.hand_detected:
                    cv2.putText(game_surface, f"Depth: {int(self.smoothed_depth)}mm",
                               (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
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
                    cv2.putText(game_surface, "Press 'R' to restart or 'S' for new game", (140, 280),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                except Exception as e:
                    print(f"Game over drawing error: {e}")

            return game_surface

        except Exception as e:
            print(f"Draw game error: {e}")
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
        self.game_started = False  # Reset to start screen
        self.spawn_pipe()

    def create_combined_feedback_window(self, rgb_display, depth_display):
        """Combine RGB and depth displays side by side"""
        try:
            # Ensure both images are the same height
            if rgb_display.shape[0] != depth_display.shape[0]:
                target_height = min(rgb_display.shape[0], depth_display.shape[0])
                rgb_display = cv2.resize(rgb_display, (self.width, target_height))
                depth_display = cv2.resize(depth_display, (self.width, target_height))

            # Combine horizontally
            combined = np.hstack([rgb_display, depth_display])

            # Add labels
            cv2.putText(combined, "RGB + Hand Detection", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, "Depth Map + Hand Position", (self.width + 10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            return combined

        except Exception as e:
            print(f"Combined window error: {e}")
            # Return a fallback combined window
            fallback = np.zeros((self.height, self.width * 2, 3), dtype=np.uint8)
            cv2.putText(fallback, "Error combining windows", (self.width // 2, self.height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return fallback

    def run(self):
        """Main game loop with two windows - game above, combined feedback below"""
        print("Depth-Controlled Flappy Bird - Two Window Layout")
        print("Window Layout:")
        print("1. Game Window - The main Flappy Bird game")
        print("2. Feedback Window - RGB frame (left) + Depth map (right)")
        print("\nControls:")
        print("- Press 'S' to start the game")
        print("- Move your hand closer to camera to make bird go up")
        print("- Move your hand further to make bird go down")
        print("- Press 'R' to restart")
        print("- Press 'Q' to quit")

        try:
            while True:
                # Get frames
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                # Convert color frame to numpy array
                rgb_frame = np.asanyarray(color_frame.get_data())

                # Flip RGB frame horizontally for mirror effect
                rgb_frame = cv2.flip(rgb_frame, 1)

                # Process hand detection and get RGB visualization
                rgb_display, hand_found = self.get_hand_depth(rgb_frame, depth_frame)

                # Create depth visualization
                depth_display = self.create_depth_visualization(depth_frame)

                # Create combined feedback window
                feedback_window = self.create_combined_feedback_window(rgb_display, depth_display)

                # Only update game if it has started and not over
                if self.game_started and not self.game_over:
                    # Update game
                    self.update_bird_position()
                    self.update_pipes()

                # Draw game window
                game_display = self.draw_game()

                # Display windows
                cv2.imshow('Depth Flappy Bird - Game', game_display)
                cv2.imshow('Camera Feedback - RGB & Depth', feedback_window)

                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    if not self.game_started or self.game_over:
                        self.game_started = True
                        if self.game_over:
                            self.reset_game()
                            self.game_started = True
                elif key == ord('r'):
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
