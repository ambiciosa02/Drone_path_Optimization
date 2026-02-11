# Drone_path_Optimization

Drone Path Optimization & Obstacle Avoidance
This project demonstrates a smart path-planning algorithm that teaches a virtual drone how to find the most efficient route from point A to point B while navigating around obstacles.

🌟 Overview
The script solves a common robotics problem: finding the shortest path that doesn't crash. It uses a mathematical technique called the "Augmented Lagrangian Method" to nudge the path away from obstacles until it finds a safe, smooth curve.

Key Highlights:
Smart Smoothing: The algorithm doesn't just find a way around; it tries to keep the path as straight and efficient as possible to save "battery" (energy).

Obstacle Awareness: The drone detects circular boundaries and adjusts its trajectory in real-time during the optimization process.

3D Visualization: While the logic works in 2D, the project projects the path into a 3D environment to simulate a real-world flight take-off and landing.

Live Animations: Includes built-in animations to watch the drone fly the final optimized route.

📊 Visualizing the Results
The project generates several views to show how the "thinking" process works:

The Optimization Journey: Charts showing how the drone slowly stops "breaking the rules" (hitting obstacles) as the algorithm runs.

2D Top-Down View: A clear map of the initial straight-line path versus the new, safe, curved path.

3D Flight Path: A perspective view showing the drone navigating through spherical danger zones.


https://github.com/user-attachments/assets/5b2944e0-55ee-4cd9-9d99-d25b569ba08f


🛠️ Built With
Python – The core logic.

NumPy – For the heavy lifting and calculations.

Matplotlib – For the 2D/3D plots and animations.

🚀 How to Run
Ensure you have Python installed, then run:

Bash
python Drone_path_Optimization.py


