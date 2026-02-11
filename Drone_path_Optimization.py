import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)

# ---------- Problem setup ----------
start = np.array([0.0, 0.0])
goal = np.array([10.0, 8.0])
num_points = 30

obstacles = [
    (4.0, 3.0, 1.5),
    (6.5, 5.0, 1.2),
    (3.5, 6.0, 1.0)
]

# ALM parameters
mu = 10.0
mu_multiplier = 5
max_outer_iters = 20
tol_constraint = 1e-3

# Initialize path
path = np.linspace(start, goal, num_points)
var_idx = np.arange(1, num_points-1)
num_vars = len(var_idx)
num_obstacles = len(obstacles)
lambdas = np.zeros((num_vars, num_obstacles))

# ---------- Helper functions ----------
def cost_function(P):
    diffs = P[1:] - P[:-1]
    return np.sum(np.sum(diffs**2, axis=1))

def constraint_values(P):
    g = np.zeros((num_vars, num_obstacles))
    for j, i in enumerate(var_idx):
        p = P[i]
        for k, (cx, cy, R) in enumerate(obstacles):
            dist2 = (p[0]-cx)**2 + (p[1]-cy)**2
            g[j,k] = R**2 - dist2
    return g

def augmented_lagrangian_and_grad(flat_vars, lambdas, mu):
    P = path.copy()
    P[var_idx] = flat_vars.reshape((num_vars,2))

    cost = cost_function(P)
    grad = np.zeros_like(P)

    for i in range(1, num_points-1):
        grad[i] = 2*(2*P[i]-P[i-1]-P[i+1])

    g = constraint_values(P)
    AL = cost

    for j in range(num_vars):
        for k in range(num_obstacles):
            gjk = g[j,k]
            AL += lambdas[j,k]*gjk + 0.5*mu*max(0.0, gjk)**2
            i = var_idx[j]
            cx, cy, R = obstacles[k]
            grad_g = -2*(P[i]-np.array([cx,cy]))
            if gjk > 0:
                grad[i] += lambdas[j,k]*grad_g + mu*gjk*grad_g
            else:
                grad[i] += lambdas[j,k]*grad_g

    return AL, grad[var_idx].reshape(-1), g

# ---------- ALM Optimization ----------
history_cost = []
history_max_violation = []
flat_vars = path[var_idx].reshape(-1)

for outer in range(max_outer_iters):
    lr = 0.01
    for _ in range(600):
        AL_val, grad_flat, g = augmented_lagrangian_and_grad(flat_vars, lambdas, mu)
        flat_vars -= lr * grad_flat
        
        # --- MODIFICATION: Record cost and violation in EVERY inner step ---
        # Update path to calculate true cost/violation
        temp_path = path.copy()
        temp_path[var_idx] = flat_vars.reshape((num_vars,2))
        
        current_g = constraint_values(temp_path)
        current_max_violation = np.max(np.maximum(0.0, current_g))
        
        history_cost.append(cost_function(temp_path))
        history_max_violation.append(current_max_violation)
        # -------------------------------------------------------------------

    path[var_idx] = flat_vars.reshape((num_vars,2))
    g = constraint_values(path)
    max_violation = np.max(np.maximum(0.0, g))

    # Note: The lists are already populated from the inner loop,
    # so we don't need to append again here for the desired plot style.
    
    lambdas = np.maximum(0.0, lambdas + mu * g)

    print(f"AL iter {outer+1}: violation={max_violation:.3e}")

    if max_violation < tol_constraint:
        break

    mu *= mu_multiplier

# ---------- 2D Path ----------
fig1, ax1 = plt.subplots(figsize=(6,6))
init_path = np.linspace(start, goal, num_points)
ax1.plot(init_path[:,0], init_path[:,1], '--', label='initial path')
ax1.plot(path[:,0], path[:,1], 'o-', label='optimized path')

for cx,cy,R in obstacles:
    ax1.add_patch(plt.Circle((cx,cy), R, fill=False))

ax1.plot(*start, 's', label='start')
ax1.plot(*goal, '*', label='goal')
ax1.axis('equal')
ax1.legend()
ax1.set_title("2D Optimized Path")

# ---------- Convergence ----------
# --- MODIFICATION: Revert to simple line plots and update titles/labels ---
fig2, (ax2,ax3) = plt.subplots(1,2,figsize=(10,4))
ax2.plot(history_cost)
ax2.set_title("Cost evolution")
ax3.plot(history_max_violation)
ax3.set_title("Constraint violation")
# -------------------------------------------------------------------------

# ---------- ✅ 2D ANIMATION (LEGEND FIXED) ----------
fig3, ax4 = plt.subplots(figsize=(6,6))
ax4.set_xlim(0,12)
ax4.set_ylim(0,10)
ax4.set_aspect('equal')

ax4.plot(init_path[:,0], init_path[:,1], '--', color='gray', label='initial path')

for cx,cy,R in obstacles:
    ax4.add_patch(plt.Circle((cx,cy), R, fill=False, color='red'))

ax4.plot(*start, 's', label='start')
ax4.plot(*goal, '*', label='goal')

drone_line, = ax4.plot([], [], 'o-', color='orange', label='drone')
ax4.legend()
ax4.set_title("2D Drone Animation")

def update_2d(i):
    drone_line.set_data(path[:i,0], path[:i,1])
    return drone_line,

ani2d = FuncAnimation(fig3, update_2d, frames=len(path), interval=200)

# ---------- ✅ STATIC 3D PATH ----------
z_path = np.linspace(0, 4, num_points)

fig4 = plt.figure(figsize=(7,6))
ax5 = fig4.add_subplot(111, projection='3d')

ax5.plot(path[:,0], path[:,1], z_path, 'o-', label='optimized path')
ax5.plot(start[0], start[1], 0, 's', label='start')
ax5.plot(goal[0], goal[1], 4, '*', label='goal')

u = np.linspace(0,2*np.pi,30)
v = np.linspace(0,np.pi,20)

for cx, cy, R in obstacles:
    X = cx + R*np.outer(np.cos(u), np.sin(v))
    Y = cy + R*np.outer(np.sin(u), np.sin(v))
    Z = np.outer(np.ones_like(u), np.cos(v))*1.5
    ax5.plot_surface(X,Y,Z,alpha=0.25)

ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.set_zlabel('Z')
ax5.set_title("Static 3D Drone Path")
ax5.legend()

# ---------- ✅ 3D DRONE ANIMATION ----------
fig5 = plt.figure(figsize=(7,6))
ax6 = fig5.add_subplot(111, projection='3d')

ax6.plot(path[:,0], path[:,1], z_path, '--', alpha=0.4, label='optimized path')

for cx, cy, R in obstacles:
    X = cx + R*np.outer(np.cos(u), np.sin(v))
    Y = cy + R*np.outer(np.sin(u), np.sin(v))
    Z = np.outer(np.ones_like(u), np.cos(v))*1.5
    ax6.plot_surface(X,Y,Z,alpha=0.15)

drone, = ax6.plot([], [], [], 'o-', color='orange', label='drone')

ax6.set_xlabel('X')
ax6.set_ylabel('Y')
ax6.set_zlabel('Z')
ax6.set_title("3D Drone Animation")
ax6.legend()
ax6.view_init(25,120)

def update_3d(i):
    drone.set_data(path[:i,0], path[:i,1])
    drone.set_3d_properties(z_path[:i])
    return drone,

ani3d = FuncAnimation(fig5, update_3d, frames=len(path), interval=200)

# ---------- Keep figures alive ----------
plt.show(block=True)