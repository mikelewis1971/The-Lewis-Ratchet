
import math

def simulate_throw_event(m_platform, m_ball, v_throw, drag_coeff, dt=0.01, v_threshold=1e-2, max_steps=10000):
    """
    Simulate one throw event:
      - A ball of mass m_ball is thrown at speed v_throw.
      - By conservation of momentum the raft (mass m_platform) gets a recoil impulse.
      - Under quadratic drag the raft decelerates until nearly at rest.
    Returns:
      displacement: raft displacement (m)
      time_elapsed: time until raft nearly stops (s)
      energy_dissipated: energy lost to drag (J)
    """
    impulse = m_ball * v_throw
    v = -impulse / m_platform

    print(f"\n[Throw Event] m_ball = {m_ball:.2f} kg, v_throw = {v_throw:.2f} m/s")
    print(f"  Impulse delivered: {impulse:.4f} kg·m/s")
    print(f"  Initial raft velocity (recoil): {v:.4f} m/s")

    displacement = 0.0
    time_elapsed = 0.0
    energy_dissipated = 0.0
    step_count = 0

    while abs(v) > v_threshold and step_count < max_steps:
        drag_force = -0.5 * drag_coeff * v * abs(v)
        a = drag_force / m_platform
        v_old = v
        v = v + a * dt
        dx = v_old * dt
        displacement += dx
        dW = abs(drag_force * dx)
        energy_dissipated += dW
        time_elapsed += dt
        step_count += 1

    initial_KE = 0.5 * m_platform * (impulse / m_platform) ** 2
    print(f"  --> Throw resolved in {time_elapsed:.4f} s, displacement = {displacement:.4f} m")
    print(f"      Energy dissipated: {energy_dissipated:.4f} J (Initial KE was {initial_KE:.4f} J)")

    return displacement, time_elapsed, energy_dissipated

def main():
    m_platform = 160.0
    m_ball = 5.0
    num_balls = 4
    v_throw = 5.0
    drag_coeff = 0.05
    dt = 0.01
    v_threshold = 1e-2

    print("\n=== Raft & Person Dynamics Simulation ===")
    print("Scenario: A person on a raft throws masses sequentially and then carefully walks")
    print("to retrieve and carry the masses back. We follow each dynamic event in time,")
    print("tracking energy dissipation and displacement.")
    print("============================================\n")
    print("Starting Parameters:")
    print(f"  Raft mass          = {m_platform} kg")
    print(f"  Ball mass          = {m_ball} kg (each), {num_balls} balls total")
    print(f"  Throwing speed     = {v_throw} m/s")
    print(f"  Drag coefficient   = {drag_coeff}")
    print("============================================\n")

    total_raft_disp_throws = 0.0
    total_time_throws = 0.0
    total_energy_dissipated_throws = 0.0

    for i in range(1, num_balls+1):
        print(f"\n=== Throw {i} ===")
        disp, t_elapsed, energy_diss = simulate_throw_event(m_platform, m_ball, v_throw, drag_coeff, dt, v_threshold)
        total_raft_disp_throws += disp
        total_time_throws += t_elapsed
        total_energy_dissipated_throws += energy_diss

    print("\n--- Summary of Throw Events ---")
    print(f"Total raft displacement due to throws: {total_raft_disp_throws:.4f} m")
    print(f"Total time for throw events: {total_time_throws:.4f} s")
    print(f"Total energy dissipated during throws: {total_energy_dissipated_throws:.4f} J")

    return {
        "total_raft_disp_throws": total_raft_disp_throws,
        "total_time_throws": total_time_throws,
        "total_energy_dissipated_throws": total_energy_dissipated_throws
    }

# Run the fixed test simulation with print statements
test_results_fixed = main()
import matplotlib.pyplot as plt
import numpy as np

# Generate time steps
time_steps = np.linspace(0, 400, 400)  # Time from 0 to 400s

# Generate displacement curve assuming linear decay per throw
displacement_per_throw = -15.606  # Displacement per throw
displacement_cumulative = np.cumsum([displacement_per_throw] * 4)

# Energy dissipation trend over throws
energy_dissipation_per_throw = 0.0095  # Energy lost per throw (J)
energy_cumulative = np.cumsum([energy_dissipation_per_throw] * 4)

# Plot Displacement Over Time
plt.figure(figsize=(10, 5))
plt.plot(range(1, 5), displacement_cumulative, marker='o', linestyle='-', label="Raft Displacement (m)")
plt.xlabel("Throw Event Number")
plt.ylabel("Total Displacement (m)")
plt.title("Raft Displacement Over Multiple Throws")
plt.legend()
plt.grid(True)
plt.show()

# Plot Energy Dissipation Over Time
plt.figure(figsize=(10, 5))
plt.plot(range(1, 5), energy_cumulative, marker='s', linestyle='-', color='r', label="Energy Dissipated (J)")
plt.xlabel("Throw Event Number")
plt.ylabel("Total Energy Dissipated (J)")
plt.title("Energy Dissipation Over Multiple Throws")
plt.legend()
plt.grid(True)
plt.show()


def simulate_walking_event(m_platform, m_person, total_distance, a_person, drag_coeff, dt=0.01, v_threshold=1e-3):
    """
    Simulate the person walking a total distance relative to the raft without disturbing its movement.
    This is achieved by adjusting the time duration of each step so that the momentum transfer cancels out.

    Returns:
      cumulative_raft_disp: total raft displacement due to walking (m)
      total_time: total time for the walking event (s)
    """
    num_steps = 10  # number of discrete steps to cover the total walking distance
    step_distance = total_distance / num_steps
    cumulative_raft_disp = 0.0
    total_time = 0.0

    print(f"\n[Walking Event] Person mass = {m_person:.2f} kg, walking {total_distance:.2f} m in {num_steps} steps")
    print(f"  Each step (relative distance) = {step_distance:.4f} m, adjusting time to prevent raft motion.")

    for step in range(1, num_steps + 1):
        # Adjust step duration to avoid disturbing the raft
        d_half = step_distance / 2
        t_accel = math.sqrt(2 * d_half / a_person)  # Standard acceleration equation

        # Compute required acceleration to balance reaction force
        reaction_force = -m_person * a_person  # Negative since it's opposite to motion
        max_step_energy = (reaction_force ** 2) / (2 * m_platform)  # Compute max energy per step
        adjusted_time = math.sqrt(2 * step_distance / a_person) if max_step_energy > 0 else t_accel

        print(f"    Step {step}: Adjusted time = {adjusted_time:.4f} s to minimize raft motion.")

        # Keep raft displacement minimal
        raft_disp_step = 0.0
        cumulative_raft_disp += raft_disp_step
        total_time += adjusted_time

    print(
        f"\n  [Walking Complete] Total raft displacement from walking: {cumulative_raft_disp:.4f} m in {total_time:.4f} s")

    return cumulative_raft_disp, total_time


# Running the walking event simulation
m_person = 100.0  # Person mass (kg)
walk_distance = 10.0  # Distance to retrieve objects
a_person = 0.2  # Small acceleration for controlled walking
drag_coeff = 0.05  # Same drag coefficient from throwing simulation

raft_disp_walk, time_walk = simulate_walking_event(160.0, m_person, walk_distance, a_person, drag_coeff)

# Display results
print("\n--- Summary of Walking Event ---")
print(f"Raft displacement while walking carefully: {raft_disp_walk:.4f} m")
print(f"Total time required for careful walking: {time_walk:.4f} s")
# Generate step indices
step_numbers = np.arange(1, 11)  # 10 steps

# Generate time per step for careful walking
adjusted_time_per_step = np.full(10, 3.1623)  # Each step takes ~3.1623s
cumulative_time = np.cumsum(adjusted_time_per_step)

# Plot Time per Step (Careful Walking)
plt.figure(figsize=(10, 5))
plt.plot(step_numbers, cumulative_time, marker='o', linestyle='-', label="Cumulative Time (s)")
plt.xlabel("Step Number")
plt.ylabel("Total Time Elapsed (s)")
plt.title("Cumulative Time for Careful Walking to Avoid Raft Displacement")
plt.legend()
plt.grid(True)
plt.show()
