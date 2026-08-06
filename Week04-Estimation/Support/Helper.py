import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from matplotlib.patches import Ellipse, Arc
import time


def generate_measurements(steps=20, a=1, b=1, u=20, c=1, true_move_noise=2, true_mes_noise=10):

	#Our true state
	true_state = np.zeros((steps,1))
	#Our measurements
	measurements = np.zeros((steps,1))

	for i in range(1,steps):
		true_state[i] = a*true_state[i-1]+b*u+np.random.randn(1,1)*true_move_noise
		measurements[i] = c*true_state[i]+np.random.randn(1,1)*true_mes_noise

	return true_state, measurements

def lissajous_reference(
    time,
    amplitude_x=5.0,
    amplitude_y=3.5,
    frequency_x=0.35,
    frequency_y=0.70,
    phase=np.pi / 2.0,
):
    x = amplitude_x * np.sin(frequency_x * time + phase)
    y = amplitude_y * np.sin(frequency_y * time)

    dx = amplitude_x * frequency_x * np.cos(
        frequency_x * time + phase
    )
    dy = amplitude_y * frequency_y * np.cos(
        frequency_y * time
    )
    return x, y, dx, dy


def controls_from_lissajous(time):
    _, _, dx, dy = lissajous_reference(time)
    speed = np.hypot(dx, dy)
    heading = np.arctan2(dy, dx)
    return speed, heading


def configure_live_plot(reference_x, reference_y, title,
                        show_measurements=False):
    """Create the live trajectory plot used by both examples."""
    plt.close('all')
    fig, axis = plt.subplots(figsize=(10, 7))
    display_handle = display(fig, display_id=True)

    axis.set_title(title)
    axis.set_xlabel("x position [m]")
    axis.set_ylabel("y position [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.3)
    axis.set_xlim(np.min(reference_x) - 1.0, np.max(reference_x) + 1.0)
    axis.set_ylim(np.min(reference_y) - 1.0, np.max(reference_y) + 1.0)

    axis.plot(
        reference_x,
        reference_y,
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
        label="Reference",
    )

    measurement_scatter = None
    if show_measurements:
        measurement_scatter = axis.scatter(
            [], [], s=10, alpha=0.35, label="Noisy position measurements"
        )

    true_line, = axis.plot([], [], linewidth=2.0, label="True trajectory")
    estimate_line, = axis.plot([], [], linewidth=1.8, label="EKF estimate")
    estimate_point, = axis.plot(
        [], [], marker="o", markersize=4, linestyle="None",
        label="Current estimate"
    )
    heading_line, = axis.plot([], [], linewidth=1.3)

    axis.legend()

    artists = {
        "covariance_ellipse": None,
        "heading_arc": None,
    }

    return (
        fig,
        axis,
        display_handle,
        measurement_scatter,
        true_line,
        estimate_line,
        estimate_point,
        heading_line,
        artists,
    )


def update_pose_artists(axis, heading_line, artists, state, covariance,
                        robot_length=0.45, number_of_sigmas=2.0):
    """Update the robot heading and the current XY covariance ellipse."""
    x, y, heading, _ = state

    if artists["covariance_ellipse"] is not None:
        artists["covariance_ellipse"].remove()
    if artists["heading_arc"] is not None:
        artists["heading_arc"].remove()

    covariance_xy = covariance[:2, :2]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_xy)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]

    width = 2.0 * number_of_sigmas * np.sqrt(eigenvalues[0])
    height = 2.0 * number_of_sigmas * np.sqrt(eigenvalues[1])
    major_axis = eigenvectors[:, 0]
    ellipse_angle = np.degrees(
        np.arctan2(major_axis[1], major_axis[0])
    )

    artists["covariance_ellipse"] = Ellipse(
        xy=(x, y),
        width=width,
        height=height,
        angle=ellipse_angle,
        fill=False,
        linewidth=1.0,
        alpha=0.65,
    )
    axis.add_patch(artists["covariance_ellipse"])

    heading_line.set_data(
        [x, x + robot_length * np.cos(heading)],
        [y, y + robot_length * np.sin(heading)],
    )

    sigma_heading = np.sqrt(max(covariance[2, 2], 0.0))
    half_angle = min(number_of_sigmas * sigma_heading, np.pi)
    artists["heading_arc"] = Arc(
        (x, y),
        width=2.0 * robot_length,
        height=2.0 * robot_length,
        theta1=np.degrees(heading - half_angle),
        theta2=np.degrees(heading + half_angle),
        linewidth=0.8,
        alpha=0.6,
    )
    axis.add_patch(artists["heading_arc"])
