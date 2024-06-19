import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tabulate import tabulate
import plotly.graph_objects as go
#2
def h(theta0, theta1, x):
    return theta0 + (theta1 * x)
#3
def cost_function(theta0, theta1, x, y):
    m = len(x)
    hyp = h(theta0, theta1, x)
    return (1 / (2 * m)) * np.sum(np.square(hyp - y))
#4 in the word document.
#5
def gradient_descent(x, y, tol=1e-16):
    iterations = 100
    learning_rate = 0.5
    m = len(y)
    theta0 = 0
    theta1 = 0
    costs = []
    thetas = [[0,0]] # Define thetas to store theta values
    table_data = [] # Define table_data to store iteration information
    for i in range(iterations):
        h_theta = h(theta0, theta1, x)
        cost = cost_function(theta0, theta1, x, y)

        # Append iteration information to table_data
        table_data.append([i, theta0, theta1, cost])

        gradient0 = (1/m) * np.sum(h_theta - y)
        gradient1 = (1/m) * np.sum((h_theta - y) * x)
        theta0 -= (learning_rate * gradient0)
        theta1 -= (learning_rate * gradient1)
        costs.append(cost)
        thetas.append([theta0, theta1]) # Append theta values to thetas
        if i > 0 and np.abs(costs[-1] - costs[-2]) < tol:
            print("Converged!")
            break
    # Print the table after the gradient descent process completes
    headers = ["6B: Iteration", "Theta 0", "Theta 1", "Cost J(Theta0,Theta1)"]
    print(tabulate(table_data, headers=headers, tablefmt="github"))

    return theta0, theta1, costs, thetas

def plot_cost_function(costs):
    plt.plot(costs, marker='.')
    plt.title("6B: Cost Function J(?0, ?1) VS Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()

def plot_line(x, y, theta0, theta1):
    plt.scatter(x, y, label='Data Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('7 : Scatter Plot Of X and Y')
    plt.plot(x, h(theta0, theta1, x), color='red', label='Plot of Line')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Define x and y with the provided data points
    #1
    x = np.random.rand(10,1)
    y = 2*x+np.random.randn(10,1)

    # Gradient Descent
    theta0, theta1, costs, thetas = gradient_descent(x, y)

    # Plot the cost function
    plot_cost_function(costs)
     # Extract data for plotting
    theta_history = np.array(thetas)
    # Plot scatter plot with line
    plot_line(x, y, theta0, theta1)
    # Print final theta values
    print("7: Final Theta values: theta0 =", theta_history[-2,0], ", theta1 =", theta_history[-2,1])



    # Plot 3D graph
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    J_history = np.array(costs)

    # Append the final cost value to J_history
    J_history = np.append(J_history, J_history[-1])

    # Plot the trajectory of optimization algorithm
    ax.plot(theta_history[:, 0], theta_history[:, 1], J_history, 'b-')

    # Mark the initial point
    initial_x, initial_y, initial_cost = theta_history[0, 0], theta_history[0, 1], J_history[0]
    ax.scatter(initial_x, initial_y, initial_cost, color='red', label='Initial Point')
    ax.text(initial_x, initial_y, initial_cost + 0.1, f'Initial: (?0={initial_x:.2f}, ?1={initial_y:.2f}, Cost={initial_cost:.2f})', color='black')

    # Mark the final point
    final_x, final_y, final_cost = theta_history[-1, 0], theta_history[-1, 1], J_history[-1]
    ax.scatter(final_x, final_y, final_cost, color='green', label='Final Point')
    ax.text(final_x, final_y, final_cost + 0.1, f'Final: (?0={final_x:.2f}, ?1={final_y:.2f}, Cost={final_cost:.2f})', color='black')

    # Label axes and set title
    ax.set_xlabel('?0')
    ax.set_ylabel('?1')
    ax.set_zlabel('Cost J')
    ax.set_title('6A: Cost Function: ')

    # Show legend
    ax.legend()

    # Show the plot
    plt.show()
    theta0_range = np.linspace(np.min(theta_history[:,0]), np.max(theta_history[:,0]), 100)
    theta1_range = np.linspace(np.min(theta_history[:,1]), np.max(theta_history[:,1]), 100)
    theta0_grid, theta1_grid = np.meshgrid(theta0_range, theta1_range)

    # Calculate the cost for each combination of theta0 and theta1
    cost_grid = np.array([[cost_function(theta0, theta1, x, y) for theta0 in theta0_range] for theta1 in theta1_range])

    # Plot the surface using Plotly
    fig = go.Figure(data=[go.Surface(z=cost_grid, x=theta0_grid, y=theta1_grid)])
    fig.update_layout(title='Cost Function Surface',
                      scene=dict(xaxis_title='Theta 0', yaxis_title='Theta 1', zaxis_title='Cost'))

    # Add annotations for the optimization path
    fig.add_trace(go.Scatter3d(x=theta_history[:, 0], y=theta_history[:, 1], z=costs,
                               mode='markers+lines', marker=dict(size=5, color='blue'),
                               name='Path'))

    # Add trace for the initial point
    fig.add_trace(go.Scatter3d(x=[theta_history[0, 0]], y=[theta_history[0, 1]], z=[costs[0]],
                               mode='markers', marker=dict(size=5, color='red'),
                               name='Initial Point',
                               text='Initial Point',
                               hoverinfo='text'))

    # Add trace for the final point
    fig.add_trace(go.Scatter3d(x=[theta0], y=[theta1], z=[costs[-1]],
                               mode='markers', marker=dict(size=5, color='green'),
                               name='Final Point',
                               text='Final Point',
                               hoverinfo='text'))

    fig.show()
