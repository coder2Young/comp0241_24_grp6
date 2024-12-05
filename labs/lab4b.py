import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from sklearn.neighbors import NearestNeighbors

#COMP0241 Lab 4b: ICP with SVD. (optional, unmarked)
#This code is a simple implementation of the Iterative Closest Point (ICP) algorithm using Singular Value Decomposition (SVD), not assuming know data association.
#The first option of the code generates a random set of points and a set of destination points, and iteratively aligns the source points to the destination points using ICP.
#The other options are pre-defined sets of points for testing purposes.

#TODO: fill in find_best_transform_svd() at the bottom.
#TODO: try varying the distance_threshold and max_iterations to see how they affect the results.



class ICPVisualization:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        plt.subplots_adjust(bottom=0.2, right=0.8)
        self.ax.set_xlim(0, 500)
        self.ax.set_ylim(0, 300)

        self.dataset_selector = RadioButtons(plt.axes([0.1, 0.05, 0.3, 0.1]), 
                                             ('Random', 'Set 1', 'Set 2', 'Set 3'))
        self.start_button = Button(plt.axes([0.5, 0.05, 0.1, 0.05]), 'Start ICP')
        self.start_button.on_clicked(self.start_icp)

        self.icp_state = None
        self.timer = None
        self.X = None
        self.destinations = None
        self.previous_indices = None

    def start_icp(self, event):
        dataset = self.dataset_selector.value_selected
        if dataset == 'Random':
            self.X = np.array([(100 + np.sin(i/10*np.pi)*50, 100 + np.cos(i/10*np.pi)*50) for i in range(10)], dtype=np.int32)
            self.destinations = np.random.normal(150, (125, 50), (50, 2)).astype(np.int32)
        elif dataset == 'Set 1':
            self.X = np.array([[7, 198], [37, 155], [76, 138], [126, 123], [177, 112], [229, 114], [277, 110], [363, 136], [399, 151], [433, 196], [439, 222]], dtype=np.int32)
            self.destinations = np.array([[6, 125], [67, 130], [97, 92], [150, 99], [188, 68], [208, 89], [285, 78], [305, 105], [369, 100], [390, 137], [432, 150]], dtype=np.int32)
        elif dataset == 'Set 2':
            self.X = np.array([[7, 198], [37, 155], [76, 138], [126, 123], [177, 112], [229, 114], [277, 110], [363, 136], [399, 151], [433, 196], [439, 222]], dtype=np.int32)
            self.destinations = np.array([[38, 232], [20, 184], [28, 127], [53, 69], [92, 33], [154, 12], [200, 12], [270, 13], [318, 26], [357, 46], [386, 76]], dtype=np.int32)
        else:  # Set 3
            self.X = np.array([[7, 198], [37, 155], [76, 138], [126, 123], [177, 112], [229, 114], [277, 110], [363, 136], [399, 151], [433, 196], [439, 222]], dtype=np.int32)
            self.destinations = np.array([[5, 217], [22, 168], [68, 141], [111, 125], [158, 115], [202, 111], [235, 106], [283, 111], [366, 136], [395, 151], [432, 188]], dtype=np.int32)

        self.previous_indices = None
        self.icp_state = self.icp_generator(self.X, self.destinations)
        if self.timer is not None:
            self.timer.stop()
        self.timer = self.fig.canvas.new_timer(interval=1000)
        self.timer.add_callback(self.update_plot)
        self.timer.start()

    def update_plot(self):
        try:
            X, X_bar, iteration, dist, indices = next(self.icp_state)
            self.ax.clear()
            self.ax.set_xlim(0, 500)
            self.ax.set_ylim(0, 300)
            
            # Plot points
            self.ax.scatter(self.destinations[:, 0], self.destinations[:, 1], c='white', s=100, edgecolors='black', label='Destination Points')
            self.ax.scatter(X[:, 0], X[:, 1], c='green', s=50, edgecolors='black', label='Source Points')
            self.ax.scatter(X_bar[:, 0], X_bar[:, 1], c='red', s=50, edgecolors='black', label='Matched Points')

            # Plot lines
            for i in range(len(X)):
                color = 'yellow' if self.previous_indices is not None and indices[i] != self.previous_indices[i] else 'red'
                self.ax.plot([X[i, 0], X_bar[i, 0]], [X[i, 1], X_bar[i, 1]], c=color, linewidth=2)
            
            for i in range(1, len(X)):
                self.ax.plot([X[i-1, 0], X[i, 0]], [X[i-1, 1], X[i, 1]], color='orange', linewidth=1)
                self.ax.plot([X_bar[i-1, 0], X_bar[i, 0]], [X_bar[i-1, 1], X_bar[i, 1]], color='lightgreen', linewidth=1)

            self.ax.set_title(f'Iteration: {iteration}, Distance: {dist:.2f}')
            self.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            # Add text annotations for changed correspondences
            if self.previous_indices is not None:
                changed = np.where(indices != self.previous_indices)[0]
                for i in changed:
                    self.ax.annotate(f'Changed: {i}', (X[i, 0], X[i, 1]), xytext=(5, 5), 
                                     textcoords='offset points', color='purple', fontweight='bold')

            self.previous_indices = indices
            self.fig.canvas.draw()

        except StopIteration:
            self.timer.stop()

    def icp_generator(self, X, destinations):
        last_dist = np.inf
        last_good = X.copy()

        for iteration in range(100):  # max_iterations = 100
            indices, distances = self.flann_knn(destinations, X)
            dist = np.sum(distances)

            X_bar = destinations[indices]
            yield X, X_bar, iteration + 1, dist, indices
            
            # if distnace remain the same, such as R=I, T=0, the loop will stop
            if last_dist - dist < 0.001:  # distance_threshold = 0.001 
                break

            if last_dist <= dist: 
                X = last_good
                break

            last_dist = dist
            last_good = X.copy()

            X = self.find_best_transform_svd(X, X_bar)

    def flann_knn(self, destinations, query):
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(destinations)
        distances, indices = nbrs.kneighbors(query)
        return indices.flatten(), distances.flatten()

    def find_best_transform_svd(self, m, d):
        m = m.astype(np.float32)
        d = d.astype(np.float32)

        m_bar = np.mean(m, axis=0)
        d_bar = np.mean(d, axis=0)
        mc = m - m_bar
        dc = d - d_bar

        ## TODO: fill in the code here to find the best transformation using SVD
        R=np.eye(2) #comment this line out when you fill in the code
        T=np.zeros(2) #comment this line out when you fill in the code

        m_new = (R @ m.T).T + T
        return m_new.astype(np.int32)

if __name__ == "__main__":
    viz = ICPVisualization()
    plt.show()