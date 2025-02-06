import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import numpy as np
import sys

class RealTimePlot(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Ultra-Fast Real-Time Plot")
        self.setGeometry(100, 100, 800, 600)

        # create central widget and a tab widget
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # Additional Plot Tab
        self.line_plot_widget = self.add_plot_tab("Line Plot")
        self.scatter_plot_widget = self.add_plot_tab("Scatter Plot")
        self.additional_plot_widget = self.add_plot_tab("Additional Plot")



        self.curve = self.line_plot_widget.plot([], [], pen='r')
        self.scatter = self.scatter_plot_widget.plot([], [], pen=None, symbol='o')
        self.additional_curve = self.additional_plot_widget.plot([], [], pen='g')

        # Extra plots in a single tab
        self.multi_plot_widget = self.add_multi_plot_tab("Multi Plot Tab", 5)
        self.multi_curves = [self.multi_plot_widget[i].plot([], [], pen=color) for i, color in enumerate(['b', 'm'])]

        # Fastest real-time update: QTimer (faster than Matplotlib's animation)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1)  # Update every 1 ms

        self.time = 0

    def update_plot(self):

        self.time += 0.01
        self.data_size = 1000  # More points, still fast
        self.x_data = np.linspace(0, 2 * np.pi, self.data_size)
        self.y_data = np.sin(self.x_data + self.time)

        """Shift data and update plot in real-time"""

        self.curve.setData(self.x_data, self.y_data)  # Fast update
        self.scatter.setData(self.x_data[::10], self.y_data[::10])  # Update scatter plot with fewer points
        self.additional_curve.setData(self.x_data, np.cos(self.y_data))  # Update additional plot with cosine transformation

        for i, curve in enumerate(self.multi_curves):
            curve.setData(self.x_data, np.sin(self.y_data + i))  # Different phase shifts

    def add_plot_tab(self, title):
        """Adds a new empty plot tab with the given title"""
        new_plot_widget = pg.PlotWidget()
        self.tabs.addTab(new_plot_widget, title)
        return new_plot_widget

    def add_multi_plot_tab(self, title, num_plots):
        """Adds a new tab with multiple plots stacked vertically"""
        new_tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        plot_widgets = []

        for _ in range(num_plots):
            plot_widget = pg.PlotWidget()
            layout.addWidget(plot_widget)
            plot_widgets.append(plot_widget)

        new_tab.setLayout(layout)
        self.tabs.addTab(new_tab, title)
        return plot_widgets


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = RealTimePlot()
    window.show()
    sys.exit(app.exec_())
