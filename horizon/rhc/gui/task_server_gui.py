import sys
import re
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QSlider, QScrollArea, QHBoxLayout, QPushButton, QLineEdit, QFileDialog
from PyQt5.QtGui import QColor, QPalette, QFont
from PyQt5.QtCore import Qt
import rospy
from rosbot_param_server.srv import GetParameterInfo, SetString
import yaml
from horizon.utils.logger import Logger

class DynamicSliderWindow(QMainWindow):
    def __init__(self, params):
        super().__init__()


        self.logger = Logger(self)
        self.logger.log(f"Initializing parameters GUI...")
        
        self.setWindowTitle("Dynamic Sliders with On-Demand Updates")
        self.setGeometry(100, 100, 600, 800)

        # Create a central widget and a vertical layout for the entire window
        central_widget = QWidget(self)
        main_layout = QVBoxLayout(central_widget)

        # Create a scroll area for sliders
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)

        # Create a container widget and layout for sliders
        container_widget = QWidget()
        layout = QVBoxLayout(container_widget)

        # Store parameter labels for live updates
        self.param_labels = {}
        self.param_sliders = {}
        self.current_values = {}

        # Add sliders based on ROS parameters
        for param_name, value, descriptor in params:
            try:

                self.logger.log(f"adding parameter: '{param_name}' ...")
                min_val, max_val = self.parse_descriptor(descriptor)
                initial_val = float(value)

                # Dynamically determine the scaling factor based on the range
                decimal_places = max(0, len(str(initial_val).split('.')[1])) + 1 # Calculate decimal places in range

                # print('param_name: ', param_name)
                # print('initial_val: ', initial_val)
                # print('decimal_places: ', decimal_places)
                # print(f'min-max: {min_val} -- {max_val}')


                # Store the initial value for reset
                self.current_values[param_name] = initial_val

                # Create a vertical layout for name, slider, and value
                v_layout = QVBoxLayout()

                # Create label for the parameter name
                name_label = QLabel(param_name)
                name_label.setAlignment(Qt.AlignCenter)

                h0_layout = QHBoxLayout()
                h0_layout.addWidget(name_label)

                # Create a label to display the current value above the slider
                value_label_above = QLabel(f"{initial_val:.{decimal_places}f}")
                value_label_above.setAlignment(Qt.AlignCenter)
                value_label_above.setFont(QFont("Arial", 8))  # Optional, adjust the font size
                h0_layout.addWidget(value_label_above)

                v_layout.addLayout(h0_layout)

                h1_layout = QHBoxLayout()


                # Define a scaling factor based on decimal places in the range
                scaling_factor = 10 ** decimal_places

                scaled_min = int(min_val * scaling_factor)
                scaled_max = int(max_val * scaling_factor)
                scaled_initial = int(initial_val * scaling_factor)

                slider = QSlider(Qt.Horizontal)
                slider.setMinimum(scaled_min)
                slider.setMaximum(scaled_max)

                slider.setValue(scaled_initial)
                slider.setTickPosition(QSlider.TicksBelow)
                slider.setTickInterval(max(1, (scaled_max - scaled_min) // decimal_places))

                # Store the slider for resetting
                self.param_sliders[param_name] = slider

                # Update the displayed value when the slider moves
                slider.valueChanged.connect(
                    lambda val, label=value_label_above, scale=scaling_factor, res=decimal_places: self.display_value_slider(label, val, scale, res)
                )

                slider.valueChanged.connect(
                    lambda val, pname=param_name, scale=scaling_factor, res=decimal_places: self.update_parameter(pname, val, scale, res)
                )

                # Add the slider itself
                h1_layout.addWidget(slider)

                # Create a label for displaying the live value on the right
                value_label = QLabel(f"{initial_val:.{decimal_places}f}")
                value_label.setAlignment(Qt.AlignRight)

                # Set green text color for the value label
                palette = value_label.palette()
                palette.setColor(QPalette.WindowText, QColor("green"))
                value_label.setPalette(palette)

                h1_layout.addWidget(value_label)

                v_layout.addLayout(h1_layout)

                # Add an input box below the slider
                input_box = QLineEdit()
                input_box.setPlaceholderText("Enter value")
                input_box.setAlignment(Qt.AlignCenter)

                # Connect the input box to update the slider
                input_box.editingFinished.connect(
                    lambda box=input_box, min_value=min_val, max_value=max_val, pname=param_name, scale=scaling_factor, item=slider: self.update_slider_from_input(box, min_value, max_value, scale, item)
                )

                v_layout.addWidget(input_box)

                # Save the value label for updates
                self.param_labels[param_name] = value_label

                layout.addLayout(v_layout)

                self.logger.log(f"done")

            except Exception as e:
                print(f"Error adding parameter {param_name}: {e}")

        # Set the container widget in the scroll area
        scroll_area.setWidget(container_widget)

        # Add the scroll area to the main layout
        main_layout.addWidget(scroll_area)

        # Add the Reset button at the bottom, outside of the scroll area
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(
            lambda scale=scaling_factor: self.reset_sliders(scaling_factor)
        )

        main_layout.addWidget(reset_button)

        # Add the Save button at the bottom, outside of the scroll area
        reset_button = QPushButton("Save")
        reset_button.clicked.connect(self.save_values)

        main_layout.addWidget(reset_button)

        self.logger.log(f"Initializing parameter GUI.")
        # Set the central widget layout
        self.setCentralWidget(central_widget)

    def __round_value(self, value, scale, decimal_places):

        val = f'%.{decimal_places}f' % (value / scale)
        return val

    def display_value_slider(self, label, value, scale, res):

        label.setText(self.__round_value(value, scale, res))

    def update_slider_from_input(self, input_box, min_val, max_val, scaling_factor, slider):
        try:
            input_value = float(input_box.text())
            if min_val <= input_value <= max_val:
                slider.setValue(int(input_value * scaling_factor))
            else:
                input_box.setText("")  # Clear invalid input
        except ValueError:
            input_box.setText("")  # Clear invalid input

    def parse_descriptor(self, descriptor):
        """Parse the descriptor string to extract min and max values."""
        min_val = 0
        max_val = 100

        # Example descriptor: "{type: InRange, min: -0.005, max: 0.005}"
        min_match = re.search(r"min:\s*(-?\d+(\.\d+)?)", descriptor)
        max_match = re.search(r"max:\s*(-?\d+(\.\d+)?)", descriptor)

        if min_match and max_match:
            min_val = float(min_match.group(1))
            max_val = float(max_match.group(1))

        return min_val, max_val

    def update_parameter(self, param_name, value, scale, res):
        """Update the parameter on the ROS server and update the label."""

        value_scaled = self.__round_value(value, scale, res)

        try:
            rospy.wait_for_service('/horizon/set_parameters', timeout=2)
            set_param = rospy.ServiceProxy('/horizon/set_parameters', SetString)
            # Format the request string correctly
            request_string = f"{param_name}: {value_scaled}"

            response = set_param(request=request_string)

            if response.success:
                print(f"Successfully set {param_name} to {value_scaled}")
                # Refresh the live value display and keep the value label green8
                self.refresh_parameter_value(param_name, res)
                self.set_value_label_color(param_name, "green")
                self.current_values[param_name] = value_scaled
            else:
                print(f"Failed to set {param_name} to {value_scaled}: {response.message}")
                # If the update failed, change the label to red
                self.set_value_label_color(param_name, "red")
        except rospy.ServiceException as e:
            print(f"Service call failed for {param_name}: {e}")
            # If the service call fails, change the label to red
            self.set_value_label_color(param_name, "red")
        except rospy.ROSException as e:
            print(f"Service timeout for {param_name}: {e}")
            # If there’s a ROS timeout error, change the label to red
            self.set_value_label_color(param_name, "red")

    def set_value_label_color(self, param_name, color):
        """Set the color of the parameter value label."""
        if param_name in self.param_labels:
            value_label = self.param_labels[param_name]
            palette = value_label.palette()
            palette.setColor(QPalette.WindowText, QColor(color))
            value_label.setPalette(palette)

    def refresh_parameter_value(self, param_name, res):
        """Fetch the latest value of the parameter and update its label."""
        try:
            rospy.wait_for_service('/horizon/get_parameter_info', timeout=2)
            get_params = rospy.ServiceProxy('/horizon/get_parameter_info', GetParameterInfo)
            response = get_params(tunable_only=False, name=[''])

            # Update the specific parameter's label
            for name, value in zip(response.name, response.value):
                if name == param_name and name in self.param_labels:
                    self.param_labels[name].setText(str(f"{float(value):.{res}f}"))
                    # Set the value label color to green when updated successfully
                    self.set_value_label_color(param_name, "green")
                    break
        except rospy.ServiceException as e:
            print(f"Service call failed while refreshing: {e}")
            self.set_value_label_color(param_name, "red")
        except rospy.ROSException as e:
            print(f"Service timeout while refreshing: {e}")
            self.set_value_label_color(param_name, "red")

    def reset_sliders(self, scaling_factor):
        """Reset all sliders to their original values."""
        for param_name, slider in self.param_sliders.items():
            # Reset the slider value to its initial value
            initial_value = self.current_values[param_name]
            slider.setValue(int(initial_value * scaling_factor))


            # Reset the displayed value on the label to its initial value
            if param_name in self.param_labels:
                self.param_labels[param_name].setText(f"{initial_value:.3f}")

                # Set the label color to green
                self.set_value_label_color(param_name, "green")

    def save_values(self):
        """Save the current values of the sliders to a YAML file."""
        # Open a file dialog to choose where to save the file
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Values", "", "YAML Files (*.yaml);;All Files (*)", options=options)

        if file_path:
            try:
                # Write the current values to the selected YAML file
                with open(file_path, 'w') as file:
                    yaml.dump(self.current_values, file, default_flow_style=False)
                print(f"Values saved successfully to {file_path}")
            except Exception as e:
                print(f"Failed to save values: {e}")

def fetch_ros_parameters():
    rospy.wait_for_service('/horizon/get_parameter_info')
    try:
        get_params = rospy.ServiceProxy('/horizon/get_parameter_info', GetParameterInfo)
        response = get_params(tunable_only=False, name=[''])
        params = []
        for name, value, descriptor in zip(response.name, response.value, response.descriptor):
            params.append((name, value, descriptor))  # Pass descriptor as string
        return params
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")
        return []





def main():
    rospy.init_node('dynamic_sliders_gui', anonymous=True)
    params = fetch_ros_parameters()

    app = QApplication(sys.argv)
    main_window = DynamicSliderWindow(params)
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()