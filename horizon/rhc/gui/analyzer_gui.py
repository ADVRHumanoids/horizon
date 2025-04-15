import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QTabWidget, QVBoxLayout, QListWidget,
    QLabel, QHBoxLayout, QMainWindow, QListWidgetItem, QTextEdit, QCheckBox, QPushButton
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QVBoxLayout
from PyQt5.QtCore import Qt

from horizon.functions import Constraint, Cost, RecedingCost, RecedingConstraint
from horizon.variables import Parameter, RecedingParameter, RecedingVariable, Variable

class NodeDisplay(QWidget):
    nodeClicked = pyqtSignal(int)  # ← add this line

    def __init__(self, total_nodes):
        super().__init__()
        self.total_nodes = total_nodes
        self.active_nodes = []
        self.nodeRects = {}  # ← keep track of rects for clicks
        self.unbounded_nodes = set()

    def setUnboundedNodes(self, node_ids):
        self.unbounded_nodes = set(node_ids)
        self.update()  # trigger redraw

    def setActiveNodes(self, nodes):
        self.active_nodes = nodes
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width = self.width()
        height = self.height()

        node_width = 30
        node_height = 24
        spacing = 10
        cols = max(1, width // (node_width + spacing))
        margin = 10

        self.nodeRects.clear()

        for i in range(self.total_nodes):
            row = i // cols
            col = i % cols
            x = margin + col * (node_width + spacing)
            y = margin + row * (node_height + spacing)

            self.nodeRects[i] = (x, y, node_width, node_height)

            if i in self.unbounded_nodes:
                # Blue border, white fill
                painter.setBrush(QColor("#ffffff"))
                painter.setPen(QColor("#3498db"))
            elif i in self.active_nodes:
                # Blue fill, dark border
                painter.setBrush(QColor("#3498db"))
                painter.setPen(QColor("#2c3e50"))
            else:
                # Light gray fill, light gray border
                painter.setBrush(QColor("#ecf0f1"))
                painter.setPen(QColor("#bdc3c7"))

            painter.drawRoundedRect(x, y, node_width, node_height, 6, 6)

            # Node label
            painter.setPen(QColor("#2c3e50"))
            painter.drawText(x + 6, y + 16, str(i))

    def mousePressEvent(self, event):
        pos = event.pos()
        for node, rect in self.nodeRects.items():
            x, y, w, h = rect
            if x <= pos.x() <= x + w and y <= pos.y() <= y + h:
                self.nodeClicked.emit(node)
                break

# ---- The GUI Tab ---- #
class ElementTab(QWidget):
    def __init__(self, elements_dict, showDetailsFn, total_nodes=41):
        super().__init__()
        self.layout = QVBoxLayout(self)

        self.listWidget = QListWidget()
        for name, obj in elements_dict.items():
            item = QListWidgetItem(name)
            item.setData(1000, (name, obj))  # store both name and object
            self.listWidget.addItem(item)

        # Add a checkbox to toggle the details box
        self.checkboxShowDetails = QCheckBox("Show details")
        self.checkboxShowDetails.setChecked(True)
        self.checkboxShowDetails.stateChanged.connect(self.toggleDetailsBox)
        self.layout.addWidget(self.checkboxShowDetails)

        # Create and add the details box
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(100)  # Smaller by default

        self.nodeDisplay = NodeDisplay(total_nodes=total_nodes)
        self.nodeDisplay.setMinimumHeight(160)

        self.listWidget.currentItemChanged.connect(self.showDetails)

        self.layout.addWidget(self.listWidget)
        self.layout.addWidget(self.nodeDisplay)
        self.layout.addWidget(self.details)
        self.showDetailsFn = showDetailsFn

        self.currentTables = None

        self.checkboxShowInactive = QCheckBox("Show inactive nodes")
        self.checkboxShowInactive.setChecked(True)
        self.layout.addWidget(self.checkboxShowInactive)

        self.nodeDisplay.nodeClicked.connect(self.highlightColumnForNode)

    def toggleDetailsBox(self):
        self.details.setVisible(self.checkboxShowDetails.isChecked())

    def highlightColumnForNode(self, node):
        if not self.currentTables:
            return

        for current_tab in self.currentTables:
            for col in range(current_tab.columnCount()):
                header_item = current_tab.horizontalHeaderItem(col)
                if header_item and int(header_item.text()) == node:
                    self.highlightColumn(col)
                    item = current_tab.item(0, col)
                    if item:
                        current_tab.scrollToItem(item, QTableWidget.PositionAtCenter)
                    break

    def highlightColumn(self, col):
        if not self.currentTables:
            return

        for current_tab in self.currentTables:

            for r in range(current_tab.rowCount()):
                item = current_tab.item(r, col)
                if item:
                    item.setBackground(QColor("#cceeff"))

            for other_col in range(current_tab.columnCount()):
                if other_col == col:
                    continue
                for r in range(current_tab.rowCount()):
                    item = current_tab.item(r, other_col)
                    if item:
                        item.setBackground(Qt.white if item.text() else QColor("#eeeeee"))

    def showDetails(self, item):
        if item:
            name, obj = item.data(1000)
            self.details.setText(self.showDetailsFn(name, obj))
            nodes = obj.getNodes() if hasattr(obj, "getNodes") else []
            self.nodeDisplay.setActiveNodes(nodes)
            self.populateTable(name, obj)

    def populateTable(self, name, obj):
        # Remove old tables (if any)
        if hasattr(self, 'lowerTable') and self.lowerTable is not None:
            self.layout.removeWidget(self.lowerTable)
            self.lowerTable.deleteLater()
            self.lowerTable = None

        if hasattr(self, 'upperTable') and self.upperTable is not None:
            self.layout.removeWidget(self.upperTable)
            self.upperTable.deleteLater()
            self.upperTable = None

        if hasattr(self, 'valueTable') and self.valueTable is not None:
            self.layout.removeWidget(self.valueTable)
            self.valueTable.deleteLater()
            self.valueTable = None

        show_inactive = self.checkboxShowInactive.isChecked()
        self.checkboxShowInactive.stateChanged.connect(self.refreshTable)
        self.checkboxShowInactive.setChecked(True)
        has_bounds = hasattr(obj, "getBounds")
        has_values = hasattr(obj, "getValues")
        has_nodes = hasattr(obj, "getNodes")

        active_nodes = obj.getNodes() if has_nodes else []
        total_nodes = self.nodeDisplay.total_nodes
        all_nodes = list(range(total_nodes)) if show_inactive else active_nodes

        if has_bounds:
            lower, upper = obj.getBounds()
            lower = np.atleast_2d(np.array(lower))
            upper = np.atleast_2d(np.array(upper))

            unbounded_nodes = set()
            if lower is not None and upper is not None:
                for i, node in enumerate(active_nodes):
                    if np.all(lower[:, i] == -np.inf) and np.all(upper[:, i] == np.inf):
                        unbounded_nodes.add(node)

            self.nodeDisplay.setUnboundedNodes(unbounded_nodes)


            dim = lower.shape[0]
            row_labels_lower = [f"Lower Bound [dim {d}]" for d in range(dim)]
            row_labels_upper = [f"Upper Bound [dim {d}]" for d in range(dim)]

            # Create and add lower bounds table
            self.lowerTable = QTableWidget()
            self.setupBoundsTable(self.lowerTable, lower, row_labels_lower, all_nodes)
            self.layout.addWidget(self.lowerTable)


            # Create and add upper bounds table
            self.upperTable = QTableWidget()
            self.setupBoundsTable(self.upperTable, upper, row_labels_upper, all_nodes)
            self.layout.addWidget(self.upperTable)

            self.currentTables = [self.lowerTable, self.upperTable]

        elif has_values:
            values = np.atleast_2d(np.array(obj.getValues()))
            dim = values.shape[0]
            row_labels = [f"Value [dim {d}]" for d in range(dim)]

            self.valueTable = QTableWidget()
            self.setupBoundsTable(self.valueTable, values, row_labels, all_nodes)
            self.layout.addWidget(self.valueTable)
            self.currentTables = [self.valueTable]

        else:
            self.currentTables = None

    def refreshTable(self):
        current_item = self.listWidget.currentItem()
        if current_item:
            self.showDetails(current_item)

    def setupBoundsTable(self, tableWidget, rows, row_labels, all_nodes):
        tableWidget.setEditTriggers(QTableWidget.NoEditTriggers)
        tableWidget.setSelectionMode(QTableWidget.NoSelection)
        tableWidget.setFocusPolicy(Qt.NoFocus)

        tableWidget.setRowCount(len(row_labels))
        tableWidget.setColumnCount(len(all_nodes))

        tableWidget.setHorizontalHeaderLabels([str(n) for n in all_nodes])
        tableWidget.setVerticalHeaderLabels(row_labels)

        for row_idx, label in enumerate(row_labels):
            for col_idx, node in enumerate(all_nodes):
                item = QTableWidgetItem()
                value = rows[row_idx][col_idx] if row_idx < len(rows) and col_idx < len(rows[row_idx]) else None

                if value is not None:
                    item.setText(f"{value:.4g}")
                    item.setTextAlignment(Qt.AlignCenter)
                else:
                    item.setText("")
                    item.setBackground(QColor("#eeeeee"))
                    item.setForeground(QColor("#aaaaaa"))

                tableWidget.setItem(row_idx, col_idx, item)


# ---- Main Window ---- #
class AnalyzerGUI(QMainWindow):
    def __init__(self, problem):
        super().__init__()
        self.setWindowTitle("Model Explorer with Visual Nodes")
        self.resize(800, 600)

        self.__prb = problem
        tabs = QTabWidget()

        self.__total_modes = self.__prb.getNNodes()
        analyzeTabs = QTabWidget()
        analyzeTabs.addTab(
            ElementTab(self.__prb.getConstraints(), self.describeConstraint, self.__total_modes),
            "Constraints"
        )
        analyzeTabs.addTab(
            ElementTab(self.__prb.getCosts(), self.describeGeneric, self.__total_modes),
            "Costs"
        )
        analyzeTabs.addTab(
            ElementTab(self.__prb.getVariables(), self.describeGeneric, self.__total_modes),
            "Variables"
        )
        analyzeTabs.addTab(
            ElementTab(self.__prb.getParameters(), self.describeParameter, self.__total_modes),
            "Parameters"
        )
        tabs.addTab(analyzeTabs, "Analyze")
        tabs.addTab(CompareTab(self.__prb, self.__total_modes), "Compare")

        self.setCentralWidget(tabs)

    def describeConstraint(self, name, constraint):
        nodes = ", ".join(map(str, constraint.getNodes()))
        bounds_str = ""

        if hasattr(constraint, "getBounds"):
            bounds = constraint.getBounds()
            if isinstance(bounds, tuple) and len(bounds) == 2:
                lower, upper = bounds

                # Flatten and format arrays cleanly
                lower_flat = np.ravel(lower)
                upper_flat = np.ravel(upper)

                lower_str = ", ".join(f"{v:.4g}" for v in lower_flat)
                upper_str = ", ".join(f"{v:.4g}" for v in upper_flat)

                bounds_str = f"\nLower Bounds: [{lower_str}]\nUpper Bounds: [{upper_str}]"
            else:
                bounds_str = f"\nBounds: {bounds}"

        return f"Name: {name}\nNodes: {nodes}\n{bounds_str}"

    def describeParameter(self, name, param):
        nodes = ", ".join(map(str, param.getNodes())) if hasattr(param, "getNodes") else ""
        values_str = ""

        if hasattr(param, "getValues"):
            values = param.getValues()

            # Flatten and format arrays cleanly
            values_flat = np.ravel(values)

            values_str = ", ".join(f"{v:.4g}" for v in values_flat)

        return f"Name: {name}\nNodes: {nodes}\nValues: {values_str}"

    def describeGeneric(self, name, obj):
        nodes = ", ".join(map(str, obj.getNodes())) if hasattr(obj, "getNodes") else ""
        return f"Name: {name}\nNodes: {nodes}"


class CompareTab(QWidget):
    COLORS = [
        "#e74c3c", "#8e44ad", "#3498db", "#16a085",
        "#f39c12", "#d35400", "#2ecc71", "#7f8c8d",
        "#1abc9c", "#c0392b", "#9b59b6", "#34495e"
    ]

    def __init__(self, model, total_nodes):
        super().__init__()
        self.model = model
        self.total_nodes = total_nodes
        self.layout = QVBoxLayout(self)

        # Create a layout to hold the category widgets
        self.categoryLists = {}  # Will hold { "Constraint": QListWidget, ... }

        categories = [("Constraints", model.getConstraints),
                      ("Costs", model.getCosts),
                      ("Variables", model.getVariables),
                      ("Parameters", model.getParameters)]

        from PyQt5.QtWidgets import QGroupBox, QGridLayout

        gridLayout = QGridLayout()
        for i, (label, getter) in enumerate(categories):
            group = QGroupBox(label)
            vbox = QVBoxLayout(group)

            listWidget = QListWidget()
            listWidget.setSelectionMode(QListWidget.MultiSelection)
            listWidget.itemSelectionChanged.connect(self.updateDisplay)

            self.categoryLists[label] = listWidget
            self.populateList(listWidget, label, getter)

            vbox.addWidget(listWidget)
            group.setLayout(vbox)

            gridLayout.addWidget(group, i // 2, i % 2)

        self.layout.addLayout(gridLayout)

        # Node display
        self.nodeDisplay = NodeDisplayMulti(total_nodes)
        self.layout.addWidget(self.nodeDisplay)

        # Reset button
        self.resetButton = QPushButton("Reset Selection")
        self.resetButton.clicked.connect(self.resetSelection)
        self.layout.addWidget(self.resetButton)

    def populateList(self, listWidget, category_name, getter):
        items = getter()
        if isinstance(items, dict):
            items = items.items()
        else:
            items = [(getattr(i, "name", f"{category_name}"), i) for i in items]
        for name, obj in items:
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, obj)
            listWidget.addItem(item)

    def updateDisplay(self):
        selected_items = []
        for listWidget in self.categoryLists.values():
            selected_items.extend(listWidget.selectedItems())

        data = []
        for idx, item in enumerate(selected_items):
            color = QColor(self.COLORS[idx % len(self.COLORS)])
            obj = item.data(Qt.UserRole)
            nodes = obj.getNodes() if hasattr(obj, "getNodes") else []
            data.append((nodes, color))

        self.nodeDisplay.setColorGroups(data)

    def resetSelection(self):
        for listWidget in self.categoryLists.values():
            listWidget.clearSelection()
        self.nodeDisplay.setColorGroups([])

class NodeDisplayMulti(QWidget):
    def __init__(self, total_nodes):
        super().__init__()
        self.total_nodes = total_nodes
        self.node_colors = {}  # Store color for each node
        self.setMinimumHeight(200)  # Increase the minimum height to accommodate bigger nodes

    def setColorGroups(self, data):
        self.node_colors.clear()  # Clear existing colors
        for nodes, color in data:
            for node in nodes:
                if node not in self.node_colors:
                    self.node_colors[node] = []
                self.node_colors[node].append(color)  # Add color to this node
        self.update()  # Trigger a re-paint

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width = self.width()
        height = self.height()

        # Increase the node width and height for bigger boxes
        node_width = 40
        node_height = 30
        spacing = 12  # Add a bit more space between nodes
        cols = max(1, width // (node_width + spacing))
        margin = 10

        for i in range(self.total_nodes):
            row = i // cols
            col = i % cols
            x = margin + col * (node_width + spacing)
            y = margin + row * (node_height + spacing)

            # Get colors for this node
            active_groups = self.node_colors.get(i, [])

            if not active_groups:
                # No colors assigned to the node
                painter.setBrush(QColor("#ecf0f1"))
                painter.setPen(QColor("#bdc3c7"))
                painter.drawRoundedRect(x, y, node_width, node_height, 6, 6)
            else:
                # Draw each color as a stripe with transparency
                stripe_height = node_height // len(active_groups)  # Divide the node height by number of colors
                for idx, color in enumerate(active_groups):
                    stripe_y = y + idx * stripe_height
                    painter.setBrush(color)
                    painter.setOpacity(0.7)  # Set transparency for blending
                    painter.setPen(Qt.NoPen)
                    painter.drawRect(x, stripe_y, node_width, stripe_height)
                    painter.setOpacity(1.0)  # Reset opacity for the next color

                # Border around the whole box
                painter.setPen(QColor("#2c3e50"))
                painter.setBrush(Qt.NoBrush)
                painter.drawRoundedRect(x, y, node_width, node_height, 6, 6)

            # Draw node index (centered)
            painter.setPen(QColor("#2c3e50"))
            painter.drawText(x + 6, y + 18, str(i))  # Adjust the position for larger node