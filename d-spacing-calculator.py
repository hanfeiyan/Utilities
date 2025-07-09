import sys
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSpinBox, QTextEdit, QComboBox
)
from PyQt5.QtGui import QDoubleValidator

class SpacingCalculator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("d-spacing Calculator")
        self.init_ui()
        self.init_silicon()

    def init_ui(self):
        self.edita, self.editb, self.editc = QLineEdit(), QLineEdit(), QLineEdit()
        self.editA, self.editB, self.editC = QLineEdit(), QLineEdit(), QLineEdit()
        self.edith, self.editk, self.editl = QSpinBox(), QSpinBox(), QSpinBox()
        self.energy = QLineEdit()
        self.energy.setValidator(QDoubleValidator())
        self.max_index = QSpinBox()
        self.max_index.setRange(1, 20)
        self.structure = QComboBox()
        self.structure.addItems(["None", "FCC", "BCC", "Diamond", "Hexagonal"])
        self.structure.currentIndexChanged.connect(self.calculate_peaks)
        self.spacing = QLineEdit()
        self.spacing.setReadOnly(True)
        self.peak_output = QTextEdit()
        self.peak_output.setReadOnly(True)
        self.peak_output.setLineWrapMode(QTextEdit.NoWrap)

        for edit in [self.edita, self.editb, self.editc, self.editA, self.editB, self.editC]:
            edit.setValidator(QDoubleValidator())

        layout = QVBoxLayout(self)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("a (Å)"))
        row1.addWidget(self.edita)
        row1.addWidget(QLabel("b (Å)"))
        row1.addWidget(self.editb)
        row1.addWidget(QLabel("c (Å)"))
        row1.addWidget(self.editc)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("α ∠(b,c) (°)"))
        row2.addWidget(self.editA)
        row2.addWidget(QLabel("β ∠(c,a) (°)"))
        row2.addWidget(self.editB)
        row2.addWidget(QLabel("γ ∠(a,b) (°)"))
        row2.addWidget(self.editC)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("h"))
        row3.addWidget(self.edith)
        row3.addWidget(QLabel("k"))
        row3.addWidget(self.editk)
        row3.addWidget(QLabel("l"))
        row3.addWidget(self.editl)
        layout.addLayout(row3)

        hlayout = QHBoxLayout()
        calc_btn = QPushButton("Calculate d-spacing")
        calc_btn.clicked.connect(self.calculate)
        hlayout.addWidget(calc_btn)
        hlayout.addWidget(QLabel("d (Å) ="))
        hlayout.addWidget(self.spacing)
        layout.addLayout(hlayout)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Energy (keV)"))
        row4.addWidget(self.energy)
        row4.addWidget(QLabel("Max Index"))
        row4.addWidget(self.max_index)
        row4.addWidget(QLabel("Structure"))
        row4.addWidget(self.structure)
        layout.addLayout(row4)

        peak_btn = QPushButton("Calculate Peaks")
        peak_btn.clicked.connect(self.calculate_peaks)
        layout.addWidget(peak_btn)

        layout.addWidget(QLabel("Bragg Peaks (hkl, d-spacing, 2θ, allowed):"))
        layout.addWidget(self.peak_output)

    def init_silicon(self):
        self.edita.setText("5.431")
        self.editb.setText("5.431")
        self.editc.setText("5.431")
        self.editA.setText("90")
        self.editB.setText("90")
        self.editC.setText("90")
        self.edith.setValue(1)
        self.editk.setValue(1)
        self.editl.setValue(1)
        self.energy.setText("12")
        self.max_index.setValue(3)
        self.structure.setCurrentText("Diamond")

    def vec_abs(self, v):
        return np.linalg.norm(v)

    def vec_cross(self, v1, v2):
        return np.cross(v1, v2)

    def vec_dot(self, v1, v2):
        return np.dot(v1, v2)

    def is_forbidden(self, h, k, l, structure):
        if structure == "FCC":
            return not (h % 2 == k % 2 == l % 2)
        elif structure == "BCC":
            return (h + k + l) % 2 != 0
        elif structure == "Diamond":
            return not (h % 2 == k % 2 == l % 2 == 1 or (h % 2 == k % 2 == l % 2 == 0 and (h + k + l) % 4 == 0))
        elif structure == "Hexagonal":
            # Basic rule: l must be even if (h - k) % 3 != 0 for reflection to be allowed
            #             (00l) is forbidden if l is odd 
            return ((h - k) % 3 != 0 and l % 2 != 0) or (h == 0 and k == 0 and l % 2 == 1)
        return False
    def calculate(self):
        try:
            a = float(self.edita.text())
            b = float(self.editb.text())
            c = float(self.editc.text())
            alpha = float(self.editA.text()) * np.pi / 180.0
            beta = float(self.editB.text()) * np.pi / 180.0
            gamma = float(self.editC.text()) * np.pi / 180.0
            h = int(self.edith.value())
            k = int(self.editk.value())
            l = int(self.editl.value())

            a1 = np.array([a, 0, 0])
            a2 = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0])
            c1 = np.cos(alpha) - np.cos(beta) * np.cos(gamma)
            c2 = np.sin(gamma)**2 * np.sin(beta)**2 - c1**2

            if c2 <= 0:
                self.spacing.setText("NaN")
                return

            ty = c1 / np.sqrt(c2)
            tx = np.cos(beta) * np.sqrt(1.0 + ty**2) / np.sin(beta)
            a3 = np.array([tx, ty, 1.0])
            a3 = c * a3 / self.vec_abs(a3)

            vol = self.vec_dot(self.vec_cross(a2, a3), a1)
            b1 = self.vec_cross(a2, a3) / vol
            b2 = self.vec_cross(a3, a1) / vol
            b3 = self.vec_cross(a1, a2) / vol

            diff_h = h * b1 + k * b2 + l * b3
            d_spacing = 1.0 / self.vec_abs(diff_h)
            self.spacing.setText(f"{d_spacing:.6f}")

        except Exception as e:
            self.spacing.setText("Error")
            print("Error:", e)


    def calculate_peaks(self):
        try:
            energy = float(self.energy.text())
            wavelength = 12.3984 / energy
            max_hkl = int(self.max_index.value())
            structure = self.structure.currentText()

            a = float(self.edita.text())
            b = float(self.editb.text())
            c = float(self.editc.text())
            alpha = float(self.editA.text()) * np.pi / 180.0
            beta = float(self.editB.text()) * np.pi / 180.0
            gamma = float(self.editC.text()) * np.pi / 180.0

            a1 = np.array([a, 0, 0])
            a2 = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0])
            c1 = np.cos(alpha) - np.cos(beta) * np.cos(gamma)
            c2 = np.sin(gamma)**2 * np.sin(beta)**2 - c1**2
            if c2 <= 0:
                self.peak_output.setText("Invalid unit cell geometry.")
                return

            ty = c1 / np.sqrt(c2)
            tx = np.cos(beta) * np.sqrt(1.0 + ty**2) / np.sin(beta)
            a3 = np.array([tx, ty, 1.0])
            a3 = c * a3 / self.vec_abs(a3)
            vol = self.vec_dot(self.vec_cross(a2, a3), a1)
            b1 = self.vec_cross(a2, a3) / vol
            b2 = self.vec_cross(a3, a1) / vol
            b3 = self.vec_cross(a1, a2) / vol

            peak_dict = {}
            for h in range(0, max_hkl + 1):
                for k in range(0, max_hkl + 1):
                    for l in range(0, max_hkl + 1):
                        if h == k == l == 0:
                            continue
                        g = h * b1 + k * b2 + l * b3
                        d = 1.0 / self.vec_abs(g)
                        try:
                            theta = np.arcsin(wavelength / (2 * d))
                        except ValueError:
                            continue
                        if np.isnan(theta) or theta <= 0:
                            continue
                        twotheta = 2 * theta * 180 / np.pi
                        if self.is_forbidden(h, k, l, structure):
                            continue
                        key = round(twotheta, 4)
                        if key not in peak_dict:
                            peak_dict[key] = {"d": d, "hkl": []}
                        peak_dict[key]["hkl"].append((h, k, l))

            # lines = ["hkl\td (Å)\t2θ (°)\tAllowed"]
            # for twotheta in sorted(peak_dict.keys()):
            #     d = peak_dict[twotheta]["d"]
            #     hkl_str = ", ".join([f"({h}{k}{l})" for h, k, l in peak_dict[twotheta]["hkl"]])
            #     lines.append(f"{hkl_str}\t{d:.4f}\t{twotheta:.2f}\tYes")
            lines = ["hkl\td (Å)\t2θ (°)\tAllowed"]
            for twotheta in sorted(peak_dict.keys()):
                d = peak_dict[twotheta]["d"]
                hkl_list = peak_dict[twotheta]["hkl"]
    
                for i, (h, k, l) in enumerate(hkl_list):
                    if i == len(hkl_list) - 1:
                    # Last hkl in group: show full data
                        lines.append(f"({h}{k}{l})\t{d:.4f}\t{twotheta:.2f}\tYes")
                    else:
                        # Intermediate hkl in group: leave d and twotheta empty
                        lines.append(f"({h}{k}{l})\t\t\tYes")
            self.peak_output.setPlainText("\n".join(lines))
        #     lines = ["{:<20s}{:<12s}{:<10s}{}".format("hkl", "d (Å)", "2θ (°)", "Allowed")]
        #     for twotheta in sorted(peak_dict.keys()):
        #         d = peak_dict[twotheta]["d"]
        #         hkl_str = ", ".join([f"({h}{k}{l})" for h, k, l in peak_dict[twotheta]["hkl"]])
        #         lines.append("{:<20s}{:<12.4f}{:<10.2f}{}".format(hkl_str, d, twotheta, "Yes"))
        #     self.peak_output.setPlainText("\n".join(lines))
        except Exception as e:
            self.peak_output.setText("Error in peak calculation.")
            print("Error:", e)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpacingCalculator()
    window.show()
    sys.exit(app.exec_())

