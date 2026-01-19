# admin.py
import sys
from PySide6.QtWidgets import QApplication
from src.ui.admin_window import AdminWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AdminWindow()
    window.show()
    sys.exit(app.exec())