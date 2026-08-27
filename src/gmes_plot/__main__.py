from __future__ import annotations

import sys


def main() -> int:
    try:
        from PySide6.QtGui import QIcon
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("缺少 PySide6。请先执行: python -m pip install -e .")
        return 2

    from gmes_plot.ui.main_window import MainWindow
    from gmes_plot.resources import asset_path

    app = QApplication(sys.argv)
    app.setApplicationName("重磁电震绘图 Pro")
    app.setOrganizationName("GMES-Plot")
    app.setWindowIcon(QIcon(str(asset_path("gmes_plot_icon.png"))))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

