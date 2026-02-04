# -*- coding: utf-8 -*-
import numpy as np
import pyqtgraph as pg

try:
    from PyQt5.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGroupBox, QPushButton, QLabel, QFormLayout, QGridLayout,
        QComboBox, QLineEdit, QSpinBox, QTabWidget, QScrollArea, QCheckBox,
        QSplitter
    )
    from PyQt5.QtGui import QFont
    from PyQt5.QtCore import Qt, pyqtSlot
except ImportError:
    from PySide6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGroupBox, QPushButton, QLabel, QFormLayout, QGridLayout,
        QComboBox, QLineEdit, QSpinBox, QTabWidget, QScrollArea, QCheckBox,
        QSplitter
    )
    from PySide6.QtGui import QFont
    from PySide6.QtCore import Qt, Slot as pyqtSlot

from ViewModel import ViewModel

# pyqtgraph 설정
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


# --- [REUSABLE] Sensor Graph View ---
class SensorGraphView(QWidget):
    def __init__(self, sensor_id, parent=None):
        super().__init__(parent)
        self.sensor_id = sensor_id
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)

        self.tx_plot = self._create_plot("TX Signal (Time Domain)", 'Voltage (mV)', 'Time (ms)', 'b')
        self.tx_plot['widget'].setYRange(-3500, 3500)
        # Add TX start marker (green dashed line)
        self.tx_marker = pg.InfiniteLine(angle=90, pen=pg.mkPen('g', width=2, style=Qt.DashLine), movable=False)
        self.tx_plot['widget'].addItem(self.tx_marker)
        self.tx_marker.setVisible(False)
        self.tx_plot['marker'] = self.tx_marker
        self.layout.addWidget(self.tx_plot['group'], 15)

        self.rx_plot = self._create_plot("RX Signal (Time Domain)", 'Voltage (mV)', 'Time (ms)', 'r')
        self.rx_plot['widget'].setYRange(-3500, 3500)
        # Add RX start (ToF) marker (orange dashed line)
        self.rx_marker = pg.InfiniteLine(angle=90, pen=pg.mkPen('orange', width=2, style=Qt.DashLine), movable=False)
        self.rx_plot['widget'].addItem(self.rx_marker)
        self.rx_marker.setVisible(False)
        self.rx_plot['marker'] = self.rx_marker
        self.layout.addWidget(self.rx_plot['group'], 15)

        self.stft_widget = self._create_stft_group()
        self.layout.addWidget(self.stft_widget['group'], 30)

        self.fft_plot = self._create_plot("Full FFT (35k-45k Hz)", 'Magnitude', 'Frequency (Hz)', 'g')
        self.fft_plot['widget'].setXRange(35000, 45000)
        self.layout.addWidget(self.fft_plot['group'], 15)

        self.p_fft_plot = self._create_plot("Partial FFT", 'Magnitude', 'Frequency (Hz)', 'g')
        self.layout.addWidget(self.p_fft_plot['group'], 15)

    def _create_plot(self, title, left_label, bottom_label, color):
        group = QGroupBox(f" {title} ")
        group.setFont(QFont("Tw Cen MT Condensed Extra Bold", 11, QFont.Bold))
        plot_widget = pg.PlotWidget()
        plot_widget.setBackground('w')
        plot_widget.showGrid(x=True, y=True, alpha=0.3)
        plot_widget.setLabel('left', left_label)
        plot_widget.setLabel('bottom', bottom_label)
        curve = plot_widget.plot(pen=pg.mkPen(color, width=2), name='Signal')
        layout = QVBoxLayout()
        layout.addWidget(plot_widget)
        group.setLayout(layout)
        return {'group': group, 'widget': plot_widget, 'curve': curve}

    def _create_stft_group(self):
        group = QGroupBox(" Spectrogram (STFT) ")
        group.setFont(QFont("Tw Cen MT Condensed Extra Bold", 11, QFont.Bold))
        layout = QHBoxLayout()
        stft_plot = pg.PlotWidget()
        stft_plot.setLabel('bottom', 'Time (s)')
        stft_plot.setLabel('left', 'Frequency (Hz)')
        stft_plot.setAspectLocked(False)
        stft_plot.getPlotItem().showGrid(False, False)
        hist = pg.HistogramLUTWidget()
        hist.setMinimumWidth(80)
        img_item = pg.ImageItem()
        hist.setImageItem(img_item)
        hist.gradient.setColorMap(pg.colormap.get('plasma'))
        stft_plot.addItem(img_item)
        layout.addWidget(stft_plot, 1)
        layout.addWidget(hist)
        group.setLayout(layout)
        return {'group': group, 'widget': stft_plot, 'img': img_item, 'hist': hist}


# --- [REUSABLE] Sensor Output View ---
class SensorOutputView(QWidget):
    def __init__(self, sensor_id, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        val_font = QFont("Arial", 14, QFont.Bold)

        mlx_group = QGroupBox(f" Sensor {sensor_id} - MLX90614 ")
        mlx_group.setFont(QFont("Arial", 10, QFont.Bold))
        mlx_layout = QFormLayout()
        self.mlx_label = QLabel("0.00 °C")
        self.mlx_label.setFont(val_font)
        self.mlx_label.setStyleSheet("color: blue;")
        mlx_layout.addRow("Object Temp:", self.mlx_label)
        mlx_group.setLayout(mlx_layout)

        bme_group = QGroupBox(f" Sensor {sensor_id} - BME280 ")
        bme_group.setFont(QFont("Arial", 10, QFont.Bold))
        bme_layout = QFormLayout()
        self.bme_temp = QLabel("0.00 °C")
        self.bme_hum = QLabel("0.00 %")
        self.bme_press = QLabel("0.00 hPa")
        for l in [self.bme_temp, self.bme_hum, self.bme_press]:
            l.setFont(val_font)
            l.setStyleSheet("color: darkgreen;")
        bme_layout.addRow("Ambient Temp:", self.bme_temp)
        bme_layout.addRow("Humidity:", self.bme_hum)
        bme_layout.addRow("Pressure:", self.bme_press)
        bme_group.setLayout(bme_layout)
        
        distance_group = QGroupBox(f" Sensor {sensor_id} - Distance ")
        distance_group.setFont(QFont("Arial", 10, QFont.Bold))
        distance_layout = QFormLayout()
        self.distance_label = QLabel("N/A mm")
        self.processing_time_label = QLabel("0.00 ms")
        for l in [self.distance_label, self.processing_time_label]:
            l.setFont(val_font)
            l.setStyleSheet("color: darkorange;")
        distance_layout.addRow("Distance:", self.distance_label)
        distance_layout.addRow("Processing Time:", self.processing_time_label)
        distance_group.setLayout(distance_layout)
        
        layout.addWidget(mlx_group)
        layout.addWidget(bme_group)
        layout.addWidget(distance_group)
        layout.addStretch(1)


# --- MainWindow ---
class MainWindow(QMainWindow):
    def __init__(self, view_model: ViewModel, parent=None):
        super().__init__(parent)
        self.view_model = view_model
        self.setWindowTitle("Multi-Sensor Embedded Viewer")
        self.resize(1600, 1000)

        # Controller 연결을 위한 참조 변수
        self.port_combobox = None
        self.baud_combobox = None
        self.btn_serialport = None
        self.acoustic_port_combos = []
        self.sensor_checkboxes = []
        self.acoustic_baud_combos = []

        # [신규] 설정 스핀박스들
        self.data_len_spinbox = QSpinBox()
        self.interval_spinbox = QSpinBox()

        # 위젯 초기화 (DSP)
        self.stft_nperseg_spinbox = QSpinBox()
        self.stft_noverlap_spinbox = QSpinBox()
        self.stft_nfft_spinbox = QSpinBox()
        self.partial_fft_start_spinbox = QSpinBox()
        self.partial_fft_end_spinbox = QSpinBox()
        
        # Distance calculation filter parameters
        self.distance_bandpass_low_spinbox = QSpinBox()
        self.distance_bandpass_high_spinbox = QSpinBox()
        self.distance_envelope_cutoff_spinbox = QSpinBox()
        self.distance_filter_order_spinbox = QSpinBox()

        # GPS
        self.gps_port_combobox = QComboBox()
        self.gps_baud_combobox = QComboBox()
        self.btn_gps_serialport = QPushButton("CONNECT GPS")
        self.img_gps_sp = QLabel()
        self.img_gps_sp.setFixedSize(20, 20)
        self.img_gps_sp.setStyleSheet("background-color: red; border-radius: 10px;")

        # Save & Graph Control
        self.save_path_lineedit = QLineEdit()
        self.browse_button = QPushButton("Browse...")
        self.save_data_button = QPushButton("Toggle Continuous Save")
        self.save_data_button.setCheckable(True)
        self.save_status_label = QLabel("SAVE: OFF")
        self.save_status_label.setStyleSheet("color: red; background-color: #F0F0F0; border: 1px solid #CCC;")
        self.save_status_label.setAlignment(Qt.AlignCenter)

        # [신규] 그래프 토글 버튼
        self.btn_toggle_graph = QPushButton("GRAPH: ON")
        self.btn_toggle_graph.setCheckable(True)
        self.btn_toggle_graph.setChecked(True)
        self.btn_toggle_graph.setFixedHeight(30)
        self.btn_toggle_graph.setStyleSheet(
            "color: white; background-color: green; border: 1px solid green; font-weight: bold;")

        # System Reset
        self.btn_refresh = QPushButton("🔄 SYSTEM RESET")

        # UI 구성
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Horizontal)

        left_panel = QWidget()
        left_panel.setMinimumWidth(420)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.left_tabs = QTabWidget()
        self.left_tabs.setStyleSheet(
            "QTabBar::tab { height: 40px; min-width: 150px; font-weight: bold; font-size: 11pt; }")

        self.settings_tab = self._create_settings_tab()
        self.left_tabs.addTab(self.settings_tab, "🛠️ SETTINGS")

        self.monitor_tab = self._create_monitor_tab()
        self.left_tabs.addTab(self.monitor_tab, "📊 OUTPUT")

        left_layout.addWidget(self.left_tabs)
        splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.graph_tabs = QTabWidget()
        self.graph_tabs.setStyleSheet("QTabBar::tab { height: 35px; width: 100px; font-weight: bold; }")

        self.sensor_graphs = []
        for i in range(1, 7):
            s_view = SensorGraphView(i)
            self.sensor_graphs.append(s_view)
            self.graph_tabs.addTab(s_view, f"Sensor {i}")

        right_layout.addWidget(self.graph_tabs)
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 35)
        splitter.setStretchFactor(1, 65)
        main_layout.addWidget(splitter)

        self.update_all_widgets_from_vm()

    def _create_settings_tab(self):
        widget = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)

        acoustic_group = QGroupBox(" Acoustic Sensors Port Setup ")
        acoustic_group.setFont(QFont("Arial", 11, QFont.Bold))
        a_layout = QGridLayout()
        a_layout.setVerticalSpacing(12)

        # Header
        a_layout.addWidget(QLabel("On/Off"), 0, 0)
        a_layout.addWidget(QLabel("Sensor"), 0, 1)
        a_layout.addWidget(QLabel("COM Port"), 0, 2)
        a_layout.addWidget(QLabel("Baud Rate"), 0, 3)

        # Data Length
        a_layout.addWidget(QLabel("Data Len:"), 0, 4)
        self.data_len_spinbox.setRange(100, 65535)
        self.data_len_spinbox.setValue(10000)
        self.data_len_spinbox.setSingleStep(100)
        a_layout.addWidget(self.data_len_spinbox, 0, 5)

        # Interval 설정
        a_layout.addWidget(QLabel("Interval(ms):"), 1, 4)
        self.interval_spinbox.setRange(10, 5000)
        self.interval_spinbox.setValue(1000)
        self.interval_spinbox.setSingleStep(50)
        a_layout.addWidget(self.interval_spinbox, 1, 5)

        baud_rates = ["9600", "115200", "921600"]
        for i in range(6):
            chk = QCheckBox()
            chk.setChecked(True if i == 0 else False)
            lbl = QLabel(f"Sensor {i + 1}")
            lbl.setFont(QFont("Arial", 10, QFont.Bold))
            c_port = QComboBox()
            c_port.setEditable(True)
            c_baud = QComboBox()
            c_baud.addItems(baud_rates)
            c_baud.setCurrentText("921600")

            self.sensor_checkboxes.append(chk)
            self.acoustic_port_combos.append(c_port)
            self.acoustic_baud_combos.append(c_baud)

            a_layout.addWidget(chk, i + 1, 0)
            a_layout.addWidget(lbl, i + 1, 1)
            a_layout.addWidget(c_port, i + 1, 2)
            a_layout.addWidget(c_baud, i + 1, 3)

            if i == 0:
                self.port_combobox = c_port
                self.baud_combobox = c_baud

        acoustic_group.setLayout(a_layout)
        layout.addWidget(acoustic_group)

        # Connection Button
        conn_layout = QHBoxLayout()
        self.btn_serialport = QPushButton("CONNECT ALL")
        self.btn_serialport.setFixedHeight(40)
        self.btn_serialport.setCheckable(True)
        self.btn_serialport.setStyleSheet("font-weight: bold; background-color: #EEE;")

        self.img_sp = QLabel()
        self.img_sp.setFixedSize(20, 20)
        self.img_sp.setStyleSheet("background-color: red; border-radius: 10px;")
        conn_layout.addWidget(self.img_sp)
        conn_layout.addWidget(self.btn_serialport)
        layout.addLayout(conn_layout)

        # GPS Setup
        gps_group = QGroupBox(" Global GPS Setup ")
        gps_group.setFont(QFont("Arial", 11, QFont.Bold))
        g_layout = QGridLayout()
        self.gps_baud_combobox.addItems(["9600", "115200"])
        self.gps_baud_combobox.setCurrentText("115200")
        self.btn_gps_serialport.setCheckable(True)
        self.btn_gps_serialport.setFixedHeight(30)
        g_layout.addWidget(QLabel("GPS Port:"), 0, 0)
        g_layout.addWidget(self.gps_port_combobox, 0, 1)
        g_layout.addWidget(QLabel("Baud:"), 1, 0)
        g_layout.addWidget(self.gps_baud_combobox, 1, 1)
        g_layout.addWidget(self.img_gps_sp, 2, 0)
        g_layout.addWidget(self.btn_gps_serialport, 2, 1)
        gps_group.setLayout(g_layout)
        layout.addWidget(gps_group)

        # DSP Setup
        dsp_group = QGroupBox(" Signal Processing (DSP) Setup ")
        dsp_group.setFont(QFont("Arial", 11, QFont.Bold))
        d_layout = QFormLayout()
        self.stft_nperseg_spinbox.setRange(32, 8192);
        self.stft_nperseg_spinbox.setValue(1024)
        self.stft_noverlap_spinbox.setRange(0, 4096);
        self.stft_noverlap_spinbox.setValue(512)
        self.stft_nfft_spinbox.setRange(32, 65536);
        self.stft_nfft_spinbox.setValue(2048)
        self.partial_fft_start_spinbox.setRange(0, 10000);
        self.partial_fft_start_spinbox.setValue(4)
        self.partial_fft_end_spinbox.setRange(1, 10000);
        self.partial_fft_end_spinbox.setValue(8)
        d_layout.addRow("STFT Window:", self.stft_nperseg_spinbox)
        d_layout.addRow("STFT Overlap:", self.stft_noverlap_spinbox)
        d_layout.addRow("STFT NFFT:", self.stft_nfft_spinbox)
        d_layout.addRow("Partial FFT Start:", self.partial_fft_start_spinbox)
        d_layout.addRow("Partial FFT End:", self.partial_fft_end_spinbox)
        dsp_group.setLayout(d_layout)
        layout.addWidget(dsp_group)

        # Distance Calculation Setup
        distance_group = QGroupBox(" Distance Calculation Setup ")
        distance_group.setFont(QFont("Arial", 11, QFont.Bold))
        dist_layout = QFormLayout()
        self.distance_bandpass_low_spinbox.setRange(1, 100)
        self.distance_bandpass_low_spinbox.setValue(20)
        self.distance_bandpass_low_spinbox.setSuffix(" kHz")
        self.distance_bandpass_high_spinbox.setRange(1, 200)
        self.distance_bandpass_high_spinbox.setValue(60)
        self.distance_bandpass_high_spinbox.setSuffix(" kHz")
        self.distance_envelope_cutoff_spinbox.setRange(1, 50)
        self.distance_envelope_cutoff_spinbox.setValue(5)
        self.distance_envelope_cutoff_spinbox.setSuffix(" kHz")
        self.distance_filter_order_spinbox.setRange(1, 10)
        self.distance_filter_order_spinbox.setValue(4)
        dist_layout.addRow("Bandpass Low Cutoff:", self.distance_bandpass_low_spinbox)
        dist_layout.addRow("Bandpass High Cutoff:", self.distance_bandpass_high_spinbox)
        dist_layout.addRow("Envelope Lowpass Cutoff:", self.distance_envelope_cutoff_spinbox)
        dist_layout.addRow("Filter Order:", self.distance_filter_order_spinbox)
        distance_group.setLayout(dist_layout)
        layout.addWidget(distance_group)

        # Save Setup (Graph Toggle Added Here)
        save_group = QGroupBox(" File Save & Control ")
        save_group.setFont(QFont("Arial", 11, QFont.Bold))
        s_layout = QGridLayout()
        self.save_path_lineedit.setPlaceholderText("Select Save Folder...")
        self.save_data_button.setFixedHeight(30)

        s_layout.addWidget(self.save_path_lineedit, 0, 0)
        s_layout.addWidget(self.browse_button, 0, 1)
        s_layout.addWidget(self.save_data_button, 1, 0)
        s_layout.addWidget(self.btn_toggle_graph, 1, 1)  # [신규] 그래프 토글 버튼
        s_layout.addWidget(self.save_status_label, 2, 0, 1, 2)

        save_group.setLayout(s_layout)
        layout.addWidget(save_group)

        layout.addStretch(1)
        self.btn_refresh.setFixedHeight(45)
        self.btn_refresh.setStyleSheet(
            "QPushButton { background-color: #D6EAF8; border: 1px solid #5DADE2; border-radius: 5px; font-weight: bold; font-size: 12pt; } QPushButton:hover { background-color: #AED6F1; }")
        layout.addWidget(self.btn_refresh)

        scroll.setWidget(content)
        wrapper_layout = QVBoxLayout(widget)
        wrapper_layout.addWidget(scroll)
        return widget

    def _create_monitor_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        gps_group = QGroupBox(" Global GPS Information ")
        gps_group.setStyleSheet(
            "QGroupBox { font-weight: bold; color: darkblue; border: 1px solid gray; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }")
        g_layout = QGridLayout()
        font_gps = QFont("Arial", 10)
        self.lbl_gps_lat = QLabel("Lat: 0.000000")
        self.lbl_gps_lon = QLabel("Lon: 0.000000")
        self.lbl_gps_spd = QLabel("Speed: 0 km/h")
        self.lbl_gps_valid = QLabel("❌ Waiting...")
        for l in [self.lbl_gps_lat, self.lbl_gps_lon, self.lbl_gps_spd, self.lbl_gps_valid]: l.setFont(font_gps)
        g_layout.addWidget(self.lbl_gps_lat, 0, 0)
        g_layout.addWidget(self.lbl_gps_lon, 0, 1)
        g_layout.addWidget(self.lbl_gps_spd, 1, 0)
        g_layout.addWidget(self.lbl_gps_valid, 1, 1)
        gps_group.setLayout(g_layout)
        layout.addWidget(gps_group)

        self.output_tabs = QTabWidget()
        self.output_tabs.setStyleSheet("QTabBar::tab { width: 55px; height: 30px; font-weight: bold; }")
        self.sensor_output_widgets = []
        for i in range(1, 7):
            out_view = SensorOutputView(i)
            self.sensor_output_widgets.append(out_view)
            self.output_tabs.addTab(out_view, f"S{i}")
        layout.addWidget(self.output_tabs)
        return widget

    def update_all_widgets_from_vm(self):
        self.lbl_gps_lat.setText(f"Lat: {self.view_model.gps_latitude}")
        self.lbl_gps_lon.setText(f"Lon: {self.view_model.gps_longitude}")
        self.lbl_gps_spd.setText(f"Speed: {self.view_model.gps_speed_kmh}")
        self.lbl_gps_valid.setText(self.view_model.gps_validity)
        self.img_sp.setStyleSheet(f"background-color: {self.view_model.img_sp}; border-radius: 10px;")
        self.img_gps_sp.setStyleSheet(f"background-color: {self.view_model.gps_img_sp}; border-radius: 10px;")
        for i, s_data in enumerate(self.view_model.sensors):
            if i < len(self.sensor_output_widgets):
                view = self.sensor_output_widgets[i]
                view.mlx_label.setText(s_data['mlx_temp'])
                view.bme_temp.setText(s_data['bme_temp'])
                view.bme_hum.setText(s_data['bme_hum'])
                view.bme_press.setText(s_data['bme_press'])
                view.distance_label.setText(s_data.get('distance', 'N/A mm'))
                view.processing_time_label.setText(s_data.get('processing_time', '0.00 ms'))

    # [수정] sensor_id 인자 추가 및 해당 그래프만 업데이트
    @pyqtSlot(int, object, object, object, object, object, object, object, object, object, object, object)
    def update_chart_data(self, sensor_id, x_data, rx_y, tx_y, f_fft, m_fft, t_stft, z_stft, f_pfft, m_pfft, tof_us_marker=None, tx_start_us_marker=None):
        if sensor_id < 0 or sensor_id >= len(self.sensor_graphs):
            return
        target_graph = self.sensor_graphs[sensor_id]
        fs = 1000000.0  # 1MHz

        def auto_scale_y_symmetric(plot_widget, y_data):
            if y_data is not None and len(y_data) > 0 and np.any(y_data):
                max_abs = np.max(np.abs(y_data))
                y_limit = max_abs * 1.05
                plot_widget.setYRange(-y_limit, y_limit)
            else:
                plot_widget.setYRange(-3500, 3500)

        def auto_scale_y_positive(plot_widget, y_data):
            if y_data is not None and len(y_data) > 0 and np.any(y_data):
                max_val = np.max(y_data)
                y_limit = max_val * 1.05
                plot_widget.setYRange(0, y_limit)
            else:
                plot_widget.setYRange(0, 1.0)

        # 1. TX Signal
        if tx_y is not None and len(tx_y) > 0:
            tx_mv = tx_y * 1000.0
            time_axis_tx = np.arange(len(tx_y)) / fs * 1000.0
            target_graph.tx_plot['curve'].setData(time_axis_tx, tx_mv)
            auto_scale_y_symmetric(target_graph.tx_plot['widget'], tx_mv)
            
            # Update TX start marker
            if tx_start_us_marker is not None and tx_start_us_marker != "Not Detected":
                tx_start_ms = tx_start_us_marker / 1000.0
                target_graph.tx_plot['marker'].setValue(tx_start_ms)
                target_graph.tx_plot['marker'].setVisible(True)
            else:
                target_graph.tx_plot['marker'].setVisible(False)
        else:
            # Hide marker when no TX data
            target_graph.tx_plot['marker'].setVisible(False)

        # 2. RX Signal
        if rx_y is not None and len(rx_y) > 0:
            rx_mv = rx_y * 1000.0
            time_axis_rx = np.arange(len(rx_y)) / fs * 1000.0
            target_graph.rx_plot['curve'].setData(time_axis_rx, rx_mv)
            auto_scale_y_symmetric(target_graph.rx_plot['widget'], rx_mv)
            
            # Update RX start (ToF) marker
            if tof_us_marker is not None and tof_us_marker != "Not Detected":
                tof_ms = tof_us_marker / 1000.0
                target_graph.rx_plot['marker'].setValue(tof_ms)
                target_graph.rx_plot['marker'].setVisible(True)
            else:
                target_graph.rx_plot['marker'].setVisible(False)
        else:
            # Hide marker when no RX data
            target_graph.rx_plot['marker'].setVisible(False)

        # 3. FFT
        if f_fft is not None and m_fft is not None:
            if len(f_fft) == len(m_fft):
                target_graph.fft_plot['curve'].setData(f_fft, m_fft)
                auto_scale_y_positive(target_graph.fft_plot['widget'], m_fft)

        # 4. Partial FFT
        if f_pfft is not None and m_pfft is not None:
            if len(f_pfft) == len(m_pfft):
                target_graph.p_fft_plot['curve'].setData(f_pfft, m_pfft)
                target_graph.p_fft_plot['widget'].autoRange()

        # 5. STFT
        if z_stft is not None:
            z_abs = np.abs(z_stft)
            target_graph.stft_widget['img'].setImage(z_abs.T)
            if t_stft is not None and len(t_stft) > 1:
                target_graph.stft_widget['img'].setRect(t_stft[0], 35000, t_stft[-1] - t_stft[0], 10000)

            min_val = np.min(z_abs)
            max_val = np.max(z_abs)
            target_graph.stft_widget['hist'].setLevels(min_val, max_val)