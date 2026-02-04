# -*- coding: utf-8 -*-
import numpy as np
import pyqtgraph as pg

try:
    from PyQt5.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGroupBox, QPushButton, QLabel, QFormLayout, QGridLayout,
        QComboBox, QTabWidget, QCheckBox, QSplitter, QSlider,
        QDoubleSpinBox, QStackedWidget, QSpinBox
    )
    from PyQt5.QtGui import QFont
    from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal, QRectF
except ImportError:
    from PySide6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGroupBox, QPushButton, QLabel, QFormLayout, QGridLayout,
        QComboBox, QTabWidget, QCheckBox, QSplitter, QSlider,
        QDoubleSpinBox, QStackedWidget, QSpinBox
    )
    from PySide6.QtGui import QFont
    from PySide6.QtCore import Qt, Slot as pyqtSlot, Signal as pyqtSignal, QRectF

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOption('antialias', True)


# --- Sensor Graph View ---
class SensorGraphView(QWidget):
    def __init__(self, sensor_id, parent=None):
        super().__init__(parent)
        self.sensor_id = sensor_id
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)

        self.tx_plot = self._create_plot("TX Signal (Time)", 'Voltage (mV)', 'Time (ms)', 'b')
        self.tx_plot['widget'].enableAutoRange(axis='y')
        # [EXPERIMENTAL] TX Start marker
        self.tx_start_line = pg.InfiniteLine(angle=90, movable=False, 
                                                 pen=pg.mkPen('orange', width=2, style=Qt.DashLine),
                                                 label='TX Start', labelOpts={'position': 0.95})
        self.tx_plot['widget'].addItem(self.tx_start_line)
        self.tx_start_line.setVisible(False)
        self.layout.addWidget(self.tx_plot['group'], 15)

        self.rx_plot = self._create_plot("RX Signal (Time)", 'Voltage (mV)', 'Time (ms)', 'r')
        self.rx_plot['widget'].enableAutoRange(axis='y')
        # [EXPERIMENTAL] RX ToF marker
        self.rx_tof_line = pg.InfiniteLine(angle=90, movable=False,
                                           pen=pg.mkPen('green', width=2, style=Qt.DashLine),
                                           label='ToF Start', labelOpts={'position': 0.95})
        self.rx_plot['widget'].addItem(self.rx_tof_line)
        self.rx_tof_line.setVisible(False)
        self.layout.addWidget(self.rx_plot['group'], 15)

        self.stft_widget = self._create_stft_group()
        self.layout.addWidget(self.stft_widget['group'], 30)

        self.fft_plot = self._create_plot("Full FFT", 'Magnitude', 'Freq (Hz)', 'g')
        # [수정] 초기 X축 범위를 전체로 보기 위해 주석 처리하거나 넓게 설정
        # self.fft_plot['widget'].setXRange(0, 100000)
        self.layout.addWidget(self.fft_plot['group'], 15)

        self.p_fft_plot = self._create_plot("Partial FFT", 'Magnitude', 'Freq (Hz)', 'g')
        self.layout.addWidget(self.p_fft_plot['group'], 15)

    def _create_plot(self, title, left_label, bottom_label, color):
        group = QGroupBox(f" {title} ")
        group.setFont(QFont("Tw Cen MT Condensed Extra Bold", 10, QFont.Bold))
        plot_widget = pg.PlotWidget()
        plot_widget.setBackground('w')
        plot_widget.showGrid(x=True, y=True, alpha=0.3)
        plot_widget.setLabel('left', left_label)
        plot_widget.setLabel('bottom', bottom_label)
        curve = plot_widget.plot(pen=pg.mkPen(color, width=1.5), name='Signal')
        layout = QVBoxLayout()
        layout.addWidget(plot_widget)
        group.setLayout(layout)
        return {'group': group, 'widget': plot_widget, 'curve': curve}

    def _create_stft_group(self):
        group = QGroupBox(" Spectrogram (STFT) ")
        group.setFont(QFont("Tw Cen MT Condensed Extra Bold", 10, QFont.Bold))
        layout = QHBoxLayout()
        stft_plot = pg.PlotWidget()
        stft_plot.setLabel('bottom', 'Time (s)')
        stft_plot.setLabel('left', 'Frequency (Hz)')
        stft_plot.setAspectLocked(False)
        stft_plot.getPlotItem().showGrid(False, False)

        hist = pg.HistogramLUTWidget()
        hist.setMinimumWidth(60)
        img_item = pg.ImageItem()
        hist.setImageItem(img_item)
        hist.gradient.setColorMap(pg.colormap.get('plasma'))
        stft_plot.addItem(img_item)

        layout.addWidget(stft_plot, 1)
        layout.addWidget(hist)
        group.setLayout(layout)
        return {'group': group, 'widget': stft_plot, 'img': img_item, 'hist': hist}


# --- Trend Analysis View ---
class TrendAnalysisView(QWidget):
    plot_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)

        top_layout = QHBoxLayout()
        self.filter_group = QGroupBox(" Visible Sensors ")
        self.filter_group.setFixedHeight(60)
        filter_layout = QHBoxLayout()

        self.sensor_checks = []
        self.colors = ['r', 'g', 'b', 'c', 'm', '#DAA520']
        for i in range(6):
            chk = QCheckBox(f"Sensor {i + 1}")
            chk.setChecked(True)
            chk.setStyleSheet(f"color: {self._qt_color_name(self.colors[i])}; font-weight: bold;")
            chk.toggled.connect(self.update_visibility)
            self.sensor_checks.append(chk)
            filter_layout.addWidget(chk)
        self.filter_group.setLayout(filter_layout)
        top_layout.addWidget(self.filter_group)

        self.btn_go_detail = QPushButton("🔍 Go to Detailed View")
        self.btn_go_detail.setFixedHeight(50)
        top_layout.addWidget(self.btn_go_detail)
        self.main_layout.addLayout(top_layout)

        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.main_layout.addWidget(self.plot_container)
        self.active_plots = {}

    def _qt_color_name(self, code):
        mapping = {'r': 'red', 'g': 'green', 'b': 'blue', 'c': 'cyan', 'm': 'magenta', '#DAA520': '#DAA520'}
        return mapping.get(code, 'black')

    def setup_plots(self, selected_options):
        for i in reversed(range(self.plot_layout.count())):
            self.plot_layout.itemAt(i).widget().setParent(None)
        self.active_plots = {}
        prev_plot = None

        for opt in selected_options:
            group = QGroupBox()
            layout = QVBoxLayout()
            plot = pg.PlotWidget()
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.scene().sigMouseClicked.connect(self.on_mouse_clicked)
            curves = []

            if opt == 'speed':
                group.setTitle(" Vehicle Speed (GPS) ")
                curve = plot.plot(pen=pg.mkPen('k', width=2), name="Speed")
                curves.append(curve)
            elif opt == 'fft_peak':
                group.setTitle(" Full FFT Peak Frequency ")
                for i in range(6):
                    c = plot.plot(pen=pg.mkPen(self.colors[i], width=1.5))
                    curves.append(c)
            elif opt == 'pfft_peak':
                group.setTitle(" Partial FFT Peak Frequency ")
                for i in range(6):
                    c = plot.plot(pen=pg.mkPen(self.colors[i], width=1.5))
                    curves.append(c)
            elif opt == 'pfft_mag':
                group.setTitle(" Partial FFT Magnitude ")
                for i in range(6):
                    c = plot.plot(pen=pg.mkPen(self.colors[i], width=1.5))
                    curves.append(c)

            line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('gray', width=1, style=Qt.DashLine))
            plot.addItem(line)
            if prev_plot: plot.setXLink(prev_plot)
            prev_plot = plot
            layout.addWidget(plot)
            group.setLayout(layout)
            self.plot_layout.addWidget(group)
            self.active_plots[opt] = {'widget': plot, 'curves': curves, 'line': line}

    def update_data(self, results):
        if 'speed' in self.active_plots and results:
            ref_id = list(results.keys())[0]
            spd = results[ref_id].get('speed')
            if spd is not None:
                self.active_plots['speed']['curves'][0].setData(np.arange(len(spd)), spd)

        for opt in ['fft_peak', 'pfft_peak', 'pfft_mag']:
            if opt in self.active_plots:
                curves = self.active_plots[opt]['curves']
                for sid, data in results.items():
                    if sid < len(curves):
                        val = data.get(opt)
                        if val is not None: curves[sid].setData(val)
        self.update_visibility()

    def update_visibility(self):
        for i, chk in enumerate(self.sensor_checks):
            visible = chk.isChecked()
            for opt in self.active_plots:
                if opt == 'speed': continue
                curves = self.active_plots[opt]['curves']
                if i < len(curves): curves[i].setVisible(visible)

    def update_indicator(self, index):
        for opt in self.active_plots:
            self.active_plots[opt]['line'].setValue(index)

    def on_mouse_clicked(self, event):
        if event.button() == Qt.LeftButton:
            for opt, data in self.active_plots.items():
                plot = data['widget']
                if plot.plotItem.sceneBoundingRect().contains(event.scenePos()):
                    mouse_point = plot.plotItem.vb.mapSceneToView(event.scenePos())
                    self.plot_clicked.emit(int(round(mouse_point.x())))
                    return


# --- Sensor Info View ---
class SensorInfoView(QWidget):
    def __init__(self, sensor_id, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        val_font = QFont("Arial", 12, QFont.Bold)
        env_group = QGroupBox(f" Environment Info ")
        env_layout = QFormLayout()
        self.lbl_obj_temp = QLabel("-")
        self.lbl_amb_temp = QLabel("-")
        self.lbl_humi = QLabel("-")
        self.lbl_pres = QLabel("-")
        for l in [self.lbl_obj_temp, self.lbl_amb_temp, self.lbl_humi, self.lbl_pres]:
            l.setFont(val_font)
            l.setStyleSheet("color: darkblue;")
        env_layout.addRow("Object Temp:", self.lbl_obj_temp)
        env_layout.addRow("Ambient Temp:", self.lbl_amb_temp)
        env_layout.addRow("Humidity:", self.lbl_humi)
        env_layout.addRow("Pressure:", self.lbl_pres)
        env_group.setLayout(env_layout)
        layout.addWidget(env_group)
        
        # Distance GroupBox
        dist_group = QGroupBox(f" Distance ")
        dist_layout = QFormLayout()
        self.lbl_distance = QLabel("-")
        self.lbl_tof = QLabel("-")
        self.lbl_processing_time = QLabel("-")
        for l in [self.lbl_distance, self.lbl_tof, self.lbl_processing_time]:
            l.setFont(val_font)
            l.setStyleSheet("color: darkblue;")
        dist_layout.addRow("Distance:", self.lbl_distance)
        dist_layout.addRow("ToF:", self.lbl_tof)
        dist_layout.addRow("Processing Time:", self.lbl_processing_time)
        dist_group.setLayout(dist_layout)
        layout.addWidget(dist_group)
        layout.addStretch(1)


# --- MainWindow ---
class MainWindow(QMainWindow):
    def __init__(self, view_model, parent=None):
        super().__init__(parent)
        self.view_model = view_model
        self.setWindowTitle("Multi-Sensor Data Player (Offline Viewer)")
        self.resize(1600, 950)

        self.sensor_select_buttons = []
        self.slider_seek = QSlider(Qt.Horizontal)
        self.lbl_current_file = QLabel("No File Loaded")
        self.lbl_index = QLabel("0 / 0")
        self.btn_prev = QPushButton("◀")
        self.btn_play = QPushButton("▶ Play")
        # [수정] Play 버튼 체크 가능하도록 설정 (Toggle 기능)
        self.btn_play.setCheckable(True)
        self.btn_next = QPushButton("▶")
        self.combo_speed = QComboBox()

        # BPF Controls
        self.chk_bpf_apply = QCheckBox("Apply BPF")
        self.spin_bpf_low = QDoubleSpinBox();
        self.spin_bpf_low.setRange(0, 500000);
        self.spin_bpf_low.setValue(35000)
        self.spin_bpf_high = QDoubleSpinBox();
        self.spin_bpf_high.setRange(0, 500000);
        self.spin_bpf_high.setValue(45000)

        # Trend Analysis
        self.pfft_start_spin = QDoubleSpinBox()
        self.pfft_end_spin = QDoubleSpinBox()
        self.chk_trend_speed = QCheckBox("Speed")
        self.chk_trend_fft = QCheckBox("Full FFT Peak")
        self.chk_trend_pfft_peak = QCheckBox("Partial FFT Peak")
        self.chk_trend_pfft_mag = QCheckBox("Partial FFT Mag")
        self.btn_analyze = QPushButton("📊 Analyze Trend")

        # STFT Parameters
        self.combo_stft_window = QComboBox()
        self.spin_stft_overlap = QSpinBox()
        self.combo_stft_padding = QComboBox()

        self.meta_sensor_buttons = []
        self.stacked_meta_info = QStackedWidget()

        self.lbl_log_time = QLabel("-")
        self.lbl_gps_lat = QLabel("-")
        self.lbl_gps_lon = QLabel("-")
        self.lbl_gps_spd = QLabel("-")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Horizontal)

        left_panel = QWidget()
        left_panel.setMinimumWidth(450)
        left_layout = QVBoxLayout(left_panel)

        self.left_tabs = QTabWidget()
        self.control_tab = self._create_control_tab()
        self.left_tabs.addTab(self.control_tab, "🎮 CONTROL")
        self.info_tab = self._create_info_tab()
        self.left_tabs.addTab(self.info_tab, "📝 METADATA")
        left_layout.addWidget(self.left_tabs)
        left_layout.addWidget(self._create_playback_group())
        splitter.addWidget(left_panel)

        self.right_main_tabs = QTabWidget()
        self.detailed_view_widget = QWidget()
        detailed_layout = QVBoxLayout(self.detailed_view_widget)
        self.graph_tabs = QTabWidget()
        self.sensor_graphs = []
        for i in range(1, 7):
            s_view = SensorGraphView(i)
            self.sensor_graphs.append(s_view)
            self.graph_tabs.addTab(s_view, f"Sensor {i}")
        detailed_layout.addWidget(self.graph_tabs)
        self.trend_view = TrendAnalysisView()
        self.right_main_tabs.addTab(self.detailed_view_widget, "🔍 Detailed View")
        self.right_main_tabs.addTab(self.trend_view, "📈 Trend Analysis")

        splitter.addWidget(self.right_main_tabs)
        splitter.setStretchFactor(0, 30)
        splitter.setStretchFactor(1, 70)
        main_layout.addWidget(splitter)

    def _create_control_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 1. Sensor Selection
        src_group = QGroupBox(" Sensor Folder Selection ")
        grid = QGridLayout()
        for i in range(6):
            btn = QPushButton(f"Set S{i + 1}\n(Empty)")
            btn.setFixedSize(80, 50)
            self.sensor_select_buttons.append(btn)
            grid.addWidget(btn, i // 3, i % 3)
        src_group.setLayout(grid)
        layout.addWidget(src_group)

        # 2. [NEW] Signal Pre-processing (BPF)
        bpf_group = QGroupBox(" Signal Pre-processing (BPF) ")
        bpf_layout = QGridLayout()

        self.chk_bpf_apply.setStyleSheet("font-weight: bold; color: blue;")
        bpf_layout.addWidget(self.chk_bpf_apply, 0, 0, 1, 2)

        bpf_layout.addWidget(QLabel("Low Cut (Hz):"), 1, 0)
        self.spin_bpf_low.setSingleStep(1000)
        bpf_layout.addWidget(self.spin_bpf_low, 1, 1)

        bpf_layout.addWidget(QLabel("High Cut (Hz):"), 2, 0)
        self.spin_bpf_high.setSingleStep(1000)
        bpf_layout.addWidget(self.spin_bpf_high, 2, 1)

        bpf_group.setLayout(bpf_layout)
        layout.addWidget(bpf_group)

        # 3. STFT Parameters Setup
        stft_group = QGroupBox(" STFT Parameters (Detail View) ")
        stft_layout = QGridLayout()

        # Window Size (Nperseg)
        stft_layout.addWidget(QLabel("Window Size:"), 0, 0)
        self.combo_stft_window.addItems(["32", "64", "128", "256", "512", "1024", "2048", "4096", "8192"])
        self.combo_stft_window.setCurrentText("256")
        stft_layout.addWidget(self.combo_stft_window, 0, 1)

        # Overlap Ratio
        stft_layout.addWidget(QLabel("Overlap (%):"), 0, 2)
        self.spin_stft_overlap.setRange(0, 99)
        self.spin_stft_overlap.setValue(50)
        self.spin_stft_overlap.setSuffix("%")
        stft_layout.addWidget(self.spin_stft_overlap, 0, 3)

        # Zero Padding (NFFT Factor)
        stft_layout.addWidget(QLabel("Zero Padding:"), 1, 0)
        self.combo_stft_padding.addItems(["1x (None)", "2x (Smooth)", "4x (Very Smooth)", "8x", "16x", "32x"])
        stft_layout.addWidget(self.combo_stft_padding, 1, 1)

        stft_group.setLayout(stft_layout)
        layout.addWidget(stft_group)

        # 4. Trend Analysis Setup
        anl_group = QGroupBox(" Trend Analysis Setup ")
        anl_layout = QVBoxLayout()
        grid_param = QGridLayout()
        grid_param.addWidget(QLabel("Partial Start(ms):"), 0, 0)
        self.pfft_start_spin.setRange(0, 1000);
        self.pfft_start_spin.setValue(3.0);
        self.pfft_start_spin.setSingleStep(0.1)
        grid_param.addWidget(self.pfft_start_spin, 0, 1)
        grid_param.addWidget(QLabel("Partial End(ms):"), 1, 0)
        self.pfft_end_spin.setRange(0, 1000);
        self.pfft_end_spin.setValue(4.0);
        self.pfft_end_spin.setSingleStep(0.1)
        grid_param.addWidget(self.pfft_end_spin, 1, 1)
        anl_layout.addLayout(grid_param)

        self.chk_trend_speed.setChecked(True)
        self.chk_trend_fft.setChecked(True)
        self.chk_trend_pfft_peak.setChecked(True)
        anl_layout.addWidget(self.chk_trend_speed)
        anl_layout.addWidget(self.chk_trend_fft)
        anl_layout.addWidget(self.chk_trend_pfft_peak)
        anl_layout.addWidget(self.chk_trend_pfft_mag)
        anl_layout.addWidget(self.btn_analyze)
        anl_group.setLayout(anl_layout)
        layout.addWidget(anl_group)
        
        # Reset Button
        self.btn_reset = QPushButton("🔄 Reset All")
        self.btn_reset.setStyleSheet("background-color: #E74C3C; color: white; font-weight: bold; padding: 10px;")
        layout.addWidget(self.btn_reset)
        
        layout.addStretch()
        return widget

    def _create_playback_group(self):
        group = QGroupBox(" Playback ")
        layout = QVBoxLayout()
        layout.addWidget(self.lbl_current_file)
        h_slider = QHBoxLayout()
        h_slider.addWidget(self.slider_seek)
        h_slider.addWidget(self.lbl_index)
        layout.addLayout(h_slider)
        h_btns = QHBoxLayout()
        h_btns.addWidget(self.btn_prev)
        h_btns.addWidget(self.btn_play)
        h_btns.addWidget(self.btn_next)
        layout.addLayout(h_btns)
        h_spd = QHBoxLayout()
        self.combo_speed.addItems(["1x", "2x", "5x", "10x", "Max"])
        h_spd.addWidget(QLabel("Speed:"))
        h_spd.addWidget(self.combo_speed)
        layout.addLayout(h_spd)
        group.setLayout(layout)
        return group

    def _create_info_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        gps_group = QGroupBox(" GPS ")
        g_layout = QGridLayout()
        g_layout.addWidget(QLabel("Time:"), 0, 0);
        g_layout.addWidget(self.lbl_log_time, 0, 1)
        g_layout.addWidget(QLabel("Lat:"), 1, 0);
        g_layout.addWidget(self.lbl_gps_lat, 1, 1)
        g_layout.addWidget(QLabel("Lon:"), 2, 0);
        g_layout.addWidget(self.lbl_gps_lon, 2, 1)
        g_layout.addWidget(QLabel("Spd:"), 3, 0);
        g_layout.addWidget(self.lbl_gps_spd, 3, 1)
        gps_group.setLayout(g_layout)
        layout.addWidget(gps_group)

        btn_group = QGroupBox(" Sensor Info ")
        btn_layout = QGridLayout()
        for i in range(6):
            btn = QPushButton(f"S{i + 1}")
            btn.setCheckable(True)
            if i == 0: btn.setChecked(True)
            self.meta_sensor_buttons.append(btn)
            btn_layout.addWidget(btn, i // 2, i % 2)
        btn_group.setLayout(btn_layout)
        layout.addWidget(btn_group)

        for i in range(6): self.stacked_meta_info.addWidget(SensorInfoView(i + 1))
        layout.addWidget(self.stacked_meta_info)
        layout.addStretch()
        return widget

    def set_sensor_ready(self, sid, ready):
        btn = self.sensor_select_buttons[sid]
        btn.setText(f"Set S{sid + 1}\n({'Ready' if ready else 'Empty'})")
        btn.setStyleSheet(f"background-color: {'#2ECC71' if ready else '#EEE'};")

    # [수정됨] f_stft를 맨 뒤로 보내고 Optional(=None)로 처리하여 인자 개수 에러 해결
    # [EXPERIMENTAL] Added tof_us and tx_start_us for visualization
    @pyqtSlot(int, object, object, object, object, object, object, object, object, object, object, object, object)
    def update_graph(self, sensor_id, x_data, rx_y, tx_y, f_fft, m_fft, t_stft, z_stft, f_pfft, m_pfft, f_stft=None, tof_us=None, tx_start_us=None):
        if sensor_id < 0 or sensor_id >= len(self.sensor_graphs): return
        target = self.sensor_graphs[sensor_id]
        fs = 1000000.0

        if tx_y is not None:
            t_tx = np.arange(len(tx_y)) / fs * 1000.0
            target.tx_plot['curve'].setData(t_tx, tx_y)
            # [EXPERIMENTAL] Update TX start marker
            if tx_start_us is not None and tx_start_us != "Not Detected":
                start_ms = tx_start_us / 1000.0  # Convert μs to ms
                target.tx_start_line.setValue(start_ms)
                target.tx_start_line.setVisible(True)
            else:
                target.tx_start_line.setVisible(False)
        if rx_y is not None:
            t_rx = np.arange(len(rx_y)) / fs * 1000.0
            target.rx_plot['curve'].setData(t_rx, rx_y)
            # [EXPERIMENTAL] Update RX ToF marker
            if tof_us is not None and tof_us != "Not Detected":
                tof_ms = tof_us / 1000.0  # Convert μs to ms
                target.rx_tof_line.setValue(tof_ms)
                target.rx_tof_line.setVisible(True)
            else:
                target.rx_tof_line.setVisible(False)
        if f_fft is not None: target.fft_plot['curve'].setData(f_fft, m_fft)
        if f_pfft is not None: target.p_fft_plot['curve'].setData(f_pfft, m_pfft)

        # [NEW] BPF 활성화 시 주파수 축 스케일 자동 조정
        if self.chk_bpf_apply.isChecked():
            low = self.spin_bpf_low.value()
            high = self.spin_bpf_high.value()
            if high > low:
                target.fft_plot['widget'].setXRange(low, high, padding=0)
                # Partial FFT도 동일하게 적용 (선택 사항이나 일관성을 위해)
                target.p_fft_plot['widget'].setXRange(low, high, padding=0)
        else:
            target.fft_plot['widget'].enableAutoRange(axis='x')
            target.p_fft_plot['widget'].enableAutoRange(axis='x')

        # STFT Logic
        if z_stft is not None and t_stft is not None:
            # 데이터 유효성 체크
            if z_stft.size == 0 or len(t_stft) < 2:
                target.stft_widget['img'].clear()
                return

            z_abs = np.abs(z_stft)

            # 만약 f_stft가 None이라면(인자 전달 안됨), 역산하여 생성
            if f_stft is None or len(f_stft) < 2:
                # STFT freq bins = z_stft.shape[0]
                # 범위: 0 ~ FS/2
                n_freq = z_stft.shape[0]
                f_stft = np.linspace(0, fs / 2.0, n_freq)

            # 1. 이미지 데이터 설정 (Transposed: Time=X, Freq=Y)
            target.stft_widget['img'].setImage(z_abs.T, autoLevels=False)
            target.stft_widget['hist'].setLevels(np.min(z_abs), np.max(z_abs))

            # 2. 물리적 좌표 계산
            x_start = t_stft[0]
            y_start = f_stft[0]
            width = t_stft[-1] - t_stft[0]
            height = f_stft[-1] - f_stft[0]

            if width <= 0: width = 0.001
            if height <= 0: height = 1000.0

            # 3. 좌표 매핑
            rect = QRectF(x_start, y_start, width, height)
            target.stft_widget['img'].setRect(rect)

            # 4. 뷰 범위 강제 설정 (그래프가 그려졌는지 확인하기 위해)
            plot_item = target.stft_widget['widget'].getPlotItem()
            plot_item.setXRange(x_start, x_start + width, padding=0)

            # Y축(주파수)은 BPF 여부에 따라 조정
            if self.chk_bpf_apply.isChecked():
                low = self.spin_bpf_low.value()
                high = self.spin_bpf_high.value()
                if high > low:
                    plot_item.setYRange(low, high, padding=0)
                else:
                    plot_item.setYRange(y_start, y_start + height, padding=0)
            else:
                plot_item.setYRange(y_start, y_start + height, padding=0)