# -*- coding: utf-8 -*-
import os
import sys
import datetime
import json
import struct
import serial.tools.list_ports
import numpy as np
import time
import concurrent.futures  # [추가] 비동기 저장을 위한 라이브러리

# [필수] FFT/STFT 계산을 위한 SciPy
try:
    from scipy import signal
except ImportError:
    raise ImportError("SciPy 라이브러리가 필요합니다. 'pip install scipy'를 실행하세요.")

# --- [!] Qt 플랫폼 플러그인 설정 ---
try:
    import PyQt5

    plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
    os.environ['QT_PLUGIN_PATH'] = plugin_path
except ImportError:
    try:
        import PySide6

        plugin_path = os.path.join(os.path.dirname(PySide6.__file__), "plugins")
        os.environ['QT_PLUGIN_PATH'] = plugin_path
    except ImportError:
        pass

# --- Qt 라이브러리 임포트 ---
try:
    from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
    from PyQt5.QtCore import QThread, pyqtSlot, QTimer, QObject, pyqtSignal
except ImportError:
    try:
        from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
        from PySide6.QtCore import QThread, QTimer, QObject, Signal as pyqtSignal
        from PySide6.QtCore import Slot as pyqtSlot
    except ImportError:
        raise ImportError("PyQt5 또는 PySide6가 필요합니다.")

# --- 사용자 정의 모듈 임포트 ---
from MainWindow import MainWindow
from ViewModel import ViewModel
from Protocol import Protocol, e_cmds
from CommWorker import CommWorker
from GpsWorker import GpsWorker
from DistanceCalculator import DistanceCalculator, DistanceKalmanFilter, hampel_filter

# 샘플링 주파수
FS = 1000000.0


class ApplicationController(QObject):
    # [신규] 데이터 변경 시그널
    data_size_changed = pyqtSignal(int)
    interval_changed = pyqtSignal(int)  # Interval 변경 시그널 추가

    def __init__(self, app):
        super().__init__()
        self.app = app

        # [신규] 파일 저장을 위한 스레드 풀 생성 (최대 4개 스레드로 병렬 저장)
        self.save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        # [신규] UI 갱신 스로틀링을 위한 변수
        self.last_ui_update_time = {}  # key: sensor_id, value: timestamp
        self.ui_update_interval = 0.05  # 50ms

        # [신규] 그래프 활성화 플래그
        self.is_graph_enabled = True

        # 1. 모델 및 뷰 초기화
        self.view_model = ViewModel()

        # [호환성 처리]
        if not hasattr(self.view_model, 'sensors'):
            self.view_model.sensors = []
            for i in range(6):
                self.view_model.sensors.append({
                    'id': i,
                    'bme_temp': "0.00 °C", 'bme_hum': "0.00 %",
                    'bme_press': "0.00 hPa", 'mlx_temp': "0.00 °C",
                    'distance': "N/A mm", 'distance_raw': None,
                    'processing_time': "0.00 ms"
                })

        # Protocol은 각 센서마다 별도로 생성해야 하므로 여기서 하나만 만들지 않음
        self.main_window = MainWindow(view_model=self.view_model)

        # 2. 센서 통신 관리 (Dictionary 사용)
        self.acoustic_threads = {}  # key: sensor_id, value: QThread
        self.acoustic_workers = {}  # key: sensor_id, value: CommWorker
        self.is_connected = False  # 전체 연결 상태 (하나라도 연결되면 True)

        # 3. GPS 통신(GpsWorker) 스레드 설정
        self.gps_comm_thread = QThread()
        self.gps_comm_worker = GpsWorker()
        self.gps_comm_worker.moveToThread(self.gps_comm_thread)
        self.is_gps_connected = False

        # GPS Worker 시그널 연결
        self.gps_comm_worker.gps_data_received.connect(self.process_gps_data)
        self.gps_comm_worker.connection_status.connect(self.on_gps_connection_status)
        self.gps_comm_worker.log_message.connect(lambda msg: self.log(f"[GPS]: {msg}"))
        self.gps_comm_worker.finished.connect(self.on_gps_worker_finished)
        self.gps_comm_thread.started.connect(self.gps_comm_worker.run)

        # 4. 데이터 저장 및 처리 변수
        # 각 센서별 마지막 데이터를 저장해야 함 (Dictionary로 변경)
        self.last_rx_data = {}  # key: sensor_id
        self.last_tx_data = {}
        self.last_bme_data = {}  # key: sensor_id
        self.last_gps_data = {}
        self.is_logging = False
        self.save_base_dir = ""
        
        # Distance calculation components
        self.distance_calculators = {}  # key: sensor_id -> DistanceCalculator
        self.kalman_filters = {}  # key: sensor_id -> DistanceKalmanFilter
        self.last_distance = {}  # key: sensor_id -> distance in mm
        self.last_processing_time = {}  # key: sensor_id -> processing time in ms
        self.distance_history = {}  # key: sensor_id -> list of recent distances for smoothing
        self.last_tof_us = {}  # key: sensor_id -> ToF in microseconds
        self.last_tx_start_us = {}  # key: sensor_id -> TX start in microseconds
        self.miss_streak = {}  # key: sensor_id -> consecutive detection misses

        # 5. GUI 업데이트 타이머 (200ms)
        self.gui_timer = QTimer()
        self.gui_timer.setInterval(200)

        # 초기화 실행
        self.populate_serial_ports()
        self._connect_signals()
        self.gui_timer.start()

    def _connect_signals(self):
        """GUI 시그널과 슬롯 연결"""

        self.main_window.btn_serialport.clicked.connect(lambda checked: self.toggle_serial_connection(checked))
        self.main_window.btn_gps_serialport.clicked.connect(lambda checked: self.toggle_gps_connection(checked))

        self.main_window.browse_button.clicked.connect(lambda: self.select_save_path())
        self.main_window.save_data_button.clicked.connect(lambda checked: self.toggle_continuous_save(checked))

        # [수정] 데이터 길이 & 인터벌 변경 시그널 연결
        if hasattr(self.main_window, 'data_len_spinbox'):
            self.main_window.data_len_spinbox.valueChanged.connect(self.data_size_changed.emit)

        if hasattr(self.main_window, 'interval_spinbox'):
            self.main_window.interval_spinbox.valueChanged.connect(self.interval_changed.emit)

        # [신규] 그래프 토글 버튼 연결
        if hasattr(self.main_window, 'btn_toggle_graph'):
            self.main_window.btn_toggle_graph.clicked.connect(self.toggle_graph)

        # System Reset 버튼
        if hasattr(self.main_window, 'btn_refresh'):
            self.main_window.btn_refresh.clicked.connect(self.reset_system)

        self.gui_timer.timeout.connect(self.update_gui_data)

    @pyqtSlot(bool)
    def toggle_graph(self, checked):
        """그래프 생성 켜기/끄기"""
        self.is_graph_enabled = checked
        if checked:
            self.main_window.btn_toggle_graph.setText("GRAPH: ON")
            self.main_window.btn_toggle_graph.setStyleSheet(
                "color: white; background-color: green; border: 1px solid green; font-weight: bold;")
        else:
            self.main_window.btn_toggle_graph.setText("GRAPH: OFF")
            self.main_window.btn_toggle_graph.setStyleSheet(
                "color: black; background-color: #EEE; border: 1px solid gray; font-weight: bold;")

    @pyqtSlot()
    def reset_system(self):
        """[신규] 전체 시스템 리셋"""
        self.log("!!! SYSTEM RESET 요청 !!!")

        # 1. 모든 센서 연결 종료
        self.toggle_serial_connection(False)
        self.main_window.btn_serialport.setChecked(False)
        self.view_model.img_sp = "red"

        # 2. GPS 종료
        if self.is_gps_connected:
            self.gps_comm_worker.stop()
            self.main_window.btn_gps_serialport.setChecked(False)
            self.view_model.gps_img_sp = "red"
            if self.gps_comm_thread.isRunning():
                self.gps_comm_thread.quit()
                self.gps_comm_thread.wait(1000)
            # GPS 스레드 재시작을 위해 다시 생성
            self.gps_comm_thread = QThread()
            self.gps_comm_worker = GpsWorker()
            self.gps_comm_worker.moveToThread(self.gps_comm_thread)
            self.gps_comm_worker.gps_data_received.connect(self.process_gps_data)
            self.gps_comm_worker.connection_status.connect(self.on_gps_connection_status)
            self.gps_comm_worker.log_message.connect(lambda msg: self.log(f"[GPS]: {msg}"))
            self.gps_comm_worker.finished.connect(self.on_gps_worker_finished)
            self.gps_comm_thread.started.connect(self.gps_comm_worker.run)

        # 4. 포트 목록 갱신
        self.populate_serial_ports()

        self.log("!!! SYSTEM RESET 완료 !!!")
        QMessageBox.information(self.main_window, "System Reset", "시스템이 초기화되었습니다.")

    def log(self, message: str):
        print(f"[CTRL]: {message}")

    def cleanup(self):
        """앱 종료 시 자원 정리"""
        self.log("종료 중...")
        # [신규] 스레드 풀 정리
        self.save_executor.shutdown(wait=False)

        self.gui_timer.stop()
        self.toggle_serial_connection(False)

        if self.is_gps_connected:
            self.gps_comm_worker.stop()
            if self.gps_comm_thread.isRunning():
                self.gps_comm_thread.quit()
                self.gps_comm_thread.wait(1000)

    # ==========================================================
    #  Port Management
    # ==========================================================
    def populate_serial_ports(self):
        ports = serial.tools.list_ports.comports()
        port_list = [p.device for p in ports] if ports else ["No ports"]

        print(f"Ports refreshed: {port_list}")

        # Main Window의 모든 콤보박스 갱신
        if hasattr(self.main_window, 'acoustic_port_combos'):
            for cb in self.main_window.acoustic_port_combos:
                curr = cb.currentText()
                cb.clear()
                cb.addItems(port_list)
                if curr in port_list: cb.setCurrentText(curr)

        self.main_window.gps_port_combobox.clear()
        self.main_window.gps_port_combobox.addItems(port_list)
        if len(ports) > 1:
            self.main_window.gps_port_combobox.setCurrentText(ports[1].device)
        elif ports:
            self.main_window.gps_port_combobox.setCurrentText(ports[0].device)

    # ==========================================================
    #  Connection Handlers (Multi-Sensor)
    # ==========================================================
    @pyqtSlot(bool)
    def toggle_serial_connection(self, checked):
        """체크된 모든 센서 연결/해제"""
        if not checked:
            # 연결 해제 로직: 모든 활성 워커 중지
            for sid, worker in self.acoustic_workers.items():
                worker.stop()

            # 스레드 종료 대기
            for sid, thread in self.acoustic_threads.items():
                if thread.isRunning():
                    thread.quit()
                    thread.wait(500)

            self.acoustic_workers.clear()
            self.acoustic_threads.clear()
            self.is_connected = False
            self.view_model.img_sp = "red"
            self.log("모든 센서 연결 종료됨")
            return

        # 연결 로직: 체크된 센서들 순회
        data_size = self.main_window.data_len_spinbox.value()
        # [수정] 인터벌 값 읽기
        interval = self.main_window.interval_spinbox.value()
        connected_count = 0

        for i in range(6):
            # 체크박스 확인
            if not self.main_window.sensor_checkboxes[i].isChecked():
                continue

            port = self.main_window.acoustic_port_combos[i].currentText()
            baud_str = self.main_window.acoustic_baud_combos[i].currentText()

            if "No ports" in port or not baud_str:
                self.log(f"Sensor {i + 1}: 포트 없음, 스킵")
                continue

            try:
                # 1. 프로토콜, 워커, 스레드 생성
                protocol = Protocol()  # 각 센서마다 새 프로토콜 인스턴스
                worker = CommWorker(protocol, sensor_id=i)
                thread = QThread()

                # [수정] 설정(start_serial)에 interval 인자 추가
                worker.start_serial(port, int(baud_str), data_size, interval)

                # 2. 스레드 이동
                worker.moveToThread(thread)

                # 3. 시그널 연결
                worker.packet_received.connect(self.process_packet)
                worker.connection_status.connect(self.on_connection_status)
                worker.log_message.connect(lambda sid, msg: self.log(f"[S{sid + 1}]: {msg}"))
                worker.finished.connect(thread.quit)

                thread.started.connect(worker.run)

                # 데이터 길이 & 인터벌 변경 시그널 구독
                self.data_size_changed.connect(worker.set_data_size)
                self.interval_changed.connect(worker.set_interval)

                # 4. 저장 및 시작
                self.acoustic_workers[i] = worker
                self.acoustic_threads[i] = thread
                thread.start()
                connected_count += 1

            except Exception as e:
                self.log(f"Sensor {i + 1} 연결 준비 중 에러: {e}")

        if connected_count > 0:
            self.is_connected = True
            self.view_model.img_sp = "green"
            self.main_window.btn_serialport.setChecked(True)
        else:
            self.main_window.btn_serialport.setChecked(False)

    @pyqtSlot(int, bool, str)
    def on_connection_status(self, sensor_id, connected, msg):
        self.log(f"[S{sensor_id + 1}] {msg}")
        # 개별 상태 표시는 UI에 없으므로 로그로만 확인하거나 필요 시 추가 구현

    @pyqtSlot()
    def on_worker_finished(self):
        # 스레드 종료 처리는 toggle_serial_connection에서 일괄 처리됨
        pass

    # GPS 핸들러는 기존과 동일
    @pyqtSlot(bool)
    def toggle_gps_connection(self, checked):
        if self.is_gps_connected:
            self.gps_comm_worker.stop()
        else:
            port = self.main_window.gps_port_combobox.currentText()
            baud_str = self.main_window.gps_baud_combobox.currentText()
            if "No ports" in port:
                self.main_window.btn_gps_serialport.setChecked(False)
                return
            if not self.gps_comm_thread.isRunning():
                self.gps_comm_worker.start_serial(port, int(baud_str))
                self.gps_comm_thread.start()

    @pyqtSlot(bool, str)
    def on_gps_connection_status(self, connected, msg):
        self.is_gps_connected = connected
        self.log(msg)
        self.view_model.gps_img_sp = "green" if connected else "red"
        self.main_window.btn_gps_serialport.setChecked(connected)

    @pyqtSlot()
    def on_gps_worker_finished(self):
        if self.gps_comm_thread.isRunning():
            self.gps_comm_thread.quit()
            self.gps_comm_thread.wait()

    # ==========================================================
    #  Data Processing (Multi-Sensor)
    # ==========================================================
    @pyqtSlot(int, int, bytearray)
    def process_packet(self, sensor_id: int, command: int, payload: bytearray):
        """CommWorker로부터 수신된 패킷 처리 (Sensor ID 포함)"""
        try:
            # 1. BME/MLX Data
            if command == e_cmds.e_bme.value:
                if len(payload) >= 16:
                    vals = struct.unpack('<ffff', payload[:16])

                    # ViewModel 리스트 업데이트
                    if sensor_id < len(self.view_model.sensors):
                        self.view_model.sensors[sensor_id]['bme_temp'] = f"{vals[0]:.2f} °C"
                        self.view_model.sensors[sensor_id]['bme_hum'] = f"{vals[1]:.2f} %"
                        self.view_model.sensors[sensor_id]['bme_press'] = f"{vals[2]:.2f} hPa"
                        self.view_model.sensors[sensor_id]['mlx_temp'] = f"{vals[3]:.2f} °C"

                    self.last_bme_data[sensor_id] = {'temp': vals[0], 'humi': vals[1], 'pres': vals[2], 'mlx': vals[3]}
                return

            # 2. Acoustic Data
            rx_data = None
            tx_data = None
            is_rx = False

            if command in [e_cmds.e_rx16.value, e_cmds.e_rx12.value]:
                is_rx = True
                rx_data = self._parse_adc_data(payload, command == e_cmds.e_rx16.value)
                self.last_rx_data[sensor_id] = rx_data

            elif command in [e_cmds.e_tx16.value, e_cmds.e_tx12.value]:
                tx_data = self._parse_adc_data(payload, command == e_cmds.e_tx16.value)
                self.last_tx_data[sensor_id] = tx_data

            elif command == e_cmds.e_tx_rx12.value:
                is_rx = True
                half = len(payload) // 2
                tx_data = self._parse_adc_data(payload[:half], False)
                rx_data = self._parse_adc_data(payload[half:], False)
                self.last_tx_data[sensor_id] = tx_data
                self.last_rx_data[sensor_id] = rx_data

            # 3. Data Handling (UI Throttling & Saving)

            # 해당 센서의 마지막 데이터 가져오기
            cur_rx = self.last_rx_data.get(sensor_id)
            cur_tx = self.last_tx_data.get(sensor_id)

            # Distance calculation - always calculate when RX data is available
            if is_rx and cur_rx is not None:
                bme_data = self.last_bme_data.get(sensor_id, {'temp': 0.0, 'humi': 0.0, 'pres': 0.0, 'mlx': 0.0})
                distance_mm, processing_time_ms, tof_us, tx_start_us = self._calculate_distance(sensor_id, cur_rx, cur_tx, bme_data)
                
                # Store results
                self.last_distance[sensor_id] = distance_mm
                self.last_processing_time[sensor_id] = processing_time_ms
                self.last_tof_us[sensor_id] = tof_us
                self.last_tx_start_us[sensor_id] = tx_start_us
                
                # Update ViewModel
                if sensor_id < len(self.view_model.sensors):
                    # Handle both None and "Not Detected" string
                    if distance_mm is not None and distance_mm != "Not Detected" and isinstance(distance_mm, (int, float)):
                        self.view_model.sensors[sensor_id]['distance'] = f"{distance_mm:.2f} mm"
                        self.view_model.sensors[sensor_id]['distance_raw'] = distance_mm
                    else:
                        self.view_model.sensors[sensor_id]['distance'] = "Not Detected"
                        self.view_model.sensors[sensor_id]['distance_raw'] = None
                    self.view_model.sensors[sensor_id]['processing_time'] = f"{processing_time_ms:.2f} ms"

            # [중요] 저장 로직은 무조건 실행 (데이터 손실 방지)
            if is_rx and self.is_logging:
                now = datetime.datetime.now()
                gps_copy = self.last_gps_data.copy() if self.last_gps_data else {}
                bme_copy = self.last_bme_data.get(sensor_id, {'temp': 0.0, 'humi': 0.0, 'pres': 0.0, 'mlx': 0.0}).copy()

                rx_copy = cur_rx.copy() if cur_rx is not None else None
                tx_copy = cur_tx.copy() if cur_tx is not None else None
                distance_copy = self.last_distance.get(sensor_id, None)

                self.save_executor.submit(self.save_snapshot, sensor_id, gps_copy, bme_copy, rx_copy, tx_copy, distance_copy, now)

            # [중요] 그래프 업데이트 (계산 포함) - 활성화 상태이고 스로틀링 시간 지났을 때만 수행
            current_time = time.time()
            last_update = self.last_ui_update_time.get(sensor_id, 0)

            if self.is_graph_enabled and is_rx and (current_time - last_update) >= self.ui_update_interval:
                self.last_ui_update_time[sensor_id] = current_time

                # DSP 연산 (평균 제거 등) - UI 업데이트 할 때만 수행
                rx_processed = None
                tx_processed = None

                if cur_rx is not None and len(cur_rx) > 0:
                    rx_processed = cur_rx - np.mean(cur_rx)
                if cur_tx is not None and len(cur_tx) > 0:
                    tx_processed = cur_tx - np.mean(cur_tx)

                if rx_processed is not None:
                    f_fft, m_fft = self._calculate_fft(rx_processed)
                    t_stft, f_stft, z_stft = self._calculate_stft(rx_processed)
                    f_pfft, m_pfft = self._calculate_partial_fft(rx_processed)

                    # Get marker positions for TX start and RX start (ToF)
                    tof_us_marker = self.last_tof_us.get(sensor_id, None)
                    tx_start_us_marker = self.last_tx_start_us.get(sensor_id, None)

                    self.main_window.update_chart_data(
                        sensor_id,
                        None, rx_processed, tx_processed,
                        f_fft, m_fft, t_stft, z_stft, f_pfft, m_pfft,
                        tof_us_marker, tx_start_us_marker
                    )

        except Exception as e:
            self.log(f"S{sensor_id + 1} 패킷 처리 오류: {e}")

    def _parse_adc_data(self, payload, is_16bit):
        """Payload -> Voltage Array 변환"""
        if is_16bit:
            cnt = len(payload) // 2
            if cnt == 0: return np.array([])
            raw = struct.unpack(f'<{cnt}H', payload)
            return (np.array(raw) / 4096.0) * 3.3
        else:
            raw = []
            for i in range(0, len(payload), 3):
                if i + 2 < len(payload):
                    b1, b2, b3 = payload[i], payload[i + 1], payload[i + 2]
                    val1 = ((b1 << 4) & 0xFF0) | ((b2 >> 4) & 0x0F)
                    val2 = ((b2 & 0x0F) << 8) | b3
                    raw.extend([val1, val2])
            return (np.array(raw) / 4096.0) * 3.3

    @pyqtSlot(dict)
    def process_gps_data(self, data: dict):
        self.last_gps_data = data
        valid = data.get('is_valid', False)
        self.view_model.gps_latitude = f"{data.get('latitude_deg', 0.0):.6f} N"
        self.view_model.gps_longitude = f"{data.get('longitude_deg', 0.0):.6f} E"
        self.view_model.gps_speed_kmh = f"{data.get('speed_kmh', 0.0):.2f} Km/h"
        self.view_model.gps_validity = "✅ VALID" if valid else "❌ VOID"

    @pyqtSlot()
    def update_gui_data(self):
        if hasattr(self.main_window, 'update_all_widgets_from_vm'):
            self.main_window.update_all_widgets_from_vm()

    # ==========================================================
    #  DSP Functions
    # ==========================================================
    def _calculate_fft(self, data):
        if len(data) < 2: return None, None
        N = len(data)
        yf = np.fft.fft(data)
        xf = np.fft.fftfreq(N, 1 / FS)
        mask = (xf >= 35000) & (xf <= 45000)
        return xf[mask], np.abs(yf[mask]) / N

    def _calculate_stft(self, data):
        if len(data) < 16: return None, None, None
        nperseg = self.main_window.stft_nperseg_spinbox.value()
        noverlap = self.main_window.stft_noverlap_spinbox.value()
        nfft = self.main_window.stft_nfft_spinbox.value()
        nperseg = min(nperseg, len(data))
        noverlap = min(noverlap, nperseg - 1)
        f, t, Zxx = signal.stft(data, FS, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        mask = (f >= 35000) & (f <= 45000)
        return t, f[mask], Zxx[mask, :]

    def _calculate_partial_fft(self, data):
        start_ms = self.main_window.partial_fft_start_spinbox.value()
        end_ms = self.main_window.partial_fft_end_spinbox.value()
        idx_s = int((start_ms / 1000.0) * FS)
        idx_e = int((end_ms / 1000.0) * FS)
        if idx_s >= idx_e or idx_e > len(data): return None, None
        segment = data[idx_s:idx_e]
        if len(segment) < 2: return None, None
        yf = np.fft.fft(segment)
        xf = np.fft.fftfreq(len(segment), 1 / FS)
        mask = (xf >= 35000) & (xf <= 45000)
        return xf[mask], np.abs(yf[mask]) / len(segment)

    def _get_or_create_distance_calculator(self, sensor_id):
        """Get or create DistanceCalculator for a sensor, updating filter params from UI."""
        if sensor_id not in self.distance_calculators:
            self.distance_calculators[sensor_id] = DistanceCalculator(sampling_rate=FS)
            self.kalman_filters[sensor_id] = DistanceKalmanFilter()
            self.distance_history[sensor_id] = []
            self.miss_streak[sensor_id] = 0
        
        # Update filter parameters from UI
        calc = self.distance_calculators[sensor_id]
        if hasattr(self.main_window, 'distance_bandpass_low_spinbox'):
            calc.bandpass_low = self.main_window.distance_bandpass_low_spinbox.value() * 1000.0  # kHz to Hz
            calc.bandpass_high = self.main_window.distance_bandpass_high_spinbox.value() * 1000.0  # kHz to Hz
            calc.envelope_cutoff = self.main_window.distance_envelope_cutoff_spinbox.value() * 1000.0  # kHz to Hz
            calc.filter_order = self.main_window.distance_filter_order_spinbox.value()
        
        return calc

    def _calculate_distance(self, sensor_id, rx_data, tx_data, bme_data):
        """
        Calculate distance from RX/TX data and BME data.
        Returns (distance_mm, processing_time_ms, tof_us, tx_start_us)
        """
        start_time = time.perf_counter()
        tof_us = "Not Detected"
        tx_start_us = "Not Detected"
        
        try:
            # Get or create distance calculator
            calc = self._get_or_create_distance_calculator(sensor_id)
            kalman = self.kalman_filters[sensor_id]
            
            # Get BME data for temperature and humidity
            temp_C = bme_data.get('temp') if bme_data else None
            humi_percent = bme_data.get('humi') if bme_data else None
            
            # Handle empty or missing RX data (don't return early, let predict_distance use cache)
            if rx_data is None or len(rx_data) == 0:
                tof_us = "Not Detected"
            else:
                # Convert voltage arrays to ADC values (0-4095 range)
                rx_adc = np.clip(np.round(rx_data * (4096 / 3.3)), 0, 4095).astype(float)
                
                # Process RX signal for ToF detection
                rx_filtered = calc.bandpass_filter(rx_adc)
                rx_envelope = calc.envelope_detection(rx_filtered)
                
                # Pass TX signal for stable noise estimation (if available)
                tx_adc_for_noise = None
                if tx_data is not None and len(tx_data) > 0:
                    tx_adc_for_noise = np.clip(np.round(tx_data * (4096 / 3.3)), 0, 4095).astype(float)
                
                tof_samples = calc.detect_tof(rx_envelope, tx_signal=tx_adc_for_noise)
                
                if tof_samples != "Not Detected":
                    # Convert ToF from samples to microseconds (1 sample = 1 μs at 1 MHz)
                    tof_us = float(tof_samples)
            
            # Handle empty or missing TX data
            if tx_data is None or len(tx_data) == 0:
                tx_start_us = "Not Detected"
            else:
                # Process TX signal for start time detection
                tx_adc = np.clip(np.round(tx_data * (4096 / 3.3)), 0, 4095).astype(float)
                tx_start_samples = calc.detect_tx_start(tx_adc)
                
                if tx_start_samples != "Not Detected":
                    # Convert TX start from samples to microseconds (1 sample = 1 μs at 1 MHz)
                    tx_start_us = float(tx_start_samples)
            
            # ALWAYS call predict_distance (even with "Not Detected") to use cache for empty files
            distance_raw = calc.predict_distance(tof_us, tx_start_us, temp_C, humi_percent)
            
            # Apply smoothing filters
            if distance_raw is not None and distance_raw != "Not Detected" and isinstance(distance_raw, (int, float)):
                # Reset miss streak on valid measurement
                self.miss_streak[sensor_id] = 0
                
                # Add to history
                self.distance_history[sensor_id].append(float(distance_raw))
                if len(self.distance_history[sensor_id]) > 10:
                    self.distance_history[sensor_id].pop(0)
                
                # Hampel filter
                if len(self.distance_history[sensor_id]) >= 5:
                    history_arr = np.array(self.distance_history[sensor_id])
                    filtered_arr = hampel_filter(history_arr, window=5, n_sigma=3.0)
                    distance_hampel = filtered_arr[-1]
                else:
                    distance_hampel = float(distance_raw)
                
                # Kalman filter
                dt = 0.2
                kalman.predict(dt)
                distance_mm = kalman.update(distance_hampel)
            else:
                # Missed detection; increment streak and optionally clear cache/filters
                self.miss_streak[sensor_id] = self.miss_streak.get(sensor_id, 0) + 1
                if self.miss_streak[sensor_id] >= 3:
                    # Clear cached distance in calculator to avoid sticky stale values
                    calc.last_valid_distance = None
                    self.distance_history[sensor_id] = []
                    kalman.reset()
                distance_mm = distance_raw  # Could be "Not Detected" or cached value
            
            processing_time_ms = (time.perf_counter() - start_time) * 1000.0
            
            # Return tof_us and tx_start_us properly (convert "Not Detected" to None for display)
            tof_display = tof_us if tof_us != "Not Detected" else None
            tx_display = tx_start_us if tx_start_us != "Not Detected" else None
            
            return (distance_mm, processing_time_ms, tof_display, tx_display)
            
        except Exception as e:
            self.log(f"S{sensor_id + 1} Distance calculation error: {e}")
            processing_time_ms = (time.perf_counter() - start_time) * 1000.0
            return (None, processing_time_ms, None, None)

    # ==========================================================
    #  File Saving
    # ==========================================================
    @pyqtSlot()
    def select_save_path(self):
        path = QFileDialog.getExistingDirectory(self.main_window, "Select Directory")
        if path: self.main_window.save_path_lineedit.setText(path)

    @pyqtSlot(bool)
    def toggle_continuous_save(self, checked):
        if checked and not self.main_window.save_path_lineedit.text():
            QMessageBox.warning(self.main_window, "경고", "저장 경로를 선택하세요.")
            self.main_window.save_data_button.setChecked(False)
            return
        self.is_logging = checked
        self.save_base_dir = self.main_window.save_path_lineedit.text()
        style = "color: white; background-color: red;" if checked else "color: red; background-color: #F0F0F0;"
        self.main_window.save_status_label.setText("SAVE: ON" if checked else "SAVE: OFF")
        self.main_window.save_status_label.setStyleSheet(style)

    # [수정] 독립적으로 동작하도록 인자를 모두 받음
    def save_snapshot(self, sensor_id, gps, bme, cur_rx, cur_tx, distance_mm, now):
        folder = now.strftime("%Y%m%d%H%M%S") + f"_{now.microsecond // 1000:03d}"
        sensor_dir_name = f"Sensor_{sensor_id + 1}"
        path = os.path.join(self.save_base_dir, sensor_dir_name, folder)
        os.makedirs(path, exist_ok=True)

        # GPS 값 추출
        lat = gps.get('latitude_deg')
        lng = gps.get('longitude_deg')
        speed = gps.get('speed_kmh', 0.0)

        # JSON 포맷 확장
        info = {
            "sensorId": sensor_id + 1,
            "loggedAt": now.isoformat() + 'Z',
            "gatewayId": 1,
            "lat": lat,
            "lng": lng,
            "temp": f"{bme.get('temp', 0.0):.2f}",
            "objectTemp": f"{bme.get('mlx', 0.0):.2f}",
            "humi": f"{bme.get('humi', 0.0):.2f}",
            "pressure": f"{bme.get('pres', 0.0):.2f}",
            "dewPoint": "0.00",
            "surface": 1,
            "skid": 0,
            "heuristicsSkid": -1,
            "kmaWeather": -1,
            "futureTemp": None,
            "surfaceStatus": None,
            "distance in mm": f"{distance_mm:.2f}" if distance_mm is not None else None,
            "gps": {
                "lat": lat,
                "lng": lng,
                "alt": None,
                "speed": speed,
                "gqi": 0,
                "hdop": 99.99
            },
            "surfaceType": {
                "result": 1
            },
            "version": 3.2
        }

        with open(os.path.join(path, "info.json"), 'w') as f:
            json.dump(info, f, indent=4)

        if cur_rx is not None:
            raw_rx = np.clip(np.round(cur_rx * (4096 / 3.3)), 0, 4095).astype(int)
            with open(os.path.join(path, "rx_plain.dat"), 'w') as f:
                f.write("\n".join(map(str, raw_rx)))

        if cur_tx is not None:
            raw_tx = np.clip(np.round(cur_tx * (4096 / 3.3)), 0, 4095).astype(int)
            with open(os.path.join(path, "tx_plain.dat"), 'w') as f:
                f.write("\n".join(map(str, raw_tx)))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ctrl = ApplicationController(app)
    ctrl.main_window.show()
    app.aboutToQuit.connect(ctrl.cleanup)
    sys.exit(app.exec_())