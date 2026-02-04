# -*- coding: utf-8 -*-
import serial
import time
import struct
import numpy as np

try:
    from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
except ImportError:
    try:
        from PySide6.QtCore import QObject, Signal as pyqtSignal, Slot as pyqtSlot
    except ImportError:
        raise ImportError("PyQt/PySide (5 or 6)를 찾을 수 없습니다.")

from Protocol import Protocol, e_cmds, e_opts_data, e_opts_init


class CommWorker(QObject):
    """
    개별 센서 통신을 담당하는 워커.
    다중 연결 지원 및 송신 주기(Interval) 설정 가능.
    """

    # --- 시그널 정의 (sensor_id 포함) ---
    packet_received = pyqtSignal(int, int, bytearray)
    connection_status = pyqtSignal(int, bool, str)
    log_message = pyqtSignal(int, str)
    finished = pyqtSignal()

    def __init__(self, protocol: Protocol, sensor_id: int, parent=None):
        super().__init__(parent)
        self._protocol = protocol
        self.sensor_id = sensor_id

        self._comm_type = None
        self._is_running = False
        self._port = None
        self._baud = None
        self._data_size = 10000
        self._interval_ms = 1000  # 기본 1초
        self._connection = None

        self._pkt_us = bytearray()
        self._pkt_bme = bytearray()

    @pyqtSlot(str, int, int, int)
    def start_serial(self, port: str, baud: int, data_size: int, interval_ms: int):
        self._comm_type = "serial"
        self._port = port
        self._baud = baud
        self._data_size = data_size
        self._interval_ms = interval_ms
        self.log_message.emit(self.sensor_id, f"시작: {port}@{baud}, Len={data_size}, Int={interval_ms}ms")

    @pyqtSlot(int)
    def set_data_size(self, size: int):
        if self._data_size != size:
            self._data_size = size
            self.log_message.emit(self.sensor_id, f"길이 변경: {size}")
            if self._is_running:
                self._refresh_packets()

    @pyqtSlot(int)
    def set_interval(self, ms: int):
        """[신규] 실행 중 송신 주기 변경"""
        if self._interval_ms != ms:
            self._interval_ms = ms
            self.log_message.emit(self.sensor_id, f"주기 변경: {ms}ms")

    @pyqtSlot()
    def stop(self):
        self.log_message.emit(self.sensor_id, "중지 요청")
        self._is_running = False

    @pyqtSlot()
    def run(self):
        if self._comm_type == "serial":
            self._run_serial_loop()

        self.log_message.emit(self.sensor_id, "스레드 종료")
        self.finished.emit()

    def _create_packet(self, cmd: int, opt: int, payload: bytearray) -> bytearray:
        packet = bytearray()
        packet.append(e_cmds.e_stx.value)
        packet.append(cmd)
        packet.append(opt)

        length = len(payload) if payload else 0
        packet.extend(struct.pack('<I', length))

        if payload:
            packet.extend(payload)

        checksum = 0xFF
        packet.append(checksum)
        packet.append(e_cmds.e_etx.value)
        return packet

    def _refresh_packets(self):
        payload_bytes_us = (self._data_size).to_bytes(2, byteorder='little')
        self._pkt_us = self._create_packet(
            e_cmds.e_rpi4cmd.value,
            e_opts_data.e_data_acc.value,
            payload_bytes_us
        )
        self._pkt_bme = self._create_packet(
            e_cmds.e_rpi4cmd.value,
            e_opts_data.e_data_us.value,
            bytearray()
        )

    def _run_serial_loop(self):
        self._is_running = True
        try:
            self._connection = serial.Serial(
                port=self._port,
                baudrate=self._baud,
                timeout=0.01,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                rtscts=False,
                dsrdtr=False
            )
            self.connection_status.emit(self.sensor_id, True, f"포트 열림: {self._port}")

        except Exception as e:
            self.connection_status.emit(self.sensor_id, False, f"연결 실패: {e}")
            self._is_running = False
            return

        self._refresh_packets()

        pkt_init = self._create_packet(e_cmds.e_rpi4init.value, e_opts_init.e_impulse_tx.value, bytearray())
        try:
            self._connection.write(pkt_init)
        except:
            pass

        last_request_time = time.time()
        request_toggle = False

        while self._is_running:
            try:
                current_time = time.time()

                # [수정] 설정된 Interval 주기로 요청 전송
                interval_sec = self._interval_ms / 1000.0

                if (current_time - last_request_time) >= interval_sec:
                    if request_toggle:
                        self._connection.write(self._pkt_bme)
                        request_toggle = False
                    else:
                        self._connection.write(self._pkt_us)
                        request_toggle = True
                    last_request_time = current_time

                if self._connection.in_waiting > 0:
                    raw_data = self._connection.read(self._connection.in_waiting)
                    for byte in raw_data:
                        if self._protocol.frame_char(byte):
                            self.packet_received.emit(
                                self.sensor_id,
                                self._protocol.command,
                                self._protocol.payload
                            )
                else:
                    time.sleep(0.001)

            except Exception as e:
                self.log_message.emit(self.sensor_id, f"통신 에러: {e}")
                self._is_running = False
                break

        if self._connection:
            self._connection.close()
        self.connection_status.emit(self.sensor_id, False, "연결 해제됨")