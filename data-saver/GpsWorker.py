# -*- coding: utf-8 -*-
import serial
import time
import struct
import json
import sys
from datetime import datetime
from typing import Optional, Dict

# Qt 라이브러리 임포트 (PyQt5/PySide6 호환성 유지)
try:
    from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QThread
except ImportError:
    try:
        from PySide6.QtCore import QObject, Signal as pyqtSignal, Slot as pyqtSlot, QThread
    except ImportError:
        # 이 에러는 PyQt5나 PySide6가 설치되지 않았을 경우 발생합니다.
        raise ImportError("PyQt5 또는 PySide6가 필요합니다.")

# NMEA 파싱 헬퍼 함수
MISSING_VALUE = 999.99


def ddm_to_deg(ddm_value: str, direction: str) -> float:
    """DDM (도분.분) 형식의 위/경도를 십진수 (Decimal Degrees)로 변환합니다."""
    if not ddm_value:
        return 0.0
    try:
        if 'N' in direction or 'S' in direction:
            degree_length = 2
        elif 'E' in direction or 'W' in direction:
            degree_length = 3
        else:
            return 0.0

        if len(ddm_value) < degree_length: return 0.0

        degree = int(ddm_value[:degree_length])
        minute = float(ddm_value[degree_length:])

        decimal_degrees = degree + (minute / 60)

        if direction in ['S', 'W']:
            return -decimal_degrees
        return decimal_degrees
    except ValueError:
        return 0.0


def parse_rmc_data(line: str) -> Optional[Dict]:
    """$GPRMC 문장에서 핵심 정보를 추출하여 딕셔너리로 반환합니다."""
    try:
        parts = line.split(',')
        if len(parts) < 12 or not parts[0].endswith('RMC'):
            return None

        status = parts[2]
        is_valid = (status == 'A')

        # 유효하지 않은 데이터이거나 필드가 부족하면 파싱 중단
        if not is_valid: return None
        if not parts[3] or not parts[5]: return None

        # 1. 속도 (Knots) 및 KM/H 계산
        speed_knots_str = parts[7]
        speed_knots = float(speed_knots_str) if speed_knots_str else 0.0  # 속도는 0.0 처리 (누락된 값은 0.0으로 간주)
        speed_kmh = round(speed_knots * 1.852, 2)

        # 2. 위도/경도
        latitude = ddm_to_deg(parts[3], parts[4])
        longitude = ddm_to_deg(parts[5], parts[6])

        # 3. UTC 시간
        utc_time = parts[1][:6]

        # 4. JSON 객체 생성
        return {
            'gps_time_utc': utc_time,
            'latitude_deg': round(latitude, 6),
            'longitude_deg': round(longitude, 6),
            'speed_kmh': speed_kmh,
            'is_valid': is_valid
        }

    except Exception as e:
        print(f"[GPS WORKER] RMC Parsing Error on line: {line} -> {e}")
        return None


class GpsWorker(QObject):
    # 시그널 정의
    connection_status = pyqtSignal(bool, str)  # 연결 상태 (True/False, 메시지)
    gps_data_received = pyqtSignal(dict)  # 파싱된 GPS 데이터
    log_message = pyqtSignal(str)  # 로그 메시지 전송

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self.ser = None
        self.port = ""
        self.baudrate = 0

    def start_serial(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self._running = True

    def stop(self):
        self._running = False

    @pyqtSlot()
    def run(self):
        """별도 스레드에서 실행되는 메인 루프"""
        self.log_message.emit(f"GPS Worker 스레드 시작.")

        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            self.connection_status.emit(True, f"GPS 시리얼 포트 연결 성공: {self.port}")
        except Exception as e:
            self.connection_status.emit(False, f"GPS 시리얼 포트 연결 실패: {e}")
            self._running = False
            return

        while self._running:
            try:
                if self.ser.in_waiting > 0:
                    # 한 줄(NMEA 문장)을 읽습니다.
                    line = self.ser.readline().decode('ascii', errors='ignore').strip()

                    if line.startswith('$GPRMC') or line.startswith('$GNRMC'):
                        parsed_data = parse_rmc_data(line)
                        if parsed_data:
                            self.gps_data_received.emit(parsed_data)
                            # self.log_message.emit(f"GPS Data Sent: {parsed_data['latitude_deg']:.6f}")
                        # else:
                        # self.log_message.emit("RMC 데이터 유효하지 않음 (Fix V)")

            except serial.SerialException as e:
                self.connection_status.emit(False, f"GPS 통신 오류: {e}")
                self._running = False
            except Exception as e:
                self.log_message.emit(f"GPS Worker 오류: {e}")

            time.sleep(0.01)  # CPU 부하 줄이기

        # 루프 종료 후 정리
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.connection_status.emit(False, "GPS 연결 해제됨.")
        self.log_message.emit("GPS Worker 스레드 종료.")
        self.finished.emit()  # 스레드 종료 시그널 (QThread 연결용)

    # QThread 연결을 위한 finished 시그널 추가 (PyQt/PySide 버전 차이 처리)
    if 'PyQt5' in sys.modules:
        finished = pyqtSignal()
    else:
        finished = pyqtSignal()