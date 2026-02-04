# -*- coding: utf-8 -*-
try:
    from PyQt5.QtCore import QObject, pyqtSignal
except ImportError:
    try:
        from PySide6.QtCore import QObject, Signal as pyqtSignal
    except ImportError:
        # 이 파일은 Qt 의존성이 있으므로, 누락 시 오류를 발생시켜야 함
        raise ImportError("PyQt5 또는 PySide6가 필요합니다.")


class ViewModel(QObject):
    # --- 시그널 정의 ---
    bme_temp_changed = pyqtSignal(str)
    bme_humidity_changed = pyqtSignal(str)
    bme_pressure_changed = pyqtSignal(str)
    mlx_temp_changed = pyqtSignal(str)
    img_sp_changed = pyqtSignal(str)  # 메인 포트 상태 시그널

    # --- [GPS 시그널 추가] ---
    gps_latitude_changed = pyqtSignal(str)
    gps_longitude_changed = pyqtSignal(str)
    gps_speed_kmh_changed = pyqtSignal(str)
    gps_validity_changed = pyqtSignal(str)
    gps_img_sp_changed = pyqtSignal(str)  # GPS 포트 상태 시그널

    def __init__(self, parent=None):
        super().__init__(parent)

        # --- Private 초기값 (센서 데이터) ---
        self._bme_temperature = 0.0
        self._bme_humidity = 0.0
        self._bme_pressure = 0.0
        self._mlx_temperature = 0.0

        # --- Private 초기값 (UI 상태) ---
        self._img_sp = "red"  # 메인 포트 상태

        # --- [CRITICAL FIX] GPS Fields Private 초기화 ---
        self._gps_latitude = "0.000000 N"
        self._gps_longitude = "0.000000 E"
        self._gps_speed_kmh = "0.00 Km/h"
        self._gps_validity = "❌ VOID"
        self._gps_img_sp = "red"  # GPS 포트 상태

    # =================================================================
    # --- BME/MLX 속성 (Float Value to String Property) ---
    # =================================================================

    # --- BME Temperature ---
    @property
    def bme_temperature(self) -> str:
        return f"{self._bme_temperature:.2f} °C"

    @bme_temperature.setter
    def bme_temperature(self, value: float):
        if self._bme_temperature != value:
            self._bme_temperature = value
            self.bme_temp_changed.emit(f"{value:.2f} °C")

    # --- BME Humidity ---
    @property
    def bme_humidity(self) -> str:
        return f"{self._bme_humidity:.2f} %"

    @bme_humidity.setter
    def bme_humidity(self, value: float):
        if self._bme_humidity != value:
            self._bme_humidity = value
            self.bme_humidity_changed.emit(f"{value:.2f} %")

    # --- BME Pressure ---
    @property
    def bme_pressure(self) -> str:
        return f"{self._bme_pressure:.2f} hPa"

    @bme_pressure.setter
    def bme_pressure(self, value: float):
        if self._bme_pressure != value:
            self._bme_pressure = value
            self.bme_pressure_changed.emit(f"{value:.2f} hPa")

    # --- MLX Temperature ---
    @property
    def mlx_temperature(self) -> str:
        return f"{self._mlx_temperature:.2f} °C"

    @mlx_temperature.setter
    def mlx_temperature(self, value: float):
        if self._mlx_temperature != value:
            self._mlx_temperature = value
            self.mlx_temp_changed.emit(f"{value:.2f} °C")

    # --- Serial Port Status Image (Main) ---
    @property
    def img_sp(self) -> str:
        return self._img_sp

    @img_sp.setter
    def img_sp(self, value: str):
        if self._img_sp != value:
            self._img_sp = value
            self.img_sp_changed.emit(value)

    # =================================================================
    # --- GPS 속성 (String Property) ---
    # =================================================================

    # --- GPS Latitude ---
    @property
    def gps_latitude(self) -> str:
        return self._gps_latitude

    @gps_latitude.setter
    def gps_latitude(self, value: str):
        if self._gps_latitude != value:
            self._gps_latitude = value
            self.gps_latitude_changed.emit(value)

    # --- GPS Longitude ---
    @property
    def gps_longitude(self) -> str:
        return self._gps_longitude

    @gps_longitude.setter
    def gps_longitude(self, value: str):
        if self._gps_longitude != value:
            self._gps_longitude = value
            self.gps_longitude_changed.emit(value)

    # --- GPS Speed (Km/h) ---
    @property
    def gps_speed_kmh(self) -> str:
        return self._gps_speed_kmh

    @gps_speed_kmh.setter
    def gps_speed_kmh(self, value: str):
        if self._gps_speed_kmh != value:
            self._gps_speed_kmh = value
            self.gps_speed_kmh_changed.emit(value)

    # --- GPS Validity ---
    @property
    def gps_validity(self) -> str:
        return self._gps_validity

    @gps_validity.setter
    def gps_validity(self, value: str):
        if self._gps_validity != value:
            self._gps_validity = value
            self.gps_validity_changed.emit(value)

    # --- GPS Port Status Image ---
    @property
    def gps_img_sp(self) -> str:
        return self._gps_img_sp

    @gps_img_sp.setter
    def gps_img_sp(self, value: str):
        if self._gps_img_sp != value:
            self._gps_img_sp = value
            self.gps_img_sp_changed.emit(value)