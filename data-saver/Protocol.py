# -*- coding: utf-8 -*-
import enum
from dataclasses import dataclass, field
import copy


# --- C#의 Enum들 ---

class e_rx_frame(enum.Enum):
    STX = 0
    CMD = 1
    OPT = 2
    LENGTH_3 = 3  # Little-Endian LSB
    LENGTH_2 = 4
    LENGTH_1 = 5
    LENGTH_0 = 6  # MSB
    PAYLOAD = 7
    CHECKSUM = 8
    ETX = 9
    ERROR = 10


class e_cmds(enum.IntEnum):
    e_stx = 0x02
    e_etx = 0x03

    e_cmd_rsvd = 0x10
    e_tx16 = 0x11
    e_rx16 = 0x12
    e_bme = 0x13
    e_tx12 = 0x14
    e_rx12 = 0x15
    e_tx_rx12 = 0x16

    e_rpi4cmd_rsvd = 0x20
    e_rpi4init = 0x21
    e_rpi4cmd = 0x22

    e_status = 0x30
    e_parameter = 0x40
    e_cmd_none = 0xFF


class e_opts_data(enum.Enum):
    e_data_us = 0x00
    e_data_acc = 0x01 # BME/MLX 요청 옵션으로 가정하고 CommWorker에서 사용
    e_data_ctrl = 0x02 # 기존 0x03에서 0x02로 순서 수정 (0x00, 0x01이 사용되었으므로)


class e_opts_init(enum.Enum):
    e_impulse_tx = 0x00
    e_continuous_tx = 0x01
    e_emissivity = 0x02


# --- C#의 Utility.LogManager (임시 구현) ---

class LogManager:
    """ C# LogManager의 임시 파이썬 대체 """

    def WriteLine(self, message: str, is_error: bool = False):
        prefix = "ERROR: " if is_error else "LOG: "
        # print(prefix + message)


# --- C#의 Struct들 -> Python @dataclass ---

@dataclass
class ClzmMessage:
    """ C# clzm_message_t 구조체 대체 """
    stx: int = 0
    cmd: int = 0
    opt: int = 0
    length: int = 0
    payload: bytearray = field(default_factory=bytearray)
    checksum: int = 0
    etx: int = 0


@dataclass
class MessageStatus:
    """ C# message_status_t 구조체 대체 """
    state: e_rx_frame = e_rx_frame.STX
    error: int = 0
    cnt: int = 0
    checksum: int = 0xFF


# --- C# Protocol 클래스 -> Python Protocol 클래스 ---

class Protocol:
    """
    C# Protocol 클래스를 파이썬으로 이식.
    센서로부터 들어오는 바이트 스트림을 파싱하는 상태 머신(State Machine)입니다.
    """

    def __init__(self):
        self.log = LogManager()
        self._current_msg = ClzmMessage()
        self._status = MessageStatus()
        self._last_good_msg = ClzmMessage()
        self._packet_counter = 0
        self.reset_parser()

    def reset_parser(self):
        """파서의 상태와 현재 메시지를 초기화합니다."""
        self._status = MessageStatus(state=e_rx_frame.STX, checksum=0xFF)
        self._current_msg = ClzmMessage()

    # --- 외부에서 완성된 데이터에 접근하기 위한 속성(Properties) ---
    @property
    def command(self) -> int:
        return self._last_good_msg.cmd

    @property
    def option(self) -> int:
        return self._last_good_msg.opt

    @property
    def payload(self) -> bytearray:
        return self._last_good_msg.payload

    @property
    def last_error(self) -> int:
        return self._status.error

    def frame_char(self, c: int) -> bool:
        """
        데이터 스트림에서 한 바이트(c)를 입력받아 파싱을 수행합니다.
        패킷 하나가 성공적으로 완성되면 True를 반환합니다.
        """

        # --- STX (Start of Text) ---
        if self._status.state == e_rx_frame.STX:
            if c == e_cmds.e_stx.value:
                self.reset_parser()
                self._current_msg.stx = c
                self._status.state = e_rx_frame.CMD

        # --- CMD (Command) ---
        elif self._status.state == e_rx_frame.CMD:
            self._current_msg.cmd = c
            self._status.state = e_rx_frame.OPT

        # --- OPT (Option) ---
        elif self._status.state == e_rx_frame.OPT:
            self._current_msg.opt = c
            self._status.state = e_rx_frame.LENGTH_3

        # --- LENGTH (Little-Endian 4-byte) ---
        elif self._status.state == e_rx_frame.LENGTH_3:
            self._current_msg.length = (c & 0xFF)
            self._status.state = e_rx_frame.LENGTH_2

        elif self._status.state == e_rx_frame.LENGTH_2:
            self._current_msg.length |= ((c << 8) & 0xFF00)
            self._status.state = e_rx_frame.LENGTH_1

        elif self._status.state == e_rx_frame.LENGTH_1:
            self._current_msg.length |= ((c << 16) & 0xFF0000)
            self._status.state = e_rx_frame.LENGTH_0

        elif self._status.state == e_rx_frame.LENGTH_0:
            self._current_msg.length |= ((c << 24) & 0xFF000000)

            if self._current_msg.length > 96000:
                self._status.state = e_rx_frame.ERROR
            else:
                self._current_msg.payload = bytearray(self._current_msg.length)
                self._status.cnt = 0 # 카운트 초기화

                if self._current_msg.length > 0:
                    self._status.state = e_rx_frame.PAYLOAD
                else:
                    self._status.state = e_rx_frame.CHECKSUM # 길이가 0이면 바로 체크섬으로 이동

        # --- PAYLOAD (Data) ---
        elif self._status.state == e_rx_frame.PAYLOAD:
            if self._status.cnt < self._current_msg.length:
                # 페이로드가 bytearray(길이)로 미리 초기화되어 있으므로 인덱스로 접근
                self._current_msg.payload[self._status.cnt] = c
                self._status.checksum = (self._status.checksum ^ c) & 0xFF  # XOR 연산
                self._status.cnt += 1

            if self._status.cnt >= self._current_msg.length:
                self._status.state = e_rx_frame.CHECKSUM

        # --- CHECKSUM ---
        elif self._status.state == e_rx_frame.CHECKSUM:
            self._current_msg.checksum = c

            if self._current_msg.checksum == self._status.checksum:
                self._status.state = e_rx_frame.ETX
            else:
                self.log.WriteLine(f"CHECKSUM ERROR: Got {c:X2}, Expected {self._status.checksum:X2}", is_error=True)
                self._status.state = e_rx_frame.ERROR

        # --- ETX (End of Text) ---
        elif self._status.state == e_rx_frame.ETX:
            self._current_msg.etx = c
            if self._current_msg.etx == e_cmds.e_etx.value:
                # --- 패킷 완성! ---
                self._last_good_msg = copy.deepcopy(self._current_msg)
                self._packet_counter += 1
                self._status.state = e_rx_frame.STX # 다음 패킷을 위해 STX로 리셋

                return True  # 패킷 완성 신호 반환
            else:
                self._status.state = e_rx_frame.ERROR  # ETX 불일치

        # --- ERROR / Default ---
        if self._status.state == e_rx_frame.ERROR:
            err_str = f"Packet Error, CMD = {self._current_msg.cmd:X2}, Packet Counter = {self._packet_counter}"
            self.log.WriteLine(err_str, is_error=True)
            self._status.error = -1
            self._status.state = e_rx_frame.STX  # 에러 발생 시 STX로 리셋

        return False  # 패킷이 아직 완성되지 않음