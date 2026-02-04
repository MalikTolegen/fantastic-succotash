import os
import json
import time

# ⚠️ 현재 사용하고 계신 JSON 파일 경로와 이름을 그대로 사용합니다.
FILE_PATH = 'C:\\Users\\servi\\Desktop\\Data_fft\\PythonProject\\gps_data_log.json'
TEST_DATA = {
    "test_id": 1,
    "timestamp": time.time(),
    "message": "파일 쓰기 테스트 데이터입니다. 이 파일이 비어 있으면 권한 문제입니다."
}


def run_write_test(file_path, data):
    """지정된 경로에 JSON 데이터를 강제로 기록하는 테스트 함수입니다."""

    print("======================================================")
    print(f"📝 파일 쓰기 테스트 시작: {file_path}")

    try:
        # 파일 핸들러와 OS 버퍼를 강제로 동기화하여 데이터를 씁니다.
        with open(file_path, 'w') as f:  # 'w' 모드로 덮어쓰기
            json_line = json.dumps(data, indent=4)
            f.write(json_line + '\n')
            f.flush()

            # 운영체제에 즉시 디스크에 쓰도록 강제 요청
            os.fsync(f.fileno())

        print(f"✅ 테스트 데이터 쓰기 성공! 파일을 열어 확인하세요.")
        print(f"   (이 파일이 비어 있다면, 해당 경로에 대한 쓰기 권한이 없습니다.)")

    except PermissionError:
        print("❌ 오류: [PermissionError]")
        print("   -> 현재 사용자에게 해당 폴더(PythonProject)에 쓸 권한이 없습니다.")
        print("   -> 해결책: 파이썬을 **관리자 권한**으로 실행해 보세요.")
    except Exception as e:
        print(f"❌ 예기치 않은 오류 발생: {e}")
        print("   -> 해결책: 파일 경로를 점검하거나 백신 프로그램을 확인하세요.")

    print("======================================================")


if __name__ == "__main__":
    run_write_test(FILE_PATH, TEST_DATA)