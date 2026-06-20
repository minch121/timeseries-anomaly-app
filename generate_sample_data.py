"""
샘플 다변량 시계열 데이터 생성 (이상치 주입)
"""
import numpy as np
import pandas as pd

np.random.seed(42)

def generate_sensor_data():
    """공장 센서 데이터 시뮬레이션 (온도, 진동, 압력)"""
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    
    # 기본 패턴
    t = np.arange(n)
    trend = 0.005 * t
    seasonal = 5 * np.sin(2 * np.pi * t / 24)  # 24시간 주기
    
    # 온도: 추세 + 계절성 + 노이즈
    temp = 25 + trend + seasonal + np.random.normal(0, 1, n)
    # 진동: 약한 계절성 + 노이즈
    vibration = 0.5 + 0.2 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 0.1, n)
    # 압력: 추세 + 노이즈
    pressure = 100 + 0.003 * t + np.random.normal(0, 2, n)
    
    # 이상치 주입
    anomaly_idx = [50, 51, 120, 200, 201, 202, 310, 400, 401]
    temp[anomaly_idx] += np.random.choice([15, -15], size=len(anomaly_idx))
    vibration[[50, 120, 200, 310, 400]] += np.random.uniform(1.5, 3.0, size=5)
    pressure[[51, 201, 202, 401]] += np.random.choice([30, -30], size=4)
    
    # 정답 라벨 (이상=1)
    label = np.zeros(n, dtype=int)
    label[anomaly_idx] = 1

    # 결측치 주입
    temp[75] = np.nan
    temp[76] = np.nan
    vibration[150] = np.nan
    pressure[300] = np.nan

    df = pd.DataFrame({
        "timestamp": dates,
        "temperature": np.round(temp, 2),
        "vibration": np.round(vibration, 4),
        "pressure": np.round(pressure, 2),
        "is_anomaly": label
    })
    return df

def generate_server_data():
    """서버 모니터링 데이터 시뮬레이션 (CPU, 메모리, 네트워크)"""
    n = 720
    dates = pd.date_range("2023-06-01", periods=n, freq="h")
    t = np.arange(n)
    
    # 일주기 패턴 (업무시간에 높음)
    daily = 20 * np.sin(2 * np.pi * t / 24 - np.pi / 2)
    weekly = 5 * np.sin(2 * np.pi * t / 168)
    
    cpu = 45 + daily + weekly + np.random.normal(0, 5, n)
    cpu = np.clip(cpu, 5, 100)
    
    memory = 60 + 0.01 * t + 10 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 3, n)
    memory = np.clip(memory, 10, 100)
    
    network = 500 + 200 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 50, n)
    network = np.clip(network, 0, None)
    
    # 이상치: CPU 스파이크
    spike_idx = [100, 101, 250, 251, 252, 450, 600, 601]
    cpu[spike_idx] = np.random.uniform(90, 100, len(spike_idx))
    memory[[100, 250, 251, 450, 600]] += 25
    network[[101, 252, 601]] *= 3
    
    # 정답 라벨 (이상=1)
    label = np.zeros(n, dtype=int)
    label[spike_idx] = 1

    # 결측치
    cpu[180] = np.nan
    memory[360] = np.nan

    df = pd.DataFrame({
        "timestamp": dates,
        "cpu_usage": np.round(cpu, 2),
        "memory_usage": np.round(memory, 2),
        "network_traffic": np.round(network, 2),
        "is_anomaly": label
    })
    return df

if __name__ == "__main__":
    df1 = generate_sensor_data()
    df1.to_csv("sample_data/factory_sensor_data.csv", index=False)
    print(f"Factory sensor data: {df1.shape}")
    
    df2 = generate_server_data()
    df2.to_csv("sample_data/server_monitoring_data.csv", index=False)
    print(f"Server monitoring data: {df2.shape}")
    
    print("Sample data generated successfully!")
