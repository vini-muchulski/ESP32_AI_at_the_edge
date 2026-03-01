#!/usr/bin/env python3
"""
Python client to test the ESP32 sine API
"""

import requests

# Set your ESP32 IP address here
ESP32_IP = "192.168.3.22"  # Replace with your ESP32 IP
BASE_URL = f"http://{ESP32_IP}"

def test_sine(angle):
    """Tests sine calculation for a specific angle."""
    try:
        url = f"{BASE_URL}/sine"
        params = {"angle": angle}

        print(f"Testing sin({angle} deg)...")
        response = requests.get(url, params=params, timeout=5)

        if response.status_code == 200:
            data = response.json()
            print(f"OK: sin({data['angle_degrees']} deg) = {data['sine']:.6f}")
            return data
        else:
            print(f"HTTP error {response.status_code}: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
        return None

def main():
    print("=== ESP32 API Test - Sine Calculation ===\n")

    # Test angles
    test_angles = [0, 90, 30, 45, 180]

    print(f"Connecting to ESP32 at: {BASE_URL}\n")

    # Test each angle
    successes = 0
    for angle in test_angles:
        result = test_sine(angle)
        if result:
            successes += 1
        print()  # Blank line

    print(f"Test completed: {successes}/{len(test_angles)} successful")

if __name__ == "__main__":
    main()
