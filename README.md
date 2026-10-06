# ArduNeun

Using [Neun](https://github.com/GNB-UAM/Neun) in Arduino and ESP32 projects: benchmarks and basic C++ scripts for simulating neuron models on microcontrollers, plus Python utilities to visualize and analyze the results.

> Developed by the [GNB-UAM](https://github.com/GNB-UAM).

## Features

- Minimal examples of neuron simulation with Neun on microcontrollers.
- Real-time and periodic real-time versions.
- Fixed-point and `float` variants.
- Performance benchmarks for **ESP32-S3** and **ESP8266**, including tests with different network configurations (single neuron, one connection, full network, and empty).
- Python scripts for live plotting and benchmark comparison.
- Build and upload with [PlatformIO](https://platformio.org/).

## Requirements

- [PlatformIO](https://platformio.org/install) (CLI or VS Code extension).
- [Neun](https://github.com/GNB-UAM/Neun), which must be available at `include/lib/neun`.
- Python 3 (only for the plotting scripts).
- A supported board:
  - ESP32-S3 (configured for `waveshare_esp32_s3_zero`)
  - ESP8266 (configured for `d1_mini_pro`)

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/GNB-UAM/ArduNeun.git
cd ArduNeun

# 2. Add Neun at include/lib/neun
git clone https://github.com/GNB-UAM/Neun.git include/lib/neun
```

> If Neun has its own directory layout, adjust the path so the project's `#include` directives can find it.

## Repository structure

```
ArduNeun/
├── benchmark/             # Benchmark code and data
├── include/               # Headers (lib/neun goes here)
├── lib/                   # Local libraries
├── src/                   # Main programs (one per PlatformIO environment)
├── test/                  # Tests
├── platformio.ini         # Build environments
├── flash_benchmark.sh     # Runs the flash benchmark
├── fast_plot.py           # Plotting utilities
├── mini_plot.py
├── plot_realtime.py       # Real-time plot
├── plot_realtime_periodic.py
├── plot_esp_benchmark.py  # Benchmark plots
└── plot_flash_benchmark.py
```

## PlatformIO environments

Each environment in `platformio.ini` builds a different program from `src/`.

| Environment | Board | Source |
|---|---|---|
| `esp32-s3-basic` | ESP32-S3 | `basic.cpp` |
| `esp32-s3` | ESP32-S3 | `main.cpp` |
| `esp32-s3-float` | ESP32-S3 | `main-float.cpp` |
| `esp32-s3-realtime` | ESP32-S3 | `main_realtime.cpp` |
| `esp32-s3-realtime-periodic` | ESP32-S3 | `real-time-periodic.cpp` |
| `esp32-s3-flash-single` | ESP32-S3 | `main_flash_single_neuron.cpp` |
| `esp32-s3-flash-one-connection` | ESP32-S3 | `main_flash_one_connection.cpp` |
| `esp32-s3-flash-full` | ESP32-S3 | `main_flash_full.cpp` |
| `esp32-s3-flash-empty` | ESP32-S3 | `main_flash_empty.cpp` |
| `esp8266-basic` | ESP8266 | `basic.cpp` |
| `esp8266_flash_single` | ESP8266 | `main_flash_single_neuron.cpp` |
| `esp8266_flash-one-connection` | ESP8266 | `main_flash_one_connection.cpp` |
| `esp8266_flash-full` | ESP8266 | `main_flash_full.cpp` |
| `esp8266_flash-empty` | ESP8266 | `main_flash_empty.cpp` |

Configuration notes:

- ESP32-S3: C++20 (`-std=gnu++20`) using the [pioarduino](https://github.com/pioarduino/platform-espressif32) platform.
- ESP8266: C++2a (`-std=gnu++2a`) using the development branch of the ESP8266 Arduino framework.
- Serial monitor at `115200` baud.

## Usage

### Build and upload a program

```bash
# Build
pio run -e esp32-s3-basic

# Build and upload to the board
pio run -e esp32-s3-basic -t upload

# Open the serial monitor
pio device monitor -b 115200
```

Replace `esp32-s3-basic` with any environment from the table above.

### Benchmarks

The `*-flash-*` environments let you measure behavior under different network configurations. To run them you can use:

```bash
./flash_benchmark.sh
```

Then generate the plots with:

```bash
python plot_flash_benchmark.py
python plot_esp_benchmark.py
```

### Real-time visualization

With a board flashed with `esp32-s3-realtime` (or `esp32-s3-realtime-periodic`) and connected over USB:

```bash
python plot_realtime.py            # real-time mode
python plot_realtime_periodic.py   # periodic real-time mode
```

> You may need to adjust the serial port inside the scripts (for example `/dev/ttyACM0` on Linux or `COMx` on Windows).

## Contributing

Contributions are welcome. Open an issue to discuss major changes, or send a pull request directly.

## License

This project is distributed under the [GPL-3.0](LICENSE) license.
