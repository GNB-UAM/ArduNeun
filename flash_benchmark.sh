echo "ESP32-s3_Basic w/o Neun" > flash_report.log
~/.platformio/penv/bin/pio run -e esp32-s3-flash-empty -t size | grep "Calculating size" -A 2 >> flash_report.log
echo "ESP8266_Basic w/o Neun" > flash_report.log
~/.platformio/penv/bin/pio run -e esp32-s3-flash-empty -t size | grep "Calculating size" -A 2 >> flash_report.log

echo "ESP32-s3_Only one neuron" >> flash_report.log
~/.platformio/penv/bin/pio run -e esp32-s3-flash-single -t size | grep "Calculating size" -A 2 >> flash_report.log
echo "ESP8266_Only one neuron" >> flash_report.log
~/.platformio/penv/bin/pio run -e esp32-s3-flash-single -t size | grep "Calculating size" -A 2 >> flash_report.log

echo "ESP32-s3_2 neurons 1 synapse" >> flash_report.log
~/.platformio/penv/bin/pio run -e esp32-s3-flash-one-connection -t size | grep "Calculating size" -A 2 >> flash_report.log
echo "ESP8266_2 neurons 1 synapse" >> flash_report.log
~/.platformio/penv/bin/pio run -e esp32-s3-flash-one-connection -t size | grep "Calculating size" -A 2 >> flash_report.log

echo "ESP32-s3_Complete Neun" >> flash_report.log
~/.platformio/penv/bin/pio run -e esp32-s3-flash-full -t size | grep "Calculating size" -A 2 >> flash_report.log

echo "ESP8266_flash-Complete Neun" >> flash_report.log
~/.platformio/penv/bin/pio run -e esp8266_flash-full -t size | grep "Calculating size" -A 2 >> flash_report.log