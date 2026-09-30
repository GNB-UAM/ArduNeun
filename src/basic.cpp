#include <Arduino.h>

#ifndef LED_BUILTIN
#define LED_BUILTIN 21
#endif

void setup() {
    Serial.begin(115200);
    pinMode(LED_BUILTIN, OUTPUT);

    delay(2000);

    Serial.println("================================");
    Serial.println("ESP32-S3-Zero SERIAL TEST");
    Serial.println("================================");
}

void loop() {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(500);

    digitalWrite(LED_BUILTIN, LOW);
    delay(500);
    Serial.println("Hi from the board!");
    delay(1000);
}