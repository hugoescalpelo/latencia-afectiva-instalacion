#!/usr/bin/env python3
from ola.ClientWrapper import ClientWrapper
import time

UNIVERSE = 0
DMX_LOW, DMX_HIGH = 22, 56
DELAY = 1.0

wrapper = ClientWrapper()
client = wrapper.Client()

def send_color(r, g, b):
    data = bytearray(512)
    for i in range(DMX_LOW - 1, DMX_HIGH, 3):
        if i + 2 < len(data):
            data[i] = r
            data[i + 1] = g
            data[i + 2] = b
    client.SendDmx(UNIVERSE, data, lambda s: None)  # ← corregido
    print(f"DMX → R:{r} G:{g} B:{b}")

try:
    print("🧪 Iniciando prueba RGB secuencial en Universe 0 (canales 22–56)...")
    while True:
        send_color(255, 0, 0)   # Rojo
        time.sleep(DELAY)
        send_color(0, 255, 0)   # Verde
        time.sleep(DELAY)
        send_color(0, 0, 255)   # Azul
        time.sleep(DELAY)
        send_color(255, 255, 255)  # Blanco
        time.sleep(DELAY)
        send_color(0, 0, 0)     # Apagado
        time.sleep(DELAY)

except KeyboardInterrupt:
    send_color(0, 0, 0)
    print("\n🛑 Prueba detenida. Canales restaurados a 0.")
    wrapper.Stop()
