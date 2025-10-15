#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ola.ClientWrapper import ClientWrapper
import array

# Ajusta estos valores si tu universo activo es otro
UNIVERSE = 0
DMX_LOW, DMX_HIGH = 22, 56
FPS = 40            # ~DMX (40–44 Hz está bien)
DELAY_MS = 1000     # cambio de color cada 1 s

# Crea wrapper/cliente con defaults (localhost)
wrapper = ClientWrapper()
client = wrapper.Client()

# Buffer DMX como array('B') — requerido por ola-python (tiene .tobytes())
dmx = array.array('B', [0] * 512)

# Secuencia de colores básicos
colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,255),(0,0,0)]
idx = 0

def apply_rgb(r,g,b):
    """Escribe (r,g,b) en el rango DMX_LOW..DMX_HIGH (formato RGB secuencial)."""
    for i in range(DMX_LOW - 1, DMX_HIGH, 3):
        if i + 2 < len(dmx):
            dmx[i]   = r
            dmx[i+1] = g
            dmx[i+2] = b

def send_frame():
    client.SendDmx(UNIVERSE, dmx, lambda s: None)
    wrapper.AddEvent(int(1000 / FPS), send_frame)

def next_color():
    global idx
    r,g,b = colors[idx]
    print(f"DMX → R:{r} G:{g} B:{b} (Universe {UNIVERSE})")
    apply_rgb(r,g,b)
    idx = (idx + 1) % len(colors)
    wrapper.AddEvent(DELAY_MS, next_color)

if __name__ == "__main__":
    try:
        apply_rgb(255,0,0)
        send_frame()
        next_color()
        wrapper.Run()
    except KeyboardInterrupt:
        print("🛑 Prueba detenida, apagando canales.")
        apply_rgb(0,0,0)
        client.SendDmx(UNIVERSE, dmx, lambda s: None)
