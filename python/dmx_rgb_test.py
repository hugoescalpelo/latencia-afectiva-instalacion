#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de prueba DMX para OLA (Open Lighting Architecture)
Envía secuencias RGB simples al Universe 0.
Confirma funcionamiento del dispositivo Enttec OpenDMX (FT232).
"""

from ola.ClientWrapper import ClientWrapper
import array

# === CONFIGURACIÓN ===
UNIVERSE = 0            # Universo que sí funciona (comprobado con ola_dmxconsole)
DMX_LOW, DMX_HIGH = 22, 56
FPS = 40                # Frecuencia de envío (~40 FPS)
DELAY_MS = 1000         # Cambio de color cada segundo
SERVER = "127.0.0.1"    # Fuerza conexión local (mismo olad)
PORT = 9010

# === CLIENTE ===
wrapper = ClientWrapper(server=SERVER, port=PORT)
client = wrapper.Client()
dmx = array.array('B', [0] * 512)
colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,255), (0,0,0)]
idx = 0

def apply_rgb(r,g,b):
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
