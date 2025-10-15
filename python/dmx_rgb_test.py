#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ola.ClientWrapper import ClientWrapper
import array

UNIVERSE = 0             # <-- AJUSTA si tu consola web usa otro universe
DMX_LOW, DMX_HIGH = 22, 56
FPS = 44                 # DMX típico ~44 Hz
DELAY_MS = 1000          # cambio de color cada 1.0 s

wrapper = ClientWrapper()
client = wrapper.Client()

# Buffer con .tobytes()
dmx = array.array('B', [0] * 512)

colors = [
    (255,   0,   0),  # rojo
    (  0, 255,   0),  # verde
    (  0,   0, 255),  # azul
    (255, 255, 255),  # blanco
    (  0,   0,   0),  # off
]
color_idx = 0

def apply_rgb(r, g, b):
    """Escribe (r,g,b) secuencial en el rango DMX_LOW..DMX_HIGH."""
    for i in range(DMX_LOW - 1, DMX_HIGH, 3):
        if i + 2 < len(dmx):
            dmx[i]     = r
            dmx[i + 1] = g
            dmx[i + 2] = b

def send_frame():
    """Envía un frame y vuelve a programarse a ~FPS."""
    client.SendDmx(UNIVERSE, dmx, lambda s: None)
    wrapper.AddEvent(int(1000 / FPS), send_frame)

def next_color():
    """Cambia el color activo y vuelve a programarse cada DELAY_MS."""
    global color_idx
    r, g, b = colors[color_idx]
    apply_rgb(r, g, b)
    print(f"DMX → R:{r} G:{g} B:{b}")
    color_idx = (color_idx + 1) % len(colors)
    wrapper.AddEvent(DELAY_MS, next_color)

def main():
    # Programa timers y arranca el loop de eventos de OLA
    apply_rgb(255, 0, 0)  # arranca en rojo
    send_frame()
    next_color()
    try:
        wrapper.Run()
    except KeyboardInterrupt:
        apply_rgb(0, 0, 0)
        client.SendDmx(UNIVERSE, dmx, lambda s: None)

if __name__ == "__main__":
    main()
