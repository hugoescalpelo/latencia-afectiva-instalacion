#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ola.ClientWrapper import ClientWrapper
import time
import array

UNIVERSE = 0
DMX_LOW, DMX_HIGH = 22, 56      # Tu rango activo
FPS = 44                        # Reenvío continuo (~DMX)
DELAY = 1.0                     # Segundos entre colores

# Buffer DMX como array('B') para que tenga .tobytes()
dmx = array.array('B', [0] * 512)

wrapper = ClientWrapper()
client = wrapper.Client()

# Estado de color actual que iremos reenviando a FPS constante
_current_rgb = (0, 0, 0)

def _apply_rgb_to_range(r, g, b):
    """Escribe el color (r,g,b) en el rango DMX_LOW..DMX_HIGH (RGB secuencial)."""
    for i in range(DMX_LOW - 1, DMX_HIGH, 3):
        if i + 2 < len(dmx):
            dmx[i]     = r
            dmx[i + 1] = g
            dmx[i + 2] = b

def _tick():
    """Envía el frame DMX actual y reprograma el siguiente envío."""
    client.SendDmx(UNIVERSE, dmx, lambda s: None)
    wrapper.AddEvent(int(1000 / FPS), _tick)

def set_color(r, g, b):
    global _current_rgb
    _current_rgb = (r, g, b)
    _apply_rgb_to_range(r, g, b)
    # Nota: no mandamos aquí; lo hace _tick() continuamente

def run_sequence():
    print("🧪 Iniciando prueba RGB continua en Universe 0 (canales 22–56, 44 FPS)...")
    while True:
        set_color(255, 0, 0)     # Rojo
        print("DMX → R:255 G:0 B:0")
        time.sleep(DELAY)

        set_color(0, 255, 0)     # Verde
        print("DMX → R:0 G:255 B:0")
        time.sleep(DELAY)

        set_color(0, 0, 255)     # Azul
        print("DMX → R:0 G:0 B:255")
        time.sleep(DELAY)

        set_color(255, 255, 255) # Blanco
        print("DMX → R:255 G:255 B:255")
        time.sleep(DELAY)

        set_color(0, 0, 0)       # Apagado
        print("DMX → R:0 G:0 B:0")
        time.sleep(DELAY)

if __name__ == "__main__":
    try:
        # Arranca el envío continuo
        _tick()
        # Corre la secuencia (cambia el color cada DELAY, pero el stream sigue a 44 FPS)
        run_sequence()
    except KeyboardInterrupt:
        set_color(0, 0, 0)
        client.SendDmx(UNIVERSE, dmx, lambda s: None)
        print("\n🛑 Prueba detenida. Canales restaurados a 0.")
