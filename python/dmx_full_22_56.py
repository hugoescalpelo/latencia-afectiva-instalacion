#!/usr/bin/env python3
from ola.ClientWrapper import ClientWrapper
import time, threading

UNIVERSE = 0
DMX_LOW  = 22   # inclusive
DMX_HIGH = 56   # inclusive
FPS = 44       # ~DMX estándar

class Tx:
    def __init__(self, universe=0, fps=44):
        self.universe = universe
        self.fps = fps
        self.wrapper = ClientWrapper()
        self.client = self.wrapper.Client()
        self.dmx = bytearray(512)
        # fija 255 en 22..56
        for ch in range(DMX_LOW, DMX_HIGH + 1):
            self.dmx[ch - 1] = 255
        self._run = True

    def _tick(self):
        if not self._run:
            return
        self.client.SendDmx(self.universe, bytes(self.dmx), lambda s: None)
        # programa siguiente envío
        self.wrapper.AddEvent(int(1000 / self.fps), self._tick)

    def run(self):
        self._tick()
        while self._run:
            self.wrapper.RunOnce()

    def stop(self):
        self._run = False

if __name__ == "__main__":
    tx = Tx(UNIVERSE, FPS)
    try:
        tx.run()
    except KeyboardInterrupt:
        tx.stop()
