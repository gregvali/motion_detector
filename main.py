import machine
import time
from machine import UART, Pin

sensor_pir = machine.Pin(16, machine.Pin.IN)
led = machine.Pin(15, machine.Pin.OUT)
uart = UART(0, baudrate=9600, tx=Pin(0), rx=Pin(1))
uart.init(bits=8, parity=None, stop=2)
message = "ALARM! Motion detected!"

def pir_handler(pin):
  print(message)   	#print the message.
  uart.write(message)
  for i in range(50):
    led.toggle()
    for j in range(25):
      time.sleep_ms(3)
     
sensor_pir.irq(trigger = machine.Pin.IRQ_RISING, handler = pir_handler)

while True:
    if uart.any(): 
        data = uart.read() 
        if data== b'm':
            led.toggle() 
    time.sleep(1)