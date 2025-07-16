import machine
import time
import server
from machine import Pin, RTC

buffer = "empty buffer"

def create_pir_handler(rtc, led):
    def pir_handler(pin):
        global buffer
        current_time = rtc.datetime()  # Get current time

        time_str = f"{current_time[4]:02}:{current_time[5]:02}:{current_time[6]:02}"  # HH:MM:SS
        date_str = f"{current_time[2]:02}/{current_time[1]:02}/{current_time[0]}"  # DD/MM/YYYY
        
        buffer = "Motion at " + time_str + " on " + date_str # Set buffer
        print(buffer)
        
        for i in range(26): # Blink Buffer
            led.toggle()
            for j in range(25):
                time.sleep_ms(3)
    return pir_handler

def main():
    sensor = machine.Pin(16, machine.Pin.IN)	# Sensor on pin 16
    led = machine.Pin(15, machine.Pin.OUT)		# LED on pin 15

    server_obj = server.Server()
    rtc = RTC()
    server_obj.sync_time(rtc)
    
    handler = create_pir_handler(rtc, led)    
    sensor.irq(trigger = machine.Pin.IRQ_RISING, handler = handler)
    
    while True:
        request = server_obj.recieve_request()
        if request[0] == "SEND DATA":
            server_obj.send_data(buffer, request[1])
            print(f"Sent: {buffer}\n")
        time.sleep(1)

main()