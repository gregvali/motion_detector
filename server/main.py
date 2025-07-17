
import time
import server
import buffer
from machine import Pin, RTC

def create_pir_handler(rtc, buffer):
    def pir_handler(pin):
        current_time = rtc.datetime()  # Get current time

        time_str = f"{current_time[4]:02}:{current_time[5]:02}:{current_time[6]:02}"  # HH:MM:SS
        date_str = f"{current_time[2]:02}/{current_time[1]:02}/{current_time[0]}"  # DD/MM/YYYY
        
        buffer.setb("Motion at " + time_str + " on " + date_str) # Set buffer
        print(buffer.getb())
    return pir_handler

if __name__ == "__main__":
    sensor = machine.Pin(16, machine.Pin.IN)	# Sensor on pin 16

    buffer = buffer.Buffer()
    server_obj = server.Server()
    rtc = RTC()
    server_obj.sync_time(rtc)
    
    handler = create_pir_handler(rtc, buffer)    
    sensor.irq(trigger = machine.Pin.IRQ_RISING, handler = handler)
    
    while True:
        request = server_obj.recieve_request()
        if request[0] == "SEND DATA":
            server_obj.send_data(buffer.getb(), request[1])
            print(f"Sent: {buffer.getb()}\n")
        time.sleep(1)
