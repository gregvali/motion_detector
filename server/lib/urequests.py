import machine
import time
import json
import network
import urequests
from machine import UART, Pin, RTC

sensor_pir = machine.Pin(16, machine.Pin.IN)
led = machine.Pin(15, machine.Pin.OUT)
uart = UART(0, baudrate=9600, tx=Pin(0), rx=Pin(1))
uart.init(bits=8, parity=None, stop=2)
message = "ALERT!"

# Wi-Fi Configuration
def connect_wifi():
    with open('config.json') as f:
        config = json.load(f)

    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(config['ssid'], config['ssid_password'])

    print("Connecting to WiFi...")
    while not wlan.isconnected():
        time.sleep(1)
        print("Still trying to connect...")

    print("Connected to Wi-Fi:", wlan.ifconfig())
    
def sync_time_with_ip_geolocation_api(rtc):
    with open('config.json') as f:
        config = json.load(f)

    date_time_api = config['date_time_api']
    timezone = config['time_zone']

    url = f'http://api.ipgeolocation.io/timezone?apiKey={date_time_api}&tz={timezone}'
    response = urequests.get(url)
    data = response.json()

    print("API Response:", data)

    if 'date_time' in data:
        current_time = data["date_time"]
        print("Current Time String:", current_time)

        if " " in current_time:
            the_date, the_time = current_time.split(" ")
            year, month, mday = map(int, the_date.split("-"))
            hours, minutes, seconds = map(int, the_time.split(":"))

            week_day = data.get("day_of_week", 0)  # Default to 0 if not available
            rtc.datetime((year, month, mday, week_day, hours, minutes, seconds, 0))
            print("RTC Time After Setting:", rtc.datetime())
        else:
            print("Error: Unexpected time format:", current_time)
    else:
        print("Error: The expected data is not present in the response.")

def create_pir_handler(rtc):
    def pir_handler(pin):
        current_time = rtc.datetime()  # Get current time
        # Format the time string
        time_str = f"{current_time[4]:02}:{current_time[5]:02}:{current_time[6]:02}"  # HH:MM:SS
        date_str = f"{current_time[2]:02}/{current_time[1]:02}/{current_time[0]}"  # DD/MM/YYYY
        
        print(message)   	#print the message.
        uart.write(time_str)
        uart.write(date_str)
        
        for i in range(50):
            led.toggle()
            for j in range(25):
                time.sleep_ms(3)
    return pir_handler

def main():
    connect_wifi()
    rtc = RTC()
    sync_time_with_ip_geolocation_api(rtc)
    
    handler = create_pir_handler(rtc)    
    sensor_pir.irq(trigger = machine.Pin.IRQ_RISING, handler = handler)
    
    while True:
        continue

main()