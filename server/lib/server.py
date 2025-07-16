import network
import usocket as socket
import urequests
import json
import time

class Server:
    server_socket = None
    
    def __init__(self):
        with open('lib/config.json') as f:
            config = json.load(f)

        wlan = network.WLAN(network.STA_IF)
        wlan.active(True)
        wlan.connect(config['ssid'], config['ssid_password'])

        print("Connecting to WiFi...")
        while not wlan.isconnected():
            time.sleep(1)
            print("Still trying to connect...")

        print("Connected to Wi-Fi:", wlan.ifconfig())

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_socket.bind((wlan.ifconfig()[0], 12345))
        print("Server is Up and Listening")

    def recieve_request(self):
        print("Waiting for a request from a client ...")
        request, client_address = self.server_socket.recvfrom(1024)
        request_decoded = request.decode()
        print("Client Request: ", request_decoded)
        print("From Client: ", client_address)
        return request_decoded, client_address
    
    def send_data(self, data, client_address):
        self.server_socket.sendto(data.encode(), client_address)
    
    def sync_time(self, rtc):
        with open('lib/config.json') as f:
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
