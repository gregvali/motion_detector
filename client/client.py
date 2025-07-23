import socket
from constants import SERVER_ADDRESS
import logger

class Client:
    # client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # server_address = ("192.168.1.51", 12345)

    def __init__(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.logger = logger.Logger()
        self.server_address = SERVER_ADDRESS

    def send_request(self, request):
        print(f"Sending Request \"{request}\" to Server: {self.server_address}")
        self.client_socket.sendto(request.encode(), self.server_address)

        print("Recieving Data ...")
        data, addr = self.client_socket.recvfrom(1024)
        data = data.decode()

        print(f"Recieved Data: {data}")
        if self.logger.get_current_data() != data or data == "empty buffer":
            print(f"Logging data to {self.logger.get_filename()}")
            self.logger.log(data)
            self.logger.set_current_data(data)

        return data