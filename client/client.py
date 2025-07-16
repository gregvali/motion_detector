import socket

class Client:
    # client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # server_address = ("192.168.1.51", 12345)

    def __init__(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_address = ("192.168.1.51", 12345)

    def send_request(self, request):
        print(f"Sending Request \"{request}\" to Server: {self.server_address}")
        self.client_socket.sendto(request.encode(), self.server_address)

        print("Recieving Data ...")
        data, addr = self.client_socket.recvfrom(1024)

        print("Recieved Data:", data.decode())
        return data.decode()