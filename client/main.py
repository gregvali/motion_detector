import time
import client

DELAY_INTERVAL = 1
REQUESTS = [
  "SEND DATA"
]

def main():
  client_obj = client.Client()
  while True:
    client_obj.send_request(REQUESTS[0])
    time.sleep(DELAY_INTERVAL)

main()