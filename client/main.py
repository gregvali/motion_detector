
import time
import client
from constants import DELAY_INTERVAL, REQUESTS

if __name__ == "__main__":
  client_obj = client.Client()
  while True:
    client_obj.send_request(REQUESTS[0])
    time.sleep(DELAY_INTERVAL)