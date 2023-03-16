import threading
import random
import time

lock = threading.Lock()

def print_ticker(ticker, period_ms):
    while(1):
        price = random.uniform(0, 1)
        lock.acquire()
        print(f"{ticker} {price}")
        lock.release()
        time.sleep(period_ms/1000)

def emulate_ticker_rate_updates(items):
    for item in items:
        thread = threading.Thread(target=print_ticker, args=(item))
        thread.setDaemon(True)
        thread.start()
    while(1):
        pass

emulate_ticker_rate_updates([('TYM2', 30), ('FVM2', 50), ('WNM2', 80)])