from datetime import datetime
import time
datetime.now()
stop_date = datetime(*datetime.now().timetuple()[:3], 15, 3)

while datetime.now() < stop_date:
    print(1)
    time.sleep(5)
print('STOP!')