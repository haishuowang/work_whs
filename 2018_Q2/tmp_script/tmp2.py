from datetime import datetime
import time
import pandas as pd
datetime.now()
stop_date = datetime(*datetime.now().timetuple()[:3], 17, 5)

# while datetime.now() < stop_date:
#     print(1)
#     time.sleep(5)
# print('STOP!')

# while True:
#     if datetime.now() > stop_date:
#         print('STOP')
#         break
#     print(1)
