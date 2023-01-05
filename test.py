import sys
import time
import math
import threading

import single_thread_no_gil
import single_thread_w_gil
import multi_thread

def rad(d):
    return d * 3.1415926535897932384626433832795 / 180.0


def geo_distance_py(lon1, lat1, lon2, lat2, test_cnt):
    distance = 0

    for i in range(test_cnt):
        radLat1 = rad(lat1)
        radLat2 = rad(lat2)
        a = radLat1 - radLat2
        b = rad(lon1) - rad(lon2)
        s = math.sin(a/2)**2 + math.cos(radLat1) * math.cos(radLat2) * math.sin(b/2)**2
        distance = 2 * math.asin(math.sqrt(s)) * 6378 * 1000

    print(distance)
    return distance


def test_single_thread_no_gil(lon1, lat1, lon2, lat2, test_cnt):
    res = single_thread_no_gil.geo_distance(lon1, lat1, lon2, lat2, test_cnt)
    print(res)
    return res

def test_single_thread_w_gil(lon1, lat1, lon2, lat2, test_cnt):
    res = single_thread_w_gil.geo_distance(lon1, lat1, lon2, lat2, test_cnt)
    print(res)
    return res

if __name__ == "__main__":
    threads = []
    test_cnt = 10000000
    test_type = sys.argv[1]
    thread_cnt = int(sys.argv[2])
    start_time = time.time()

    if test_type == 'multi_thread':
        t = multi_thread.geo_distance(113.973129, 22.599578, 114.3311032, 22.6986848, test_cnt);
    else:
        for i in range(thread_cnt):
            if test_type == 'python':
                t = threading.Thread(target=geo_distance_py,
                    args=(113.973129, 22.599578, 114.3311032, 22.6986848, test_cnt,))
            elif test_type == 'single_thread_no_gil':
                t = threading.Thread(target=test_single_thread_no_gil,
                    args=(113.973129, 22.599578, 114.3311032, 22.6986848, test_cnt,))
            elif test_type == 'single_thread_w_gil':
                t = threading.Thread(target=test_single_thread_w_gil,
                    args=(113.973129, 22.599578, 114.3311032, 22.6986848, test_cnt,))
            else:
                raise Exception("unknown function")
            threads.append(t)
            t.start()

        for thread in threads:
            thread.join()

    print('calc time = %d' % int((time.time() - start_time) * 1000))
