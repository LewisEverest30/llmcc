import concurrent.futures
import time

def worker():
    print(f"开始执行任务")
    time.sleep(5)
    print(f"任务执行结束")
    return

if __name__ == "__main__":
    print("主线程开始")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(worker) for _ in range(3)]

        print("主线程继续执行其他任务")
        for _ in range(3):
            print("主线程正在执行其他操作...")
            time.sleep(1)

    print("主线程结束")