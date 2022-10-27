# class Solution:
#     def myPow(self, x: float, n: int) -> float:
#         isXNeg, isNNeg = x < 0, n < 0
#         x, n = abs(x), abs(n)
#         if n == 0:
#             return 1
#         if n == 1:
#             return 1 / x if isNNeg else x
#
#         half_ans = self.myPow(x, n // 2)
#
#         ans = half_ans * half_ans if n % 2 == 0 else half_ans * half_ans * x
#
#         if isXNeg and n % 2 == 1:
#             ans = -ans
#         if isNNeg:
#             ans = 1 / ans
#         return ans

import pyautogui
import random
import numpy as np
import time
if __name__ == '__main__':
    pyautogui.FAILSAFE = False
    pyautogui.moveRel(-1000, 250, duration = 1)
    while True:
        random_duration = np.random.randint(1,4)
        random_x = np.random.randint(-250,250)
        random_y = np.random.randint(-250,250)
        random_scroll_dist = np.random.randint(-5,5)
        random_sleep = np.random.randint(0,21)
        pyautogui.moveRel(random_x, random_y, duration=random_duration)
        pyautogui.vscroll(random_scroll_dist)
        time.sleep(random_sleep)