import cupy as cp
import time

# タイム計測開始
start_time = time.time()

# 行列生成
a = cp.random.rand(5000, 5000)
b = cp.random.rand(5000, 5000)

# 行列の積
result = cp.dot(a, b)

# タイム計測終了
end_time = time.time()

print(f"CuPy Time: {end_time - start_time} seconds")
