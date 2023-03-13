## vector max

```cpp
int max_ele = *max_element(v.begin(), v.end());
```



## 记录时间

```cpp
auto start = chrono::high_resolution_clock::now();
print(1, 2, 3, 4, 5, 6);
auto end = chrono::high_resolution_clock::now();
auto cost_time = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
```

