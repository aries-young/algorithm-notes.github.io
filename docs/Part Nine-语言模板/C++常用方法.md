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

## 字符串操作

取子串

```cpp
s.substr(pos, n)     // 截取s中从pos开始（包括0）的n个字符的子串，并返回
s.substr(pos)        // 截取s中从从pos开始（包括0）到末尾的所有字符的子串，并返回
```

查找

```cpp
if (pos != str.npos) // 查找
int pos = str.find("a", 1);  // 从 str[1] 开始查找
```

数转换为对应进制字符串

```cpp
string Hex2Str (int n, int k) {
    string res = "";
    while (n > 0) { 
        if (n % k >= 10) res += (n % k + 'A' - 10);
        else res += (n % k + '0');
        n /= k;
    }
    reverse(res.begin(), res.end());
    return res;
}
```

## queue 操作

```cpp
// 入队
q.push(x);

// 出队
auto x = q.front();
q.pop();
```

## 算法

### 二分法

```cpp
upper_bound() 函数定义在<algorithm>头文件中，用于在指定范围内查找大于目标值的第一个元素。该函数的语法格式有 2 种，分别是：
//查找[first, last)区域中第一个大于 val 的元素。
ForwardIterator upper_bound (ForwardIterator first, ForwardIterator last,
                             const T& val);
//查找[first, last)区域中第一个不符合 comp 规则的元素
ForwardIterator upper_bound (ForwardIterator first, ForwardIterator last,
                             const T& val, Compare comp);
```

`upper_bound()` 底层实现采用的是二分查找的方式，因此该函数仅适用于“已排好序”的序列。注意，这里所说的“已排好序”，并不要求数据完全按照某个排序规则进行升序或降序排序，而仅仅要求 [first, last) 区域内所有令 element < val 或者 comp(val, element）成立的元素都位于不成立元素的前面（其中 element 为指定范围内的元素）

### 前缀和

<img src="https://mmbiz.qpic.cn/sz_mmbiz_jpg/gibkIz0MVqdGFL8VaGGr0vzRcmibenAMtMGcMLfUt26I5y8ibbgA6YiawXP2UGU3ke758gO1GqogeOV9FiarJThypBA/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:50%;" />

```cpp
preSum.push_back(0);
partial_sum(w.begin(), w.end(), back_inserter(preSum));
```

