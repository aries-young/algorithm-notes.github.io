## 870. 优势洗牌

**题目描述**

[870. 优势洗牌](https://leetcode.cn/problems/advantage-shuffle/)

**解法**

这道题其实就是田忌赛马，只要比不过就拿个垫底的去送人头就是，唯一的技巧就在双指针处理送人头

```cpp
class Solution {
public:
    vector<int> advantageCount(vector<int>& nums1, vector<int>& nums2) {
        int n =  nums1.size();
        vector<pair<int, int>> hash;
        vector<int> res(n, 0) ;
        for (int i = 0; i < n; i++){
            hash.push_back(pair<int, int>(nums2[i], i));
        }
        sort(hash.begin(), hash.end(),[](const pair<int,int>& a1, const pair<int,int>& a2){
            return a1.first > a2.first;
        });
        sort(nums1.begin(), nums1.end());
        int left = 0, right = n - 1;
        for (auto a: hash){
            int val = a.first, i = a.second;
            if (val < nums1[right]){
                res[i] = nums1[right];
                right--;
            }
            else {
                res[i] = nums1[left];
                left++;
            }
        }
        return res;
    }
};
```

## 1652. 拆炸弹

**题目描述**

[1652. 拆炸弹](https://leetcode.cn/problems/defuse-the-bomb/)

**解法**

这道题其实就是一个 $$2 * n$$ 的前缀和，对于循环数组我们一般都会想到 $$2 * n$$ 的处理，但是，$$2 * n$$ 没必要在原数组操作，而是利用 $$\text{mod}$$ 运算生成一个 $$2 * n$$  大小的前缀和数组

此时，若有 $$k < 0$$，先将位置 $$i$$ 往后进行 $$n$$ 个偏移（即位置 $$i + n$$），随后可知对应前缀和的区为 $$[i + n - 1, i + n + k - 1]$$

```cpp
class Solution {
public:
    vector<int> decrypt(vector<int>& code, int k) {
        int n = code.size();
        vector<int> res(n, 0);
        if(k == 0) return res;
        vector<int> presum(2 * n + 5, 0);
        for (int i = 1; i <= 2 * n; i++)
            presum[i] = presum[i - 1] + code[(i - 1) % n];
        for (int i = 1; i <= n; i++){
            if(k < 0) res[i - 1] = presum[i + n - 1] - presum[i + n + k - 1];
            else res[i - 1] = presum[i + k] - presum[i];
        }
        return res;
    }
};
```

## 316. 去除重复字母

**题目描述**

[316. 去除重复字母](https://leetcode.cn/problems/remove-duplicate-letters/)

**解法**

题目有三个要求，

- 去重
- 字符出现的相对顺序不变
- 在所有符合上一条要求的去重字符串中，字典序最小的作为最终结果

意思就是说，输入字符串 `s = "babc"`，去重且符合相对位置的字符串有两个，分别是 `"bac"` 和 `"abc"`，但是我们的算法得返回 `"abc"`，因为它的字典序更小

前两个要求好解决，用栈就可以实现，这时候我们可以得到 `"bac"` 

```cpp
class Solution {
public:
    string removeDuplicateLetters(string s) {
        int n = s.size();
        stack<char> st;
        vector<int> visited(256, 0);

        for (auto c: s){
            if (visited[c]) continue;
            st.push(c);
            visited[c] = 1;
        }
        string res = "";
        while(!st.empty()){
            char tmp = st.top();
            st.pop();
            res += tmp;
        }
        reverse(res.begin(), res.end());
        return res;
    }
};
```

要得到  `"bac"`  呢？

在向栈 `st` 中插入字符 `'a'` 的这一刻，我们的算法需要知道，字符 `'a'` 的字典序和之前的两个字符 `'b'` 和 `'c'` 相比，谁大谁小？

**如果当前字符 `'a'` 比之前的字符字典序小，就有可能需要把前面的字符 pop 出栈，让 `'a'` 排在前面，对吧**？但是这里还是有一个问题

假设 `s = "bcac"`，按照刚才的算法逻辑，返回的结果是 `"ac"`，而正确答案应该是 `"bac"`，分析一下这是怎么回事？

很容易发现，因为 `s` 中只有唯一一个 `'b'`，即便字符 `'a'` 的字典序比字符 `'b'` 要小，字符 `'b'` 也不应该被 pop 出去

那么，**在 `st.top() > c` 时才会 pop 元素，这时候应该分两种情况：**

- 情况一，如果 `st.top()` 这个字符之后还会出现，那么可以把它 pop 出去，反正后面还有嘛，后面再 push 到栈里，刚好符合字典序的要求
- 情况二，如果`st.top()`这个字符之后不会出现了，前面也说了栈中不会存在重复的元素，那么就不能把它 pop 出去，否则你就永远失去了这个字符。

回到 `s = "bcac"` 的例子，插入字符 `'a'` 的时候，发现前面的字符 `'c'` 的字典序比 `'a'` 大，且在 `'a'` 之后还存在字符 `'c'`，那么栈顶的这个 `'c'` 就会被 pop 掉。

while 循环继续判断，发现前面的字符 `'b'` 的字典序还是比 `'a'` 大，但是在 `'a'` 之后再没有字符 `'b'` 了，所以不应该把 `'b'` pop 出去

**那么关键就在于，如何让算法知道字符 `'a'` 之后有几个 `'b'` 有几个 `'c'` **

```cpp
class Solution {
public:
    string removeDuplicateLetters(string s) {
        int n = s.size();
        stack<char> st;
        vector<int> cnt(256, 0);
        vector<int> visited(256, 0);

        for (auto c: s) cnt[c]++;
        for (auto c: s){
            cnt[c]--;

            if (visited[c]) continue;
            while(!st.empty() && st.top() > c){
                if (cnt[st.top()] == 0) break;
                visited[st.top()] = 0;
                st.pop();
            }
            st.push(c);
            visited[c] = 1;
        }
        string res = "";
        while(!st.empty()){
            char tmp = st.top();
            st.pop();
            res += tmp;
        }
        reverse(res.begin(), res.end());
        return res;
    }
};
```

## 1784. 检查二进制字符串字段

**题目描述**

[1784. 检查二进制字符串字段](https://leetcode.cn/problems/check-if-binary-string-has-at-most-one-segment-of-ones/)

**解法一**

三行，直接模拟

```cpp
class Solution {
public:
    bool checkOnesSegment(string s) {
        int ans = 1;
        for (int i = 1; i < s.length();i++){
            if (s[i] == '1' && s[i - 1] != '1') ans++;
        }
        return ans < 2;
    }
};
```

**解法二**

还有一种解法，一行，但是 比较巧一点，题目所描述的情况的都包含 `01` 串，同时，不包含 `01` 串的二进制字符串有且仅有上面两种，所以我们在 `s` 中寻找是否存在 `01` 串即可

```cpp
class Solution {
public:
    bool checkOnesSegment(string s) {
        return s.find("01") == string::npos;
    }
};
```

## 1250. 检查「好数组」

**题目描述**

[1250. 检查「好数组」](https://leetcode.cn/problems/check-if-it-is-a-good-array/description/)

**解法**

### 【裴蜀定理】

![在这里插入图片描述](https://img-blog.csdnimg.cn/27d1edb1906e4cde866cb6c5264ebe2d.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/a495ef7c8a554237953b17152cb88751.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/296b55ba22e64e50b27eeb708b1e8c63.png)

```cpp
class Solution {
public:
    bool isGoodArray(vector<int>& nums) {
        int flag = nums[0];
        for (int i = 1; i < nums.size() && flag != 1; i++) {
            flag = gcd(flag, nums[i]);
        }
        return flag == 1;
    }
    int gcd(int num1, int num2) {
        while (num2 != 0) {
            int tmp = num1;
            num1 = num2;
            num2 = tmp % num2;
        }
        return num1;
    }
};
```



## 1139. 最大的以 1 为边界的正方形

**题目描述**

[1139. 最大的以 1 为边界的正方形](https://leetcode.cn/problems/largest-1-bordered-square/description/)

**解法一：二维前缀和**

二维前缀和的运用很巧妙，在网格中我们可以得到任意一个正方形的和，怎么判断这个正方形是边界全为 1 的呢？正方形里面一个取小一圈的正方形，两个正方形区域和的差值应该正好等于边界上 1 的和

<img src="https://img-blog.csdnimg.cn/d2e1ad1056da447983b23def6c0ce504.png" alt="在这里插入图片描述" style="zoom:20%;" />

```cpp
class Solution {
public:
    int largest1BorderedSquare(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        vector<vector<int>> preSum(m + 1, vector<int>(n + 1));

        int ans = 0;
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                preSum[i][j] = preSum[i - 1][j] + preSum[i][j - 1] - preSum[i - 1][j - 1] + grid [i - 1][j - 1];
                if (grid[i - 1][j - 1] != 1) continue;
                for (int k = 1; i - k >= 0 && j - k >=0; k++) {
                    int s1 = preSum[i][j] - preSum[i - k][j] -preSum[i][j - k] + preSum[i - k][j - k];
                    int s2 = preSum[i - 1][j - 1] - preSum[i - k + 1][j - 1] - preSum[i - 1][j - k + 1] + preSum[i - k + 1][j - k + 1];
                    int t = k - 2;
                    if (s1 - s2 == k * k - t * t) ans = max(ans, k * k);

                }
            }
        }
        return ans;
    }
};
```

## 6360. 最小无法得到的或值

**题目描述**

[6360. 最小无法得到的或值](https://leetcode.cn/problems/minimum-impossible-or/description/)

**解法：**

或运算只能将二进制中的 0 变成 1，因此，如果 $2^k$ 是可表达的，那么 `nums` 中一定存在 $2^k$

```cpp
class Solution {
public:
    int minImpossibleOR(vector<int>& nums) {
        map<int, int> cnt;
        for(auto i: nums) cnt[i]++;
        for(int i = 1; ; i <<= 1) {
            if (!cnt[i]) return i;
        }
    }
};
```

## 2570. 合并两个二维数组 - 求和法

**题目描述**

[2570. 合并两个二维数组 - 求和法](https://leetcode.cn/problems/merge-two-2d-arrays-by-summing-values/description/)

**解法：双指针**

```cpp
class Solution {
public:
    vector<vector<int>> mergeArrays(vector<vector<int>>& nums1, vector<vector<int>>& nums2) {
        int p1 = 0, p2 = 0;
        int n = nums1.size(), m = nums2.size();
        vector<vector<int>> ans;

        while (p1 < n && p2 < m) {
            if(nums1[p1][0] == nums2[p2][0]) {
                ans.push_back({nums1[p1][0], nums1[p1][1] + nums2[p2][1]});
                p1++;
                p2++;
            }
            else if (nums1[p1][0] < nums2[p2][0]) {
                ans.push_back(nums1[p1]);
                p1++;
            }
            else {
                ans.push_back(nums2[p2]);
                p2++;
            }
        }

        if(p1 != n) {
            while(p1 < n) {
                ans.push_back(nums1[p1]);
                p1++;
            }
        }
        if(p2 != m) {
            while (p2 < m) {
                ans.push_back(nums2[p2]);
                p2++;
            }
        }
        return ans;
    }
};
```

## 2571. 将整数减少到零需要的最少操作数

**题目描述**

[2571. 将整数减少到零需要的最少操作数](https://leetcode.cn/problems/minimum-operations-to-reduce-an-integer-to-0/description/)

**解法：**

先说几个位操作的技巧

### 【lowbit】 

`lowbit(x)` 取 x 在二进制表示下最低位的 1 以及它后面的 0 构成的数值

```cpp
lowbit(x) = x & (-x)
x = x - lowbit(x); // 去掉最低位的 1
```

基于 lowbit 的运算，我们可以贪心考虑将连续的 1 加上 lowbit，不连续的 1 减去 lowbit，判断连续的 1 代码如下

```cpp
x & (lowbit << 1) // 判断连续的 1
// 1010 lowbit: 10 1010 & 0100 = 0000
// 1110 lowbit: 10 1110 & 0100 = 0100
```

位运算 x 是否是 2 的幂次

```cpp
x & (x - 1)
```

完整代码如下

```cpp
class Solution {
public:
    int minOperations(int n) {
        int ans  = 1;
        while (n & (n - 1)) {
            int lowbit = n & (-n);
            if (n & (lowbit << 1)) n += lowbit;
            else n -= lowbit;
            ans++;
        }
        return ans;
    }
};
```

 ## 2572. 无平方子集计数

**题目描述**

[2572. 无平方子集计数](https://leetcode.cn/problems/count-the-number-of-square-free-subsets/description/)

**解法：状态压缩 DP**

这道题要注意下提示 `1 <= nums[i] <= 30`，前 30 的数，只有 10 个质数 `vector<int> prime = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}`。换句话说，`nums[i]` 是平方因子数，只可能是 `prime` 里面的平方数够成的，如 28 =2 * 2 * 7

维护 $f(i, m)$ 表示前 $i$ 个数的子集中，质因数出现情况用二进制数 $m$ 表示的子集的个数。注意是用二进制数表示的质因数组合，即 0000110100 表示组合中出现 5, 11, 13

那么状态转移方程即是
$$
f[i][j | msk] = (f[i][j | msk] + f[i - 1][j])
$$
同时注意，要求 $msk\ \&\ msk' = 0$，因为出现过的质因数不能再次出现，否则子集乘积会变成平方数

最后，优化一下空间复杂度，由于第 $i$ 轮的 dp 状态和第 $i-1$ 轮相比，新一轮的方案 $j$ 中只满足 `j & msk == msk` 的方案才会增加，其他方案的总数不变，而且增加的量全部来源于不变的量，所以可以直接取消 `f` 的第一维,代码其他部分可以完全不变

```cpp
class Solution {
private:
    const int MAXK = 10;
    const int MOD = 1e9 + 7;
    vector<int> prime = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
    bool check(int x) {
        for (int i = 0; i < MAXK; i++) {
            if (x % (prime[i] * prime[i]) == 0) return true;
        }
        return false;
    }
public:
    int squareFreeSubsets(vector<int>& nums) {
        long long f[1 << MAXK];
        memset(f, 0, sizeof(f));
        f[0] = 1;

        int n = nums.size();
        for (int i = 1; i <= n; i++) {
            int x = nums[i - 1];
            if (check(x)) continue;
            // 计算第 i 个数的质因数分解
            int msk = 0; 
            for (int j = 0; j < MAXK;  j++) {
                if (x % prime[j] == 0) msk |= 1 << j;
            }
            // 把第 i 个数加入子集的方案数
            for (int j = 0; j < (1 << MAXK); j++) {
                if ((j & msk) == 0) f[j | msk] = (f[j | msk] + f[j]) % MOD;
            }
        }
        long long ans = 0;
        for (int j = 0; j < (1 << MAXK); j++) ans = (ans + f[j]) % MOD;
        // 非空集合
        ans = (ans - 1 + MOD) % MOD;
        return ans;
    }
};
```

## 2573. 找出对应 LCP 矩阵的字符串

**题目描述**

[2573. 找出对应 LCP 矩阵的字符串](https://leetcode.cn/problems/find-the-string-with-lcp/description/)

**解法：**

首先，要知道到底什么是 LCP 矩阵，比如说示例 1 的字符串 abab，`s[0...n] = abab` 和 `s[2...n] =ab`，所以 `lcp[0][2] = 2`

如果要构造字典序最小的 `s`，那么肯定从 `s[0] = a` 开始填

那么，`s[0] = a` 时，根据 `lcp[0]` 还有哪些是 `a` 哪些不能是 `a` 呢？

根据 LCP 的定义，`lcp[0][i]>0` 的一定是 `a`，`lcp[0][i]=0` 的一定不是 `a`

现在流程很明确了

- 先填 `a`， 然后根据 LCP 的一行把填 `a`  的位置补满
- 然后填 `b`，根据LCP 的一行把填 `b`  的位置补满
- 如此循环，直到 26 个字母用完

这样，构造的部分就结束了。但是，有两种情况，一是构造出来的字符串还有位置没填上字符，二是构造的字符串的 LPC 和提供的 LPC 不一致，原因很简单可能输入的 LCP 是一个不合法的矩阵

情况一的检查很容易

情况二的检查利用 LCP 的DP 转移方程，首先明确边界条件 `(lcp[i][n−1],lcp[n−1][i])`，转移方程如下

$$
lcp[i][j]=lcp[i+1][j+1]+1\ 若\ s[i]==s[j]\ 否则\ lcp[i][j]=0\
$$

```cpp
class Solution {
public:
    string findTheString(vector<vector<int>>& lcp) {
        int i = 0, n = lcp.size();
        string s(n, '#');
        for (char c = 'a'; c <= 'z'; c++) {
            while (i < n && s[i] != '#') i++;
            if (i == n) break;
            for (int j = i; j < n; j++) {
                if (lcp[i][j]) s[j] = c;
            }
        }
        for (auto c: s) {
            if (c == '#') return "";
        }
        for (int i = n - 1; i >=0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                int actualLCP = s[i] != s[j] ? 0 : i == n - 1 || j == n - 1 ? 1 : lcp[i + 1][j + 1] + 1;
                if (lcp[i][j] !=actualLCP) return "";
            }
        }
        return s;
    }
};
```

## 89. 格雷编码

**题目描述**

[89. 格雷编码](https://leetcode.cn/problems/gray-code/description/)

**解法：镜射排列**

![镜面.png](https://pic.leetcode-cn.com/7a14e1e43158a6ba9435988faafa59464608636b6f9e8ee07ee74b46f75a5995-%E9%95%9C%E9%9D%A2.png)

```cpp
class Solution {
public:
    vector<int> grayCode(int n) {
        vector<int> ans;
        ans.push_back(0);
        for (int i = 0; i < n; i++) {
            for (int j = ans.size() - 1; j >= 0; j--) {
                ans.push_back(ans[j] ^ (1<<i));
            }
        }
        return ans;
    }
};
```

## 1238. 循环码排列

**题目描述**

[1238. 循环码排列](https://leetcode.cn/problems/circular-permutation-in-binary-representation/description/)

**解法**

先看 89 题，再看 1238 就很简单了，先按照格雷码生成数组，然后再根据 start 循环排列一下数组就好

```cpp
class Solution {
public:
    vector<int> circularPermutation(int n, int start) {
        int s = 0;
        vector<int> gray_code = {0};
        for (int i = 0; i < n; i++) {
            for (int j = gray_code.size() - 1; j >= 0; j--){
                int num = gray_code[j] ^ (1 << i);
                if (num == start) s = gray_code.size();
                gray_code.push_back(num);
            }
        }
        int length = gray_code.size();
        vector<int> ans;
        for (int i = s; i < s + length; i++) {
            ans.push_back(gray_code[i % length]);
        }
        return ans;
    }
};
```

## 6368. 找出字符串的可整除数组

**题目描述**

[6368. 找出字符串的可整除数组](https://leetcode.cn/problems/find-the-divisibility-array-of-a-string/description/)

**解法**

如果单纯利用 $$num = num * 10 + d$$ 去对 $$m$$ 取余的话，$$num$$ 会远大于 long long 可以表示的范围，那么这里就有一个技巧了，$$num = (num*10+d)\ \text{mod}\ m$$

```cpp
class Solution {
public:
    vector<int> divisibilityArray(string word, int m) {
        long long num = 0;
        vector<int> ans(word.size(), 0);
        for (int i = 0; i < word.size(); i++) {
            num = num * 10 + (word[i] - '0');
            num = num % m;
            if (num == 0) ans[i] = 1;
        }
        return ans;
    }
};
```

## 6367. 求出最多标记下标

**题目描述**

[6367. 求出最多标记下标](https://leetcode.cn/problems/find-the-maximum-number-of-marked-indices/description/)

**解法：二分法**

先明确题目两个地方，一是标记过的一对就不可以再和其他数配对标记，二是一组标记包含两个小标，所以答案时可标记组数 \* 2

将数组排序后就可以分为两部分，且如果 $$2*nums[0] \leq nums[j]$$，那么下一组标记 $$nums[1]$$ 只能从下标大于 $$j$$ 的数去找

```cpp
class Solution {
public:
    int maxNumOfMarkedIndices(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int i = 0, n = nums.size();
        for (int j = (n + 1) / 2; j < nums.size(); j++) {
            if (nums[i] * 2 <= nums[j]) {
                i++;
            }
        }
        return i * 2;
    }
};
```

## 6366. 在网格图中访问一个格子的最少时间

**题目描述**

[6366. 在网格图中访问一个格子的最少时间](https://leetcode.cn/problems/minimum-time-to-visit-a-cell-in-a-grid/description/)

**解法：Dijkstra**

定义 $$dis[i][j]$$ 为到达 $$(i, j)$$ 的最小时间，如果没有别的约束，那么每条边的边权可以视作 1，根据 Dijkstra 算法就可以算出答案

根据题意，$$dis[i][j]$$ 至少要是 $$grid[i][j]$$

但是这个题目中有一个比较神奇的地方，如果当前时间还不满足，可以反复横跳来凑够时间

那么在可以反复横跳的情况下，到达一个格子的时间的奇偶性是不变的，那么 $$dis[i][j]$$ 应当与 $$i+j$$ 的奇偶性相同

算上上面两个约束，就可以计算出正确结果了

```cpp
class Solution {
private:
    vector<vector<int>> dirs = { {-1, 0}, {1, 0}, {0, -1}, {0, 1} };
public:
    int minimumTime(vector<vector<int>>& grid) {
        int n = grid.size(), m = grid[0].size();
        if (grid[0][1] > 1 && grid[1][0] > 1) return -1;
        vector<vector<int>> dis(n, vector<int>(m, INT_MAX));
        dis[0][0] = 0;
        priority_queue<tuple<int, int, int>, vector<tuple<int, int, int>>, greater<>> pq;
        pq.emplace(0, 0, 0);
        for (;;) {
            auto[d, i, j] = pq.top();
            pq.pop();
            if (i == n - 1 && j == m - 1) return d;
            for (auto &q:dirs) {
                int x = i + q[0], y = j + q[1];
                if (0 <= x && x < n && 0 <= y && y < m) {
                    int nd = max(d + 1, grid[x][y]);
                    nd += (nd - x - y) % 2;
                    if (nd  < dis[x][y]) {
                        dis[x][y] = nd;
                        pq.emplace(nd, x, y);
                    }
                }
            }
        }

    }
};
```

## 146. LRU 缓存机制

**题目描述**

[146. LRU 缓存机制](https://leetcode-cn.com/problems/lru-cache/)

**解法**

LRU 全称 Least Recently Used，那么在 get 和 put 都要注意对队列进行更新

<img src="https://mmbiz.qpic.cn/sz_mmbiz_jpg/gibkIz0MVqdHkcPqjzoDYrtO88MrDuPB5SzcpCichTh2Rxd7qJKS6bBTzEX5ldI6r9H9NJltgPMHnrOqKibe1eCdw/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:50%;" />

LRU 采用哈希链表的数据结构，哈希中存储关键字和关键字在双向链表中的位置

<img src="https://mmbiz.qpic.cn/sz_mmbiz_jpg/gibkIz0MVqdHkcPqjzoDYrtO88MrDuPB5TN0Pr0iax20pqyeWibyjDtapiaCaJChucMTjhlibwyHBToIyaLqkr2Tdxw/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:50%;" />

```cpp
class LRUCache {
public:
    LRUCache(int capacity):capacity_(capacity){

    }
    
    int get(int key) {
        if(hash_.find(key)==hash_.end()) return -1;
        else
        {
            // 把刚访问的(key, value)放在链表头
            int value = hash_[key]->second;
            ls_.erase(hash_[key]);
            ls_.push_front(make_pair(key, value));
            // 更新(key, value)在链表中的位置
            hash_[key] = ls_.begin();
            return value;
        }
    }
    
    void put(int key, int value) {
        if(hash_.find(key)!=hash_.end()) ls_.erase(hash_[key]);
        if(ls_.size()>=capacity_)
        {
            hash_.erase(ls_.back().first);
            ls_.pop_back();
        }
        ls_.push_front(make_pair(key, value));
        hash_[key] = ls_.begin();
    }

private:
    int capacity_;
    list<pair<int, int>> ls_;
    unordered_map<int, list<pair<int, int>>::iterator> hash_;
};

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache* obj = new LRUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */
```

## 460. LFU缓存

**题目描述**

[460. LFU缓存](https://leetcode-cn.com/problems/lfu-cache/)

**解法一：哈希表 +  AVL 树**

在 C++ 中 AVL 树采取 `STL::set` 实现

时间复杂度： get 时间复杂度 $O(\log n)$，put 时间复杂度 $O(\log n)$，操作的时间复杂度瓶颈在于平衡二叉树的插入删除均需要 $O(\log n)$ 的时间

```cpp
struct Node {
    int cnt;
    int time;
    int key, value;
    Node(int _cnt, int _time, int _key, int _value) : cnt(_cnt), time(_time), key(_key), value(_value) {}
    bool operator < (const Node& rhs) const {
        return cnt == rhs.cnt ? (time < rhs.time) : (cnt < rhs.cnt);
    }
};

class LFUCache {
private:
    int capacity, time;
    unordered_map<int, Node> key_table;
    set<Node> S;
public:
    LFUCache(int _capacity) {
        capacity = _capacity;
        time = 0;
        key_table.clear();
        S.clear();
    }
    
    int get(int key) {
        if (capacity == 0) return -1;
        auto it = key_table.find(key);
        if (it == key_table.end()) return -1;
        Node cache = it->second;
        S.erase(cache);
        cache.cnt += 1;
        cache.time = ++time;
        S.insert(cache);
        it->second = cache;
        return cache.value;
    }
    
    void put(int key, int value) {
        if (capacity == 0) return;
        auto it = key_table.find(key);
        if (it == key_table.end()) {
            if (key_table.size() == capacity) {
                key_table.erase(S.begin()->key);
                S.erase(S.begin());
            }
            Node cache = Node(1, ++time, key, value);
            key_table.insert(make_pair(key, cache));
            S.insert(cache);
        }
        else {
            Node cache = it->second;
            S.erase(cache);
            cache.cnt += 1;
            cache.time = ++time;
            cache.value = value;
            S.insert(cache);
            it->second = cache;
        }
    }
};

/**
 * Your LFUCache object will be instantiated and called as such:
 * LFUCache* obj = new LFUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */
```

**解法二：双哈希表**

双哈希表的方式比起哈希 + AVL 树最大的好处就是将操作的时间复杂度降为了 $$\mathcal O(1)$$

双哈希表一个 freq_table 记录 freq = n 的双向链表，一个 key_table 记录 key = k 的节点在 freq = n  的双向链表中的位置，具体如下

```cpp
unordered_map<int, list<Node>> freq_table;
unordered_map<int, list<Node>::iterator> key_table;
```

<img src="https://imgconvert.csdnimg.cn/aHR0cHM6Ly9waWMubGVldGNvZGUtY24uY29tLzYxNDIzZWVmNzg2M2Q4N2MyNWJjNzczODFlMjk1MDRkOTFhZDM0MjIzNGNhOWI5OTFlMGM5MDQ2Y2EwMzUzM2EtaW1hZ2UtMjAyMDA0MDUyMTQ4MDA0OTUucG5n?x-oss-process=image/format,png" alt="在这里插入图片描述" style="zoom:50%;" />

双哈希表中有两点需要注意的，

- 在双向链表中我们才用头插入的方式来维护一个时间序列
- 对于双向链表的操作，当双向链表为空时要记得释放

```cpp
struct Node {
    int key, val, freq;
    Node(int _key,int _val,int _freq): key(_key), val(_val), freq(_freq){}
};

class LFUCache {
private:
    int minfreq, capacity;
    unordered_map<int, list<Node>> freq_table;
    unordered_map<int, list<Node>::iterator> key_table;
public:
    LFUCache(int _capacity) {
        minfreq = 0;
        capacity = _capacity;
        key_table.clear();
        freq_table.clear();
    }
    
    int get(int key) {
        if (capacity == 0) return -1;
        auto it = key_table.find(key);
        if (it == key_table.end()) return -1;
        auto node_it = it->second;
        int val = node_it->val, freq = node_it->freq;
        freq_table[freq].erase(node_it);
        if (freq_table[freq].size() == 0) {
            freq_table.erase(freq);
            if (minfreq == freq) minfreq++;
        }
        freq_table[freq + 1].push_front(Node(key, val, freq + 1));
        key_table[key] = freq_table[freq + 1].begin();
        return val;
    }
    
    void put(int key, int value) {
        if (capacity == 0) return;
        auto it = key_table.find(key);
        if (it == key_table.end()) {
            if (key_table.size() == capacity) {
                auto it2 = freq_table[minfreq].back();
                key_table.erase(it2.key);
                freq_table[minfreq].pop_back();
                if (freq_table[minfreq].size() == 0) {
                    freq_table.erase(minfreq);
                }
            }
            freq_table[1].push_front(Node(key, value, 1));
            key_table[key] = freq_table[1].begin();
            minfreq = 1;
        }
        else {
            auto node_it = it->second;
            int freq = node_it->freq;
            freq_table[freq].erase(node_it);
            if (freq_table[freq].size() == 0) {
                freq_table.erase(freq);
                if (minfreq == freq) minfreq++;
            }
            freq_table[freq + 1].push_front(Node(key, value, freq + 1));
            key_table[key] = freq_table[freq + 1].begin();
        }
    }
};

/**
 * Your LFUCache object will be instantiated and called as such:
 * LFUCache* obj = new LFUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */
```

## 6312. 最小和分割

**题目描述**

[6312. 最小和分割](https://leetcode.cn/problems/split-with-minimum-sum/description/)

**解法**

神奇的地方就在于你给 num 的每个数按升序排列后还不能按着挑，要跳着挑才能组合出最小和

```cpp
class Solution {
public:
    int splitNum(int num) {
        string s = to_string(num);
        sort(s.begin(), s.end());
        int nums[2]{};
        for (int i = 0; i < s.size(); i++) {
            nums[i % 2] = nums[i % 2] * 10 + s[i] - '0';
        }
        return nums[0] + nums[1];
    }
};
```

## 6313. 统计将重叠区间合并成组的方案数

**题目描述**

[6313. 统计将重叠区间合并成组的方案数](https://leetcode.cn/problems/count-ways-to-group-overlapping-ranges/description/)

**解法**

假设我们合并完最终剩下 $$n$$ 个不相交区间分为两组，那么最终的答案就是 $$2^n$$ 

问题在于我们有必要执行合并的操作吗？没必要，将 ranges 按照 start 升序排列，，同时维护区间右端点的最大值 $$maxR$$，如果存在 `ranges[i][0] <= maxR` 那么说明该区间应该合并到上一个区间中，这时候不用更新 $$n$$，只用更新 $$maxR$$

```cpp
class Solution {
public:
    const int mod  = 1e9 +7;
    int countWays(vector<vector<int>>& ranges) {
        sort(ranges.begin(), ranges.end(), [](auto& a, auto& b) { return a[0] < b[0]; } );
        int ans = 2, maxR = ranges[0][1];
        for (int i = 1; i < ranges.size(); i++) {
            if (ranges[i][0] > maxR) ans = ans * 2 % mod;
            maxR = max(maxR, ranges[i][1]);
        }
        return ans;
    }
};
```

## 459. 重复的子字符串

**题目描述**

[459. 重复的子字符串](https://leetcode.cn/problems/repeated-substring-pattern/description/)

**解法**

这道题很巧妙，你想如果一个子串重复 $k$ 次可以构成串 $s$，那么这个字符串 $s$ 是不是有点像循环数组，记 $s=s's's'...s'$，那么移除字符串串 $s$ 前面的若干个 $s'$ 再添加到剩余字符串的尾部刚好又能得到一个完整的 $s$

这时候方法就来了，一个字符串 $s$ 至少由子串重复两次构成，即 $s=s's'$，那么 $2 *s = s's's's'$，移除第一个和最后一个字符，那么至少在剩下的串中应该还能凑出一个 $s$

反过来，如果字符串 $s$ 不能由子串重复构成，那么 $2*s = ss$，此时移除第一个和最后一个字符，那么剩下的串中就凑不出一个完整的 $s$

```cpp
class Solution {
public:
    bool repeatedSubstringPattern(string s) {
        return (s + s).find(s, 1) != s.size();
    }   
};
```

# 二叉树

## 

**题目描述**

[99. 恢复二叉搜索树](https://leetcode.cn/problems/recover-binary-search-tree/description/)

**解法**

中序遍历得到结果，然后在数组中查找可能出现错误的结点

![1.jpg](https://pic.leetcode-cn.com/ceaf09da74f78f235f329dbc588f63da7464590947edb8c0415a4bd9ff493299-1.jpg)

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
private:
    TreeNode* pre = nullptr;
    TreeNode* x  = nullptr;
    TreeNode* y = nullptr;
public:
    void recoverTree(TreeNode* root) {
        inOrderTraverse(root);
        if (x != nullptr && y != nullptr) {
            swap(x->val, y->val);
        }
    }

    void inOrderTraverse(TreeNode * root) {
        if (root == nullptr) return;
        inOrderTraverse(root->left);
        if (pre != nullptr && pre->val > root->val) {
            y = root;
            if (x ==  nullptr) x = pre;
        }
        pre = root;
        inOrderTraverse(root->right);
    }
};
```

