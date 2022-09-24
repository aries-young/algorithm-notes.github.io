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