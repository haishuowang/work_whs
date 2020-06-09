class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if k == 1:
            return nums
        res, q = [], []
        for i in range(len(nums)):
            if len(q) > 0 and q[0] < i - k + 1:
                q = q[1:]
            while len(q) > 0 and nums[q[-1]] <= nums[i]:
                q.pop()
            q.append(i)
            if i >= k - 1: res.append(nums[q[0]])
        return res
