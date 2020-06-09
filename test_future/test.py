class Solution(object):
    # 15
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        res = []
        i = 0
        for i in range(len(nums)):
            if i == 0 or nums[i] > nums[i - 1]:
                l = i + 1
                r = len(nums) - 1
                while l < r:
                    s = nums[i] + nums[l] + nums[r]
                    if s == 0:
                        res.append([nums[i], nums[l], nums[r]])
                        l += 1
                        r -= 1
                        while l < r and nums[l] == nums[l - 1]:
                            l += 1
                        while r > l and nums[r] == nums[r + 1]:
                            r -= 1
                    elif s > 0:
                        r -= 1
                    else:
                        l += 1
        return res

    # 23
    class Solution:
        def mergeKLists(self, lists):
            if not lists: return
            n = len(lists)
            return self.merge(lists, 0, n - 1)

        def merge(self, lists, left, right):
            if left == right:
                return lists[left]
            mid = left + (right - left) // 2
            l1 = self.merge(lists, left, mid)
            l2 = self.merge(lists, mid + 1, right)
            return self.mergeTwoLists(l1, l2)

        def mergeTwoLists(self, l1, l2):
            if not l1: return l2
            if not l2: return l1
            if l1.val < l2.val:
                l1.next = self.mergeTwoLists(l1.next, l2)
                return l1
            else:
                l2.next = self.mergeTwoLists(l1, l2.next)
                return l2

    # 34
    def searchRange(self, nums, target):
        begin_loc = -1
        end_loc = -1
        for i in range(len(nums)):
            if begin_loc == -1:
                if target == nums[i]:
                    begin_loc = i
                    end_loc = i
            else:
                if target == nums[i]:
                    end_loc = i
                else:
                    return [begin_loc, end_loc]
        return [begin_loc, end_loc]

    # 37
    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """

        def check(x, y, s):
            for i in range(9):
                if board[i][y] == s or board[x][i] == s:
                    return False
            for i in [0, 1, 2]:
                for j in [0, 1, 2]:
                    if board[x // 3 * 3 + i][y // 3 * 3 + j] == s:
                        return False
            return True

        def bt(cur):
            if cur == 81:
                return True
            x, y = cur // 9, cur % 9
            if board[x][y] != '.':
                return bt(cur + 1)
            for i in range(1, 10):
                s = str(i)
                if check(x, y, s):
                    board[x][y] = s
                    if bt(cur + 1):
                        return True
                    board[x][y] = '.'
            return False

        bt(0)

    # 40
    def combinationSum2(self, candidates, target):
        def fun(candidates, target, result_list, tmp_list):
            for i in range(len(candidates)):
                if candidates[i] == target:
                    if tmp_list + [candidates[i]] not in result_list:
                        result_list.append(tmp_list + [candidates[i]])
                else:
                    remain_target = target - candidates[i]
                    fun(candidates[i + 1:], remain_target, result_list, tmp_list + [candidates[i]])
            return result_list

        candidates.sort(reverse=True)
        return fun(candidates, target, [], [])

    # 42
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        rain_num = 0
        if not height:
            return rain_num
        a_fill = []
        max_now = None
        a_max = max(height)
        i_max = 0
        for i in range(len(height)):
            if height[i] != a_max:
                if max_now is None or max_now <= height[i]:
                    max_now = height[i]
                else:
                    rain_num += max_now - height[i]
            if height[i] == a_max:
                i_max = i
                break

        max_now = None
        for i in range(0, len(height) - i_max):
            if max_now is None or max_now <= height[-i - 1]:
                max_now = height[-i - 1]
            else:
                rain_num += max_now - height[-i - 1]
        return rain_num

    # 45
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) <= 1:
            return 0

        i = 1
        choose_len = nums[0]
        now_len = i + choose_len
        step_num = 1
        while now_len <= len(nums):
            part_a = nums[i: i + choose_len]
            real_part_a = [part_a[x] + x + 1 for x in range(len(part_a))]

            now_len = i + max(real_part_a)
            step_num += 1
            i = i + choose_len
            choose_len = now_len - i
        return step_num

    # 51
    def solveNQueens(self, n):
        def judge_fun(x, tmp_list):
            for i in range(len(tmp_list)):
                if x == (tmp_list[i] + (len(tmp_list) - i)) \
                        or x == (tmp_list[i] - (len(tmp_list) - i)):
                    return False
            return True

        def fun(nums, result_list, tmp_list):
            if nums:
                for i in range(len(nums)):
                    if judge_fun(nums[i], tmp_list):
                        fun(nums[:i] + nums[i + 1:], result_list, tmp_list + [nums[i]])
            else:
                result_list.append(tmp_list)
            return result_list

        return [['.' * num + 'Q' + '.' * (n - num - 1) for num in x] for x in fun(list(range(n)), [], [])]

    # 101
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """

        def main(queue):
            next_queue = []
            now_value_list = []
            for x in queue:
                if x is not None:
                    now_value_list += [x.val]
                    next_queue += [x.left]
                    next_queue += [x.right]
                else:
                    now_value_list += [None]
            print('_____________')
            if not now_value_list == now_value_list[::-1]:
                return False
            elif not next_queue:
                return True
            else:
                return main(next_queue)

        return main([root])

    # 5222
    def balancedStringSplit(self, s):
        """
        :type s: str
        :rtype: int
        """

        def fun(s, result_list=[]):
            tmp_dict = dict({'R': 0, 'L': 0})
            for i in range(int(len(s) / 2)):
                # s[2 * i: 2 * (i + 1)]
                # s[2 * i]
                tmp_dict[s[2 * i]] += 1
                tmp_dict[s[2 * i + 1]] += 1
                if tmp_dict['R'] == tmp_dict['L']:
                    part_s = s[2 * (i + 1):]
                    result_list.append([s[:2 * (i + 1)]])
                    return fun(part_s, result_list)
            return result_list

        return len(fun(s))

    #
    def maxEqualFreq(self, nums):
        d1 = {}  # 记录nums中数字出现次数
        d2 = {}  # 记录出现次数的频次,出现次数为0次的情况需要特殊处理
        maxx = 0
        for idx, val in enumerate(nums):
            # 维护两个字典
            d1[val] = d1.get(val, 0) + 1
            if d1[val] - 1 == 0:
                pass
            elif d2[d1[val] - 1] == 1:
                d2.pop(d1[val] - 1)
            else:
                d2[d1[val] - 1] -= 1
            d2[d1[val]] = d2.get(d1[val], 0) + 1
            # 进行判断
            if len(d2) == 1 and (list(d2.values()) == [1] or list(d2.keys()) == [1]):
                # 仅有一种频次,要么是一个数出现多次,要么是多个数出现一次
                maxx = idx + 1
            elif len(d2) == 2:
                min_freq = min(list(d2.keys()))
                max_freq = max(list(d2.keys()))
                if d2[min_freq] == 1 and min_freq == 1:
                    # 所有数字出现的次数要么是max_freq次要么是min_freq次，
                    # 且d2[min_freq]==1，且min_freq==1
                    # 则删去出现min_freq次的
                    maxx = idx + 1
                elif (max_freq - min_freq == 1) and d2[max_freq] == 1:
                    # 所有数字出现的次数要么是max_freq次要么是min_freq次
                    # 且max_freq-min_freq==1,且出现max_freq次的数字只有一个
                    # 对应比如所有数都出现 2 次，只有一个数出现 3 次，这个时候我们可以删去一个数来保持所有数频次一致。
                    maxx = idx + 1
        return maxx

    # 5223
    def queensAttacktheKing(self, queens, king):
        x_k, y_k = king

        result_list = []
        d1 = dict({1: dict(), -1: dict()})
        d2 = dict({1: dict(), -1: dict()})
        d3 = dict({1: dict(), -1: dict()})
        d4 = dict({1: dict(), -1: dict()})

        for x_q, y_q in queens:
            if x_k == x_q:
                if y_q > y_k:
                    d1[1].update({y_q - y_k: [x_q, y_q]})
                else:
                    d1[-1].update({y_k - y_q: [x_q, y_q]})
            elif y_k == y_q:
                if x_q > x_k:
                    d2[1].update({x_q - x_k: [x_q, y_q]})
                else:
                    d2[-1].update({x_k - x_q: [x_q, y_q]})

            elif x_q - x_k == y_q - y_k:
                if x_q > x_k:
                    d3[1].update({x_q - x_k: [x_q, y_q]})
                else:
                    d3[-1].update({x_k - x_q: [x_q, y_q]})

            elif -x_q + x_k == y_q - y_k:
                if x_q > x_k:
                    d4[1].update({x_q - x_k: [x_q, y_q]})
                else:
                    d4[-1].update({x_k - x_q: [x_q, y_q]})

        for target_dict in [d1, d2, d3, d4]:
            if target_dict[1]:
                result_list.append(target_dict[1][min(target_dict[1].keys())])
            if target_dict[-1]:
                result_list.append(target_dict[-1][min(target_dict[-1].keys())])
        return result_list

    def checkStraightLine(self, coordinates):
        if len(coordinates) < 2:
            return True
        x1, y1 = coordinates[0]
        x2, y2 = coordinates[1]

        def judge_fun(x, y):
            return (x - x1) * (y2 - y1) != (y - y1) * (x2 - x1)

        for tmp_list in coordinates:
            if judge_fun(*tmp_list):
                return False
        return True