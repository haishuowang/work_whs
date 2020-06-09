class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.keep_list = []
        self.save_dict = {}


    def get(self, key: int) -> int:
        res = self.save_dict.get(key, -1)
        if res != -1:
            self.keep_list.remove(key)
            self.keep_list.append(key)
        return res

    def put(self, key: int, value: int) -> None:
        self.save_dict[key] = value
        if key not in self.keep_list:
            if len(self.keep_list)<self.capacity:
                if key not in self.keep_list:
                    self.keep_list.append(key)
            else:
                del_key = self.keep_list.pop(0)
                self.save_dict.pop(del_key)
                self.keep_list.append(key)
        else:
            self.keep_list.remove(key)
            self.keep_list.append(key)


cache = LRUCache(2)
print(cache.put(2, 1))
print(cache.put(2, 2))
print(cache.get(2))
print(cache.put(1, 4))
print(cache.put(4, 1))
print(cache.get(2))
# [[2],[],[2,2],[2],[1,1],[4,1],[2]]