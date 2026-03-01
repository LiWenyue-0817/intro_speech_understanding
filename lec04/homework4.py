def next_birthday(date, birthdays):
    '''
    Find the next birthday after the given date.

    @param:
    date - a tuple of two integers specifying (month, day)
    birthdays - a dict mapping from date tuples to lists of names, for example,
      birthdays[(1,10)] = list of all people with birthdays on January 10.

    @return:
    birthday - the next day, after given date, on which somebody has a birthday
    list_of_names - list of all people with birthdays on that date
    '''
    # 提取所有生日日期并排序（按月份→日期）
    birthday_dates = sorted(birthdays.keys())
    if not birthday_dates:  # 无生日数据的情况
        return (1, 1), []
    
    # 定义辅助函数：判断date1是否在date2之后
    def is_after(date1, date2):
        m1, d1 = date1
        m2, d2 = date2
        if m1 > m2:
            return True
        elif m1 == m2 and d1 > d2:
            return True
        return False
    
    # 寻找第一个在给定日期之后的生日
    next_date = None
    for bd in birthday_dates:
        if is_after(bd, date):
            next_date = bd
            break
    
    # 如果全年都没有（比如给定日期是12月31日），取明年第一个生日
    if next_date is None:
        next_date = birthday_dates[0]
    
    return next_date, birthdays[next_date]


# 测试代码
if __name__ == "__main__":
    # 构建测试用的生日字典
    test_birthdays = {
        (1, 10): ["Alice"],
        (5, 20): ["Bob", "Charlie"],
        (12, 25): ["David"]
    }

    # 测试场景1：给定日期为(3, 15)，预期下一个生日是5月20日
    test_date1 = (3, 15)
    bd1, names1 = next_birthday(test_date1, test_birthdays)
    print(f"测试1 - 给定日期{test_date1}，下一个生日：{bd1}，人员：{names1}")

    # 测试场景2：给定日期为(11, 1)，预期下一个生日是12月25日
    test_date2 = (11, 1)
    bd2, names2 = next_birthday(test_date2, test_birthdays)
    print(f"测试2 - 给定日期{test_date2}，下一个生日：{bd2}，人员：{names2}")

    # 测试场景3：给定日期为(12, 30)，预期下一个生日是1月10日（次年）
    test_date3 = (12, 30)
    bd3, names3 = next_birthday(test_date3, test_birthdays)
    print(f"测试3 - 给定日期{test_date3}，下一个生日：{bd3}，人员：{names3}")

    # 测试场景4：空生日字典
    test_date4 = (5, 5)
    bd4, names4 = next_birthday(test_date4, {})
    print(f"测试4 - 给定日期{test_date4}（空字典），下一个生日：{bd4}，人员：{names4}")
