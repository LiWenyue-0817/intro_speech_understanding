def words2characters(words):
    """
    This function converts a list of words into a list of characters.

    @param:
    words - a list of words

    @return:
    characters - a list of characters

    Every element of "words" should be converted to a str, then split into
    characters, each of which is separately appended to "characters." For 
    example, if words==['hello', 1.234, True], then characters should be
    ['h', 'e', 'l', 'l', 'o', '1', '.', '2', '3', '4', 'T', 'r', 'u', 'e']
    """
    characters = []
    # 遍历输入列表中的每个元素
    for item in words:
        # 将元素转换为字符串
        str_item = str(item)
        # 遍历字符串的每个字符，添加到结果列表
        for char in str_item:
            characters.append(char)
    return characters

# 测试函数（运行示例）
if __name__ == "__main__":
    # 示例输入
    test_input = ['hello', 1.234, True]
    # 调用函数
    result = words2characters(test_input)
    # 打印结果
    print("输入:", test_input)
    print("输出:", result)
    # 验证结果是否符合预期
    expected = ['h', 'e', 'l', 'l', 'o', '1', '.', '2', '3', '4', 'T', 'r', 'u', 'e']
    print("是否符合预期:", result == expected)
