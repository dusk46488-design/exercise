def ipo_data_process(user_input: str) -> str:
    """
    IPO Model Functional Processing Function
    :param user_input: Input original text content
    :return: Return processed formatted result
    Input: Receive user plain text content
    Process: Remove redundant spaces, count valid character length
    Output: Integrate information and output standardized results
    """
    # Data cleaning processing
    clean_content = user_input.strip()
    # Data statistics processing
    content_length = len(clean_content)
    # Standard result output
    final_result = f"Processed Content：{clean_content}\nValid Character Number：{content_length}"
    return final_result

if __name__ == "__main__":
    input_info = input("Please enter content：")
    print(ipo_data_process(input_info))