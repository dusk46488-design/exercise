def ipo_process(input_text: str) -> str:
    """
    IPO Model function: Input → Process → Output.
    Input:  Raw text string from the user or command line.
    Process:
        1. Strip leading/trailing whitespace
        2. Count the total number of characters
    Output: Formatted result string with cleaned text and its length.
    """
    # Step 1: Clean the input (remove extra spaces)
    cleaned_text = input_text.strip()

    # Step 2: Calculate character count
    char_count = len(cleaned_text)

    # Step 3: Format the output result
    return f"Cleaned Text: {cleaned_text}\nCharacter Count: {char_count}"


# -------------------
# Example usage
if __name__ == "__main__":
    user_input = input("Enter some text: ")
    result = ipo_process(user_input)
    print("\nProcessed Result:")
    print(result)