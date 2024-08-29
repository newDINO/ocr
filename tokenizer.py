ascii_char_len = 126 - 32 + 1

special_token_ids = {
    "<begin>": ascii_char_len,
    "<eos>": ascii_char_len + 1,
}

special_id_tokens = {value: key for key, value in special_token_ids.items()}

vocab_size = ascii_char_len + len(special_token_ids) # ascii tokens + special_tokens

# the encoder does not encode special tokens
def encode(string):
    return [ord(char) - 32 for char in string]


def decode(idx):
    result = ''
    for index in idx:
        if index >= ascii_char_len:
            result += special_id_tokens[index]
        else:
            result += chr(index + 32)
    return result


if __name__ == "__main__":
    s = "123+abc=ABC[,./+`~]"
    idx = encode(s)
    print(idx)
    idx = [ascii_char_len] + idx + [ascii_char_len + 1]
    decoded = decode(idx)
    print(decoded)