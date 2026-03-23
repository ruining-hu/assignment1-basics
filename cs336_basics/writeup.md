# Byte-Pair Encoding (BPE) Tokenizer

## The Unicode Standard

1.

(a) chr(0) returns '\x00'.

(b) Printed representation is '', and string representation is '\x00'.

(c) It appears as its string representation in text.

2.

(a) UTF-8 returns the shortest byte sequence, and is more dense (each byte is used more). In UTF-16, UTF-32, there's a lot more zeros, so lower information density and longer sequences.

(b) This function assumes that each character is represented by one byte in UTF-8. But this is not true. Running this function with test_string from above give an error.

(c) Simply taking the last two bytes in the UTF-8 encoding of the test string gives such an example. The bytes are [175, 33] and trying to decode gives
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xaf in position 0: invalid start byte

