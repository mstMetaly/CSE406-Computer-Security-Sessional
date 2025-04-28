from BitVector import BitVector
from info import Sbox
import copy

Rcon = [BitVector(intVal=x, size=8) for x in
        (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36)]

round_keys = {}  # the caller still sees exactly this global


def key_expansion(key_matrix):
    key_matrix = copy.deepcopy(key_matrix)  # Make a deep copy of the key matrix
    words = []

    # Generate the first four words from the key matrix
    for col in range(4):  # Now iterate over the rows
        word = BitVector(size=0)
        for row in range(4):  # Concatenate each byte correctly for each word
            word += key_matrix[row][col]
        words.append(word)

    # Expanding the key into 44 words (for AES-128)
    for i in range(4, 44):
        temp = words[i-1].deep_copy()  # Copy the last word
        if i % 4 == 0:
            temp = key_schedule_core(temp, Rcon[(i//4)-1])
        
        new_word = words[i-4] ^ temp
        words.append(new_word)
        
    # Generate the round keys from the expanded words
    for round_num in range(11):
        round_key = [[0 for _ in range(4)] for _ in range(4)]
        for col in range(4):
            word = words[round_num * 4 + col]
            for row in range(4):
                # Corrected: Extract the correct byte from each word for the round key
                byte = word[row * 8 : (row + 1) * 8]  # Extract bytes in the correct order (from top to bottom)
                round_key[row][col] = byte
        round_keys[round_num] = round_key

    return round_keys


def key_schedule_core(word, rcon):
    # Rotate the word (circular left shift) 
    for _ in range(8):
        word.circular_rot_left()   
  
    new_word = BitVector(size=0)
    for i in range(0, 32, 8):  # Substitution of each byte using the S-box
        byte = word[i:i + 8]
        sbox_value = Sbox[byte.intValue()]
        sub_byte = BitVector(intVal=sbox_value, size=8)
        new_word += sub_byte

    # XOR the first byte with the rcon value
    first_byte = new_word[0:8] ^ rcon
    new_word = first_byte + new_word[8:]  # Keep the rest of the word unchanged

    return new_word


