from BitVector import BitVector
from info import Sbox, InvSbox, Mixer, InvMixer
from roundkey import key_expansion, round_keys

# Make Sbox and InvSbox 2D arrays
Sbox = [Sbox[i:i + 16] for i in range(0, len(Sbox), 16)]
InvSbox = [InvSbox[i:i + 16] for i in range(0, len(InvSbox), 16)]


# Helper function to print the state matrix
def print_matrix_hexvalue(matrix, label="Matrix"):
    print(f"\n{label}:")
    for row in range(4):
        for col in range(4):
            print(matrix[row][col].get_bitvector_in_hex().upper(), end=" ")
        print()


# Convert text to hex
def get_hex_string(input_text):
    bv_text = BitVector(textstring=input_text)
    return bv_text.get_bitvector_in_hex()


# Make a 4x4 matrix from input bv hex
def make_four_by_four_matrix(bv_hex):
    matrix = [[0 for _ in range(4)] for _ in range(4)]
    for col in range(4):
        for row in range(4):
            byte = bv_hex[(col * 32) + (row * 8): (col * 32) + (row + 1) * 8]
            matrix[row][col] = byte
    return matrix


def matrix_to_bitvector(matrix):
    bv = BitVector(size=0)
    for col in range(4):
        for row in range(4):
            bv += matrix[row][col]
    return bv


# Add round key
def add_round_key(state_matrix, round_key):
    for row in range(4):
        for col in range(4):
            state_matrix[row][col] ^= round_key[row][col]
    return state_matrix


# Substitute bytes using Sbox
def sub_bytes(state_matrix):
    for i in range(4):
        for j in range(4):
            byte_bv = state_matrix[i][j]
            row = byte_bv[0:4].intValue()
            col = byte_bv[4:8].intValue()
            state_matrix[i][j] = BitVector(intVal=Sbox[row][col], size=8)
    return state_matrix


# Inverse substitute bytes using IInvSbox
def inverse_sub_bytes(state_matrix):
    for i in range(4):
        for j in range(4):
            byte_bv = state_matrix[i][j]
            row = byte_bv[0:4].intValue()
            col = byte_bv[4:8].intValue()
            state_matrix[i][j] = BitVector(intVal=InvSbox[row][col], size=8)
    return state_matrix


# Shift rows
def shift_rows(state_matrix):
    for i in range(1, 4):
        state_matrix[i] = state_matrix[i][i:] + state_matrix[i][:i]
    return state_matrix


def inverse_shift_rows(state_matrix):
    for i in range(1, 4):
        state_matrix[i] = state_matrix[i][-i:] + state_matrix[i][:-i]
    return state_matrix


# Mix columns
def mix_columns(state_matrix):
    modulus = BitVector(bitstring='100011011')
    new_matrix = [[BitVector(size=8) for _ in range(4)] for _ in range(4)]
    for col in range(4):
        column = [state_matrix[row][col] for row in range(4)]
        for row in range(4):
            value = BitVector(size=8)
            for k in range(4):
                if Mixer[row][k].intValue() == 1:
                    product = column[k]
                elif Mixer[row][k].intValue() == 2:
                    product = column[k].gf_multiply_modular(BitVector(intVal=2, size=8), modulus, 8)
                elif Mixer[row][k].intValue() == 3:
                    product = column[k].gf_multiply_modular(BitVector(intVal=3, size=8), modulus, 8)
                value ^= product
            new_matrix[row][col] = value
    return new_matrix


# inverse mix columns
def inverse_mix_columns(state_matrix):
    modulus = BitVector(bitstring='100011011')
    new_matrix = [[BitVector(size=8) for _ in range(4)] for _ in range(4)]
    for col in range(4):
        column = [state_matrix[row][col] for row in range(4)]
        for row in range(4):
            value = BitVector(size=8)
            for k in range(4):
                coeff = InvMixer[row][k].intValue()
                if coeff == 1:
                    product = column[k]
                else:
                    product = column[k].gf_multiply_modular(BitVector(intVal=coeff, size=8), modulus, 8)
                value ^= product
            new_matrix[row][col] = value
    return new_matrix


# AES encryption algorithm
def aes_encryption_algo(key_matrix, state_matrix):
    # Initial round key addition
    state_matrix = add_round_key(state_matrix, key_matrix)

    key_expansion(key_matrix)  # Key expansion to generate round keys

    # round 1 to 9
    for round in range(1, 10):  # 10 rounds for AES-128
        # SubBytes
        state_matrix = sub_bytes(state_matrix)
        # ShiftRows
        state_matrix = shift_rows(state_matrix)
        # MixColumns
        state_matrix = mix_columns(state_matrix)
        # Add round key
        round_key = round_keys[round]
        state_matrix = add_round_key(state_matrix, round_key)

    # final round (without MixColumns)
    # SubBytes
    state_matrix = sub_bytes(state_matrix)
    # shift rows
    state_matrix = shift_rows(state_matrix)
    # add round key
    round_key = round_keys[10]
    state_matrix = add_round_key(state_matrix, round_key)

    return state_matrix


def aes_decryption_algo(state_matrix, key_matrix):
    key_expansion(key_matrix)  # Key expansion to generate round keys
    # initial round key addition
    round_key = round_keys[10]
    state_matrix = add_round_key(state_matrix, round_key)

    # for loop 1 to 9 rounds
    for round in range(9, 0, -1):
        # inverse shift rows
        state_matrix = inverse_shift_rows(state_matrix)
        # inverse sub bytes
        state_matrix = inverse_sub_bytes(state_matrix)
        # add round key
        round_key = round_keys[round]
        state_matrix = add_round_key(state_matrix, round_key)
        # inverse mix columns
        state_matrix = inverse_mix_columns(state_matrix)

    # final round (without inverse mix columns)
    # inverse shift rows
    state_matrix = inverse_shift_rows(state_matrix)
    # inverse sub bytes
    state_matrix = inverse_sub_bytes(state_matrix)
    # add round key
    round_key = round_keys[0]
    state_matrix = add_round_key(state_matrix, round_key)

    return state_matrix


def remove_pkcs7_padding(bv):
    last_byte = bv[-8:]  # Last 8 bits
    pad_value = last_byte.intValue()
    if pad_value <= 0 or pad_value > 16:
        raise ValueError("Invalid PKCS#7 padding.")
    total_pad_bits = pad_value * 8
    return bv[:-total_pad_bits]


def pkcs7_padding(plaintext_bv):
    original_length = plaintext_bv.length()
    pad_len = (128 - (original_length % 128)) // 8
    if pad_len == 0:
        pad_len = 16

    pad_byte = BitVector(intVal=pad_len, size=8)
    padding = BitVector(size=0)
    for _ in range(pad_len):
        padding += pad_byte

    plaintext_bv += padding
    return plaintext_bv


def cbc_decryption(full_cipheredtext_bv, key_matrix, iv_matrix):
    decrypted_bv = BitVector(size=0)
    total_bits = full_cipheredtext_bv.length()
    block_size = 128

    prev_cipher = iv_matrix  # storing previous cipher as matrix

    for i in range(0, total_bits, block_size):
        cipher_block = full_cipheredtext_bv[i:i + block_size]
        cipher_block = BitVector(hexstring=cipher_block.get_bitvector_in_hex())
        state_matrix = make_four_by_four_matrix(cipher_block)

        # AES decryption
        decrypted_matrix = aes_decryption_algo(state_matrix, key_matrix)
        # XOR with previous cipher block (or IV for the first block)
        for row in range(4):
            for col in range(4):
                decrypted_matrix[row][col] ^= prev_cipher[row][col]

        decrypted_bv += matrix_to_bitvector(decrypted_matrix)
        prev_cipher = make_four_by_four_matrix(cipher_block)

    return decrypted_bv


def cbc_encryption(plaintext_bv, key_matrix, iv_matrix):
    # CBC mode of operation
    # Initialize the IV (initialization vector) if not provided
    prev_cipher = iv_matrix  # stroing prev cypher as matrix
    ciphertext_blocks = []

    for i in range(0, plaintext_bv.length(), 128):
        plaintext_block = plaintext_bv[i:i + 128]
        state_matrix = make_four_by_four_matrix(plaintext_block)

        # CBC initial step XOR with IV
        for row in range(4):
            for col in range(4):
                state_matrix[row][col] ^= prev_cipher[row][col]

        # AES encryption
        encrypted_matrix = aes_encryption_algo(key_matrix, state_matrix)
        cipher_block_bv = matrix_to_bitvector(encrypted_matrix)
        ciphertext_blocks.append(cipher_block_bv)
        prev_cipher = encrypted_matrix

    # combine ciphertext blocks into a single Bblocks
    full_ciphertext_bv = BitVector(size=0)
    for block in ciphertext_blocks:
        full_ciphertext_bv += block

    return full_ciphertext_bv


# alice input process
def send_receive(input_plaintext=None, input_key=None, role=None):
    input_key = input_key
    key_bv = BitVector(textstring=input_key)

    # key length handling
    if key_bv.length() < 128:
        key_bv += BitVector(size=128 - key_bv.length())
    elif key_bv.length() > 128:
        key_bv = key_bv[0:128]

    key_matrix = make_four_by_four_matrix(key_bv)
    iv = BitVector(intVal=0, size=128)
    iv_matrix = make_four_by_four_matrix(iv)

    if role == "Alice":
        input_plaintext = input_plaintext
        plaintext_bv = BitVector(textstring=input_plaintext)

        # padding using PKCS7 padding
        plaintext_bv = pkcs7_padding(plaintext_bv)

        full_ciphertext_bv = cbc_encryption(plaintext_bv, key_matrix, iv_matrix)

        return full_ciphertext_bv

    else:
        decrypted_bv = cbc_decryption(input_plaintext, key_matrix, iv_matrix)
        decrypted_bv = remove_pkcs7_padding(decrypted_bv)
        decrypted_text = decrypted_bv.get_text_from_bitvector()

        return decrypted_text


# Input and processing function
def input_process():
    input_key = input("Key: ")
    key_bv = BitVector(textstring=input_key)
    print("In ASCII: ", input_key)
    print("In hex: ", key_bv.get_bitvector_in_hex().upper())

    input_plaintext = input("Plain Text: ")
    plaintext_bv = BitVector(textstring=input_plaintext)
    print("In ASCII: ", input_plaintext)
    print("In hex: ", plaintext_bv.get_bitvector_in_hex().upper())

    # key length handling
    if key_bv.length() < 128:
        key_bv += BitVector(size=128 - key_bv.length())
    elif key_bv.length() > 128:
        key_bv = key_bv[0:128]

    key_matrix = make_four_by_four_matrix(key_bv)
    iv = BitVector(intVal=0, size=128)
    iv_matrix = make_four_by_four_matrix(iv)

    # padding using PKCS7 padding
    plaintext_bv = pkcs7_padding(plaintext_bv)
    print("In ASCII (After Padding): ", plaintext_bv.get_text_from_bitvector())
    print("In hex (After Padding): ", plaintext_bv.get_bitvector_in_hex().upper())

    full_ciphertext_bv = cbc_encryption(plaintext_bv, key_matrix, iv_matrix)
    print("Ciphered Text:")
    print("In Hex: ", full_ciphertext_bv.get_bitvector_in_hex().upper())
    print("In ASCII: ", full_ciphertext_bv.get_text_from_bitvector())

    decrypted_bv = cbc_decryption(full_ciphertext_bv,key_matrix, iv_matrix)
    print("Decrypted Text:")
    print("Before Unpadding:")
    print("In Hex)", decrypted_bv.get_bitvector_in_hex().upper())
    print("In ASCII", decrypted_bv.get_text_from_bitvector())

    # After CBC decryption, remove PKCS#7 padding
    decrypted_bv = remove_pkcs7_padding(decrypted_bv)
    print("After Unpadding:")
    print("In ASCII)", decrypted_bv.get_text_from_bitvector())
    print("In Hex", decrypted_bv.get_bitvector_in_hex().upper())


# Main function
def main():
    input_process()


if __name__ == "__main__":
    main()
