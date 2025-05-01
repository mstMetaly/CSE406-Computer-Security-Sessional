import socket
import random
import pickle
from ecc_2005110 import generate_curve_params, scalar_mult, generate_point_on_curve_fast
from aes_2005110 import send_receive

HOST = '127.0.0.1'
PORT = 65432

# === ECC Key Exchange ===
P, a, b = generate_curve_params(128)
G = generate_point_on_curve_fast(a, b, P)
Ka = random.randint(1, P)
A = scalar_mult(Ka, G, a, P)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(pickle.dumps((P, a, b, G, A)))

    B = pickle.loads(s.recv(4096))
    R = scalar_mult(Ka, B, a, P)
    shared_key = str(R[0])  # Convert x-coordinate to string key

    print("[+] Secure connection established. Type 'exit' to quit.")

    while True:
        msg = input("Alice (you): ")
        if msg.lower() == "exit":
            break

        ciphertext_bv = send_receive(input_plaintext=msg, input_key=shared_key, role="Alice")
        print("Alice (cipheredtext In HEX):", ciphertext_bv.get_bitvector_in_hex().upper())
        s.sendall(pickle.dumps(ciphertext_bv))
        print("[+] Encrypted message sent.")
