import socket
import random
import pickle
from ecc_2005110 import scalar_mult
from aes_2005110 import send_receive

HOST = '127.0.0.1'
PORT = 65432

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print("Bob is waiting for Alice...")
    conn, addr = s.accept()

    with conn:
        print(f"Connected to Alice: {addr}")
        P, a, b, G, A = pickle.loads(conn.recv(4096))

        Kb = random.randint(1, P)
        B = scalar_mult(Kb, G, a, P)
        conn.sendall(pickle.dumps(B))

        R = scalar_mult(Kb, A, a, P)
        shared_key = str(R[0])  # Convert x-coordinate to string key

        print("[+] Secure channel ready. Waiting for encrypted messages...")

        while True:
            try:
                data = conn.recv(4096)
                if not data:
                    print("[!] Connection closed by Alice.")
                    break

                ciphertext_bv = pickle.loads(data)
                print("Bob received cipheredtext: (In HEX)",ciphertext_bv.get_bitvector_in_hex().upper())
                decrypted_text = send_receive(input_plaintext=ciphertext_bv, input_key=shared_key, role="Bob")
                print("In ASCII (after unpadding):", decrypted_text)

            except Exception as e:
                print("[!] Error during decryption:", e)
                break
