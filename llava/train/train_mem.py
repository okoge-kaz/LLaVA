import sys
import os

current_path: str = os.getcwd()
sys.path.append(f"{current_path}/")

from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
