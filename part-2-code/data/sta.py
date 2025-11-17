from transformers import T5TokenizerFast
import numpy as np

# ------------ CONFIG -----------------
train_nl_path = "train.nl"
train_sql_path = "train.sql"
dev_nl_path = "dev.nl"
dev_sql_path = "dev.sql"
# -------------------------------------

# Use google-t5/t5-small
tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

def read_lines(path):
    return [line.strip() for line in open(path, encoding="utf-8").readlines()]

def compute_stats(nl_file, sql_file):
    nl = read_lines(nl_file)
    sql = read_lines(sql_file)

    assert len(nl) == len(sql), f"Mismatch: {nl_file} and {sql_file}"

    # Tokenize using T5 tokenizer
    nl_tokens = tokenizer(nl, add_special_tokens=False).input_ids
    sql_tokens = tokenizer(sql, add_special_tokens=False).input_ids

    # Sentence lengths
    nl_lengths = [len(toks) for toks in nl_tokens]
    sql_lengths = [len(toks) for toks in sql_tokens]

    # Vocabulary: number of unique token IDs appearing
    nl_vocab = set([tid for seq in nl_tokens for tid in seq])
    sql_vocab = set([tid for seq in sql_tokens for tid in seq])

    stats = {
        "num_examples": len(nl),
        "mean_nl_length": float(np.mean(nl_lengths)),
        "mean_sql_length": float(np.mean(sql_lengths)),
        "nl_vocab_size": len(nl_vocab),
        "sql_vocab_size": len(sql_vocab),
    }

    return stats


# ----- Compute for train and dev -----
train_stats = compute_stats(train_nl_path, train_sql_path)
dev_stats = compute_stats(dev_nl_path, dev_sql_path)

print("\n===== TRAIN SET STATISTICS =====")
for k, v in train_stats.items():
    print(f"{k}: {v}")

print("\n===== DEV SET STATISTICS =====")
for k, v in dev_stats.items():
    print(f"{k}: {v}")

print("Tokenizer vocab size:", len(tokenizer))
print("Tokenizer path:", tokenizer.name_or_path)

