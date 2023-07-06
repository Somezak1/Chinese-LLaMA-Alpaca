import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import argparse

# 调试此程序时的代码如下
# python merge_tokenizers.py \
#    --llama_tokenizer_dir /data/model_weights/llama-7b-hf-yahma \
#    --chinese_sp_model_file chinese_sp.model

parser = argparse.ArgumentParser()
parser.add_argument('--llama_tokenizer_dir', default=None, type=str, required=True)
parser.add_argument('--chinese_sp_model_file', default='./chinese_sp.model', type=str)
args = parser.parse_args()

llama_tokenizer_dir = args.llama_tokenizer_dir
chinese_sp_model_file = args.chinese_sp_model_file

# load
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
chinese_sp_model = spm.SentencePieceProcessor()
chinese_sp_model.Load(chinese_sp_model_file)

llama_spm = sp_pb2_model.ModelProto()
llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
chinese_spm = sp_pb2_model.ModelProto()
chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())

# print number of tokens
print(len(llama_tokenizer),len(chinese_sp_model))  # 32000 20000
print(llama_tokenizer.all_special_tokens)  # ['<s>', '</s>', '<unk>']
print(llama_tokenizer.all_special_ids)  # [1, 2, 0]
print(llama_tokenizer.special_tokens_map)  # {'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>'}

## Add Chinese tokens to LLaMA tokenizer
llama_spm_tokens_set=set(p.piece for p in llama_spm.pieces)
print(len(llama_spm_tokens_set))  # 32000
print(f"Before:{len(llama_spm_tokens_set)}")  # 32000
for p in chinese_spm.pieces:
    piece = p.piece
    if piece not in llama_spm_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        llama_spm.pieces.append(new_p)
print(f"New model pieces: {len(llama_spm.pieces)}")  # 49953

## Save
output_sp_dir = 'merged_tokenizer_sp'
output_hf_dir = 'merged_tokenizer_hf' # the path to save Chinese-LLaMA tokenizer
os.makedirs(output_sp_dir,exist_ok=True)
with open(output_sp_dir+'/chinese_llama.model', 'wb') as f:
    f.write(llama_spm.SerializeToString())
tokenizer = LlamaTokenizer(vocab_file=output_sp_dir+'/chinese_llama.model')

tokenizer.save_pretrained(output_hf_dir)
print(f"Chinese-LLaMA tokenizer has been saved to {output_hf_dir}")


# Test
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
chinese_llama_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)
print(llama_tokenizer.all_special_tokens)  # ['<s>', '</s>', '<unk>']
print(llama_tokenizer.all_special_ids)  # [1, 2, 0]
print(llama_tokenizer.special_tokens_map)  # {'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>'}

print(chinese_llama_tokenizer.all_special_tokens)  # ['<s>', '</s>', '<unk>']
print(chinese_llama_tokenizer.all_special_ids)  # [1, 2, 0]
print(chinese_llama_tokenizer.special_tokens_map)  # {'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>'}

text='''白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
The primary use of LLaMA is research on large language models, including'''
print("Test text:\n",text)
print(f"Tokenized by LLaMA tokenizer:{llama_tokenizer.tokenize(text)}")
# ['▁', '白', '日', '<0xE4>', '<0xBE>', '<0x9D>', '山', '<0xE5>', '<0xB0>', '<0xBD>', '，', '黄', '河', '入', '海', '流',
# '。', '<0xE6>', '<0xAC>', '<0xB2>', '<0xE7>', '<0xA9>', '<0xB7>', '千', '里', '目', '，', '更', '上', '一', '<0xE5>',
# '<0xB1>', '<0x82>', '<0xE6>', '<0xA5>', '<0xBC>', '。', '<0x0A>', 'The', '▁primary', '▁use', '▁of', '▁L', 'La', 'MA',
# '▁is', '▁research', '▁on', '▁large', '▁language', '▁models', ',', '▁including']
print(f"Tokenized by Chinese-LLaMA tokenizer:{chinese_llama_tokenizer.tokenize(text)}")
# ['▁白', '日', '依', '山', '尽', '，', '黄河', '入', '海', '流', '。', '欲', '穷', '千里', '目', '，', '更', '上', '一层',
# '楼', '。', '<0x0A>', 'The', '▁primary', '▁use', '▁of', '▁L', 'La', 'MA', '▁is', '▁research', '▁on', '▁large',
# '▁language', '▁models', ',', '▁including']