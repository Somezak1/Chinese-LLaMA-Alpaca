import logging
import os
from dataclasses import dataclass
from typing import Dict, Sequence, Union, List
import datasets
import torch
from datasets import load_dataset, concatenate_datasets
import transformers


IGNORE_INDEX = -100

logger = logging.getLogger('__name__')

PROMPT_TEMPLATE = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    )

def build_instruction_dataset(data_path: Union[List[str],str],
                tokenizer: transformers.PreTrainedTokenizer,
                max_seq_length: int, data_cache_dir = None,
                preprocessing_num_workers = None,
                ):

    def tokenization(examples):
        sources = []
        targets = []
        prompt = PROMPT_TEMPLATE
        for instruction, input, output in zip(examples['instruction'],examples['input'],examples['output']):
            if input is not None and input !="":
                instruction = instruction+'\n'+input
            source = prompt.format_map({'instruction':instruction})
            target = f"{output}{tokenizer.eos_token}"  # tokenizer.eos_token: '</s>'

            sources.append(source)
            targets.append(target)

        tokenized_sources = tokenizer(sources,return_attention_mask=False)
        tokenized_targets = tokenizer(targets,return_attention_mask=False,add_special_tokens=False)

        all_input_ids = []
        all_labels = []
        for s,t in zip(tokenized_sources['input_ids'],tokenized_targets['input_ids']):
            input_ids = torch.LongTensor(s + t)[:max_seq_length]
            labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:max_seq_length]
            assert len(input_ids) == len(labels)
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        results = {'input_ids':all_input_ids, 'labels': all_labels}
        return results


    logging.warning("building dataset...")
    all_datasets = []

    if not isinstance(data_path,(list,tuple)):
        data_path = [data_path]
    for file in data_path:
        # file: '../data/alpaca_data_zh_51k.json'
        if data_cache_dir is None:
            data_cache_dir = str(os.path.dirname(file))  # data_cache_dir: '../data'
        cache_path = os.path.join(data_cache_dir,os.path.basename(file).split('.')[0])
        # cache_path: '../data/alpaca_data_zh_51k'
        os.makedirs(cache_path, exist_ok=True)
        try:
            processed_dataset = datasets.load_from_disk(cache_path)
            # datasets.load_from_disk: Loads a dataset that was previously saved using save_to_disk()
            # from a dataset directory, or from a filesystem using either S3FileSystem or any
            # implementation of fsspec.spec.AbstractFileSystem.
            logger.info(f'training datasets-{file} has been loaded from disk')
        except Exception:
            raw_dataset = load_dataset("json", data_files=file, cache_dir=cache_path)
            # raw_dataset:
            # DatasetDict({
            #     train: Dataset({
            #         features: ['instruction', 'output', 'input'],
            #         num_rows: 51179
            #     })
            # })

            # raw_dataset中的某个数据:
            #   'instruction': '编辑文章，使其更吸引读者。',
            #   'input': '自主机器人是计算机控制的机器，被编程执行特定任务而不需要任何人类输入。
            #             自主机器人在各个行业中被越来越广泛地应用，从制造业到医疗保健再到安全。'
            #   'output': '自主机器人是计算机控制的机器，被编程执行特定任务而不需要任何人类输入，
            #              从而实现了新的效率、精确度和可靠性水平。自主机器人在各个行业中被越来
            #              越广泛地应用，从制造业，它们可以使用精度和一致的质量组装复杂的组件，
            #              到医疗保健，可以协助进行医疗测试和处理，再到安全，可以监控大面积地区，
            #              保障人们和财产的安全。自主机器人还可以减少在危险或有害环境中的错误和
            #              增加安全，在工业流程的检查或维修期间等。由于其多样性，自主机器人将彻
            #              底改变我们工作方式的方式，使任务变得更加简单、快速，最终更加愉悦。'

            tokenization_func = tokenization
            tokenized_dataset = raw_dataset.map(
                tokenization_func,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=["instruction","input","output"],
                keep_in_memory=False,
                desc="preprocessing on dataset",
            )
            processed_dataset = tokenized_dataset
            processed_dataset.save_to_disk(cache_path)  # Saves a dataset to a dataset directory
            # processed_dataset:
            # DatasetDict({
            #     train: Dataset({
            #         features: ['input_ids', 'labels'],
            #         num_rows: 51179
            #     })
            # })

            # tokens = tokenizer.convert_ids_to_tokens(processed_dataset['train'][1]['input_ids'])
            # for idx, (i, j) in enumerate(zip(tokens, processed_dataset['train'][1]['labels'])):
            #     print(f"idx: {idx}   token: {i}   label: {j}")
            # idx: 0   token: <s>   label: -100
            # idx: 1   token: ▁Below   label: -100
            # idx: 2   token: ▁is   label: -100
            # idx: 3   token: ▁an   label: -100
            # idx: 4   token: ▁instruction   label: -100
            # idx: 5   token: ▁that   label: -100
            # idx: 6   token: ▁describes   label: -100
            # idx: 7   token: ▁a   label: -100
            # idx: 8   token: ▁task   label: -100
            # idx: 9   token: .   label: -100
            # idx: 10   token: ▁Write   label: -100
            # idx: 11   token: ▁a   label: -100
            # idx: 12   token: ▁response   label: -100
            # idx: 13   token: ▁that   label: -100
            # idx: 14   token: ▁appropri   label: -100
            # idx: 15   token: ately   label: -100
            # idx: 16   token: ▁comple   label: -100
            # idx: 17   token: tes   label: -100
            # idx: 18   token: ▁the   label: -100
            # idx: 19   token: ▁request   label: -100
            # idx: 20   token: .   label: -100
            # idx: 21   token: <0x0A>   label: -100
            # idx: 22   token: <0x0A>   label: -100
            # idx: 23   token: ##   label: -100
            # idx: 24   token: #   label: -100
            # idx: 25   token: ▁Inst   label: -100
            # idx: 26   token: ruction   label: -100
            # idx: 27   token: :   label: -100
            # idx: 28   token: <0x0A>   label: -100
            # idx: 29   token: 编辑   label: -100
            # idx: 30   token: 文章   label: -100
            # idx: 31   token: ，   label: -100
            # idx: 32   token: 使其   label: -100
            # idx: 33   token: 更   label: -100
            # idx: 34   token: 吸引   label: -100
            # idx: 35   token: 读者   label: -100
            # idx: 36   token: 。   label: -100
            # idx: 37   token: <0x0A>   label: -100
            # idx: 38   token: 自主   label: -100
            # idx: 39   token: 机器人   label: -100
            # idx: 40   token: 是   label: -100
            # idx: 41   token: 计算机   label: -100
            # idx: 42   token: 控制   label: -100
            # idx: 43   token: 的   label: -100
            # idx: 44   token: 机器   label: -100
            # idx: 45   token: ，   label: -100
            # idx: 46   token: 被   label: -100
            # idx: 47   token: 编程   label: -100
            # idx: 48   token: 执行   label: -100
            # idx: 49   token: 特定   label: -100
            # idx: 50   token: 任务   label: -100
            # idx: 51   token: 而不   label: -100
            # idx: 52   token: 需要   label: -100
            # idx: 53   token: 任何人   label: -100
            # idx: 54   token: 类   label: -100
            # idx: 55   token: 输入   label: -100
            # idx: 56   token: 。   label: -100
            # idx: 57   token: 自主   label: -100
            # idx: 58   token: 机器人   label: -100
            # idx: 59   token: 在   label: -100
            # idx: 60   token: 各个   label: -100
            # idx: 61   token: 行业   label: -100
            # idx: 62   token: 中   label: -100
            # idx: 63   token: 被   label: -100
            # idx: 64   token: 越   label: -100
            # idx: 65   token: 来   label: -100
            # idx: 66   token: 越   label: -100
            # idx: 67   token: 广泛   label: -100
            # idx: 68   token: 地   label: -100
            # idx: 69   token: 应用   label: -100
            # idx: 70   token: ，   label: -100
            # idx: 71   token: 从   label: -100
            # idx: 72   token: 制造业   label: -100
            # idx: 73   token: 到   label: -100
            # idx: 74   token: 医疗   label: -100
            # idx: 75   token: 保健   label: -100
            # idx: 76   token: 再   label: -100
            # idx: 77   token: 到   label: -100
            # idx: 78   token: 安全   label: -100
            # idx: 79   token: 。   label: -100
            # idx: 80   token: <0x0A>   label: -100
            # idx: 81   token: <0x0A>   label: -100
            # idx: 82   token: ##   label: -100
            # idx: 83   token: #   label: -100
            # idx: 84   token: ▁Response   label: -100
            # idx: 85   token: :   label: -100
            # idx: 86   token: ▁   label: -100
            # idx: 87   token: ▁自   label: 37291
            # idx: 88   token: 主机   label: 39986
            # idx: 89   token: 器   label: 30943
            # idx: 90   token: 人   label: 30313
            # idx: 91   token: 是   label: 30392
            # idx: 92   token: 计算机   label: 33482
            # idx: 93   token: 控制   label: 32287
            # idx: 94   token: 的   label: 30210
            # idx: 95   token: 机器   label: 35454
            # idx: 96   token: ，   label: 30214
            # idx: 97   token: 被   label: 31407
            # idx: 98   token: 编程   label: 38531
            # idx: 99   token: 执行   label: 33106
            # idx: 100   token: 特定   label: 36444
            # idx: 101   token: 任务   label: 32885
            # idx: 102   token: 而不   label: 35161
            # idx: 103   token: 需要   label: 32054
            # idx: 104   token: 任何人   label: 41114
            # idx: 105   token: 类   label: 30832
            # idx: 106   token: 输入   label: 34485
            # idx: 107   token: ，   label: 30214
            # idx: 108   token: 从而   label: 33241
            # idx: 109   token: 实现了   label: 37721
            # idx: 110   token: 新的   label: 33077
            # idx: 111   token: 效率   label: 34323
            # idx: 112   token: 、   label: 30330
            # idx: 113   token: 精确   label: 40263
            # idx: 114   token: 度   label: 30898
            # idx: 115   token: 和   label: 30503
            # idx: 116   token: 可靠性   label: 45161
            # idx: 117   token: 水平   label: 32387
            # idx: 118   token: 。   label: 30267
            # idx: 119   token: 自主   label: 35364
            # idx: 120   token: 机器人   label: 36880
            # idx: 121   token: 在   label: 30505
            # idx: 122   token: 各个   label: 34350
            # idx: 123   token: 行业   label: 32260
            # idx: 124   token: 中   label: 30275
            # idx: 125   token: 被   label: 31407
            # idx: 126   token: 越   label: 31844
            # idx: 127   token: 来   label: 30805
            # idx: 128   token: 越   label: 31844
            # idx: 129   token: 广泛   label: 33818
            # idx: 130   token: 地   label: 30533
            # idx: 131   token: 应用   label: 32596
            # idx: 132   token: ，   label: 30214
            # idx: 133   token: 从   label: 31594
            # idx: 134   token: 制造业   label: 38254
            # idx: 135   token: ，   label: 30214
            # idx: 136   token: 它们   label: 33392
            # idx: 137   token: 可以   label: 32003
            # idx: 138   token: 使用   label: 32049
            # idx: 139   token: 精度   label: 42186
            # idx: 140   token: 和   label: 30503
            # idx: 141   token: 一致   label: 34199
            # idx: 142   token: 的质量   label: 40793
            # idx: 143   token: 组   label: 31263
            # idx: 144   token: 装   label: 31905
            # idx: 145   token: 复杂的   label: 38848
            # idx: 146   token: 组件   label: 46158
            # idx: 147   token: ，   label: 30214
            # idx: 148   token: 到   label: 30780
            # idx: 149   token: 医疗   label: 33885
            # idx: 150   token: 保健   label: 36836
            # idx: 151   token: ，   label: 30214
            # idx: 152   token: 可以   label: 32003
            # idx: 153   token: 协助   label: 37319
            # idx: 154   token: 进行   label: 32019
            # idx: 155   token: 医疗   label: 33885
            # idx: 156   token: 测试   label: 33628
            # idx: 157   token: 和   label: 30503
            # idx: 158   token: 处理   label: 32391
            # idx: 159   token: ，   label: 30214
            # idx: 160   token: 再   label: 31733
            # idx: 161   token: 到   label: 30780
            # idx: 162   token: 安全   label: 32225
            # idx: 163   token: ，   label: 30214
            # idx: 164   token: 可以   label: 32003
            # idx: 165   token: 监控   label: 36927
            # idx: 166   token: 大   label: 30257
            # idx: 167   token: 面积   label: 32719
            # idx: 168   token: 地区   label: 32178
            # idx: 169   token: ，   label: 30214
            # idx: 170   token: 保障   label: 33579
            # idx: 171   token: 人们   label: 32482
            # idx: 172   token: 和   label: 30503
            # idx: 173   token: 财产   label: 35707
            # idx: 174   token: 的   label: 30210
            # idx: 175   token: 安全   label: 32225
            # idx: 176   token: 。   label: 30267
            # idx: 177   token: 自主   label: 35364
            # idx: 178   token: 机器人   label: 36880
            # idx: 179   token: 还可以   label: 34402
            # idx: 180   token: 减少   label: 32800
            # idx: 181   token: 在   label: 30505
            # idx: 182   token: 危险   label: 34222
            # idx: 183   token: 或   label: 31391
            # idx: 184   token: 有害   label: 39401
            # idx: 185   token: 环境中   label: 42087
            # idx: 186   token: 的   label: 30210
            # idx: 187   token: 错误   label: 33910
            # idx: 188   token: 和   label: 30503
            # idx: 189   token: 增加   label: 32332
            # idx: 190   token: 安全   label: 32225
            # idx: 191   token: ，   label: 30214
            # idx: 192   token: 在   label: 30505
            # idx: 193   token: 工业   label: 32545
            # idx: 194   token: 流程   label: 35641
            # idx: 195   token: 的   label: 30210
            # idx: 196   token: 检查   label: 32752
            # idx: 197   token: 或   label: 31391
            # idx: 198   token: 维修   label: 36682
            # idx: 199   token: 期间   label: 32520
            # idx: 200   token: 等   label: 31184
            # idx: 201   token: 。   label: 30267
            # idx: 202   token: 由于   label: 32196
            # idx: 203   token: 其   label: 31149
            # idx: 204   token: 多样   label: 40893
            # idx: 205   token: 性   label: 30952
            # idx: 206   token: ，   label: 30214
            # idx: 207   token: 自主   label: 35364
            # idx: 208   token: 机器人   label: 36880
            # idx: 209   token: 将   label: 30998
            # idx: 210   token: 彻底   label: 34785
            # idx: 211   token: 改变   label: 32733
            # idx: 212   token: 我们   label: 32007
            # idx: 213   token: 工作   label: 32017
            # idx: 214   token: 方式   label: 32290
            # idx: 215   token: 的方式   label: 33784
            # idx: 216   token: ，   label: 30214
            # idx: 217   token: 使   label: 30785
            # idx: 218   token: 任务   label: 32885
            # idx: 219   token: 变得更   label: 44844
            # idx: 220   token: 加   label: 30666
            # idx: 221   token: 简单   label: 32862
            # idx: 222   token: 、   label: 30330
            # idx: 223   token: 快速   label: 33238
            # idx: 224   token: ，   label: 30214
            # idx: 225   token: 最终   label: 32469
            # idx: 226   token: 更加   label: 32744
            # idx: 227   token: 愉悦   label: 46384
            # idx: 228   token: 。   label: 30267
            # idx: 229   token: </s>   label: 2

        processed_dataset.set_format('torch')  # 将input_ids和labels从List转化为torch.tensor
        all_datasets.append(processed_dataset['train'])
    # Converts a list of Dataset with the same schema into a single Dataset
    all_datasets = concatenate_datasets(all_datasets)
    return all_datasets

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # input_ids: [instance["input_ids"], instance["input_ids"], ..., instance["input_ids"]]
        # labels: [instance["labels"], instance["labels"], ..., instance["labels"]]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),  # 'pad_token': '[PAD]'
        )
